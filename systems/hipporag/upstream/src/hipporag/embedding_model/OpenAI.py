from copy import deepcopy
import os
import threading
import time
from typing import List, Optional

import httpx
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from openai import OpenAI
from openai import AzureOpenAI
import tiktoken

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


def _pick_embedding_api_key() -> str | None:
    return (
        os.getenv("HIPPORAG_EMBEDDING_API_KEY", "").strip()
        or os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("GPT_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or None
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return float(default)
    return max(0.001, value)


def _make_http_timeout(default_total: float = 5 * 60) -> httpx.Timeout:
    total = _env_float("HIPPORAG_HTTP_TIMEOUT_SEC", default_total)
    connect = _env_float("HIPPORAG_HTTP_CONNECT_TIMEOUT_SEC", min(30.0, total))
    read = _env_float("HIPPORAG_HTTP_READ_TIMEOUT_SEC", total)
    write = _env_float("HIPPORAG_HTTP_WRITE_TIMEOUT_SEC", total)
    pool = _env_float("HIPPORAG_HTTP_POOL_TIMEOUT_SEC", min(30.0, total))
    return httpx.Timeout(total, connect=connect, read=read, write=write, pool=pool)

class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)
        self._usage_lock = threading.Lock()
        self.reset_usage_totals()

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        http_client = httpx.Client(limits=limits, timeout=_make_http_timeout())
        max_retries = int(getattr(self.global_config, "max_retry_attempts", 5) or 5)
        api_key = _pick_embedding_api_key()
        if self.global_config.azure_embedding_endpoint is None:
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.global_config.embedding_base_url,
                http_client=http_client,
                max_retries=max_retries,
            )
        else:
            self.client = AzureOpenAI(
                api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                azure_endpoint=self.global_config.azure_embedding_endpoint,
                http_client=http_client,
                max_retries=max_retries,
            )


    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                'device_map': "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        texts = self._prepare_texts_for_api(texts)
        response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        self._record_usage(prompt_tokens=prompt_tokens, batch_size=len(texts))
        results = np.array([v.embedding for v in response.data])

        return results

    def _embedding_token_limit(self) -> int:
        limit = int(getattr(self.global_config, "embedding_max_seq_len", 0) or 0)
        if limit > 0:
            return limit
        return 8191

    def _get_tokenizer(self):
        try:
            return tiktoken.encoding_for_model(self.embedding_model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _prepare_texts_for_api(self, texts: List[str]) -> List[str]:
        limit = self._embedding_token_limit()
        if limit <= 0:
            return texts

        tokenizer = self._get_tokenizer()
        prepared: List[str] = []
        truncated_count = 0

        for text in texts:
            token_ids = tokenizer.encode(text, disallowed_special=())
            if len(token_ids) > limit:
                truncated_count += 1
                prepared.append(tokenizer.decode(token_ids[:limit]))
            else:
                prepared.append(text)

        if truncated_count:
            logger.warning(
                "Truncated %s/%s overlength texts to %s tokens for embedding model %s.",
                truncated_count,
                len(texts),
                limit,
                self.embedding_model_name,
            )

        return prepared

    def _encode_with_retry(self, texts: List[str]) -> np.ndarray:
        max_retries = int(getattr(self.global_config, "max_retry_attempts", 5) or 5)
        max_retries = max(1, max_retries)
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                return self.encode(texts)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Embedding batch encode failed on attempt %s/%s for batch size %s: %s",
                    attempt,
                    max_retries,
                    len(texts),
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(min(30, 2 ** (attempt - 1)))

        raise RuntimeError(
            f"Embedding batch encode failed after {max_retries} attempts for batch size {len(texts)}"
        ) from last_exc

    def reset_usage_totals(self) -> None:
        with self._usage_lock:
            self._prompt_tokens_total = 0
            self._api_call_count = 0
            self._encoded_text_count = 0

    def _record_usage(self, *, prompt_tokens: int, batch_size: int) -> None:
        with self._usage_lock:
            self._prompt_tokens_total += int(prompt_tokens or 0)
            self._api_call_count += 1
            self._encoded_text_count += int(batch_size or 0)

    def get_usage_totals(self) -> dict:
        with self._usage_lock:
            return {
                "prompt_tokens": int(self._prompt_tokens_total),
                "completion_tokens": 0,
                "total_tokens": int(self._prompt_tokens_total),
                "api_call_count": int(self._api_call_count),
                "encoded_text_count": int(self._encoded_text_count),
            }

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self._encode_with_retry(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results.append(self._encode_with_retry(batch))
                pbar.update(batch_size)
            pbar.close()
            results = np.concatenate(results)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
