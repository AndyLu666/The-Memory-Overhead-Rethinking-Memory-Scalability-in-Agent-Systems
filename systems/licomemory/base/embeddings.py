from typing import List, Dict, Any
import asyncio
import numpy as np
import torch
import sys
import os
import random
import time
from openai import AsyncOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger

class EmbeddingManager:
    """Manager for text embeddings."""

    def __init__(self, config=None):
        """Initialize embedding manager."""
        if config:
            self.api_type = config.api_type
            self.api_key = config.api_key
            self.model_name = config.model
            self.cache_dir = config.cache_dir
            self.dimensions = config.dimensions
            self.max_token_size = config.max_token_size
            self.embed_batch_size = config.embed_batch_size
            self.embedding_func_max_async = config.embedding_func_max_async
            self.base_url = getattr(config, "base_url", "")
        else:
            # Default values
            self.api_type = "openai"
            self.api_key = None
            self.model_name = "text-embedding-ada-002"
            self.cache_dir = None
            self.dimensions = 1536
            self.max_token_size = 8192
            self.embed_batch_size = 100
            self.embedding_func_max_async = 10
            self.base_url = ""

        # Initialize GPU/CUDA device for torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for embedding calculations: {self.device}")
        self.hf_local_files_only = os.getenv("LICOMEMORY_HF_LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes", "on"}
        self.hf_load_retry_attempts = max(1, int(os.getenv("LICOMEMORY_HF_LOAD_RETRY_ATTEMPTS", "3")))
        self.retry_attempts = max(1, int(os.getenv("LICOMEMORY_EMBEDDING_RETRY_ATTEMPTS", "4")))
        self.retry_backoff = max(0.0, float(os.getenv("LICOMEMORY_EMBEDDING_RETRY_BACKOFF", "1.0")))
        self.retry_backoff_max = max(self.retry_backoff, float(os.getenv("LICOMEMORY_EMBEDDING_RETRY_BACKOFF_MAX", "20.0")))

        try:
            if self.api_type == "openai":
                resolved_api_key = (
                    self.api_key
                    or os.getenv("LICOMEMORY_EMBEDDING_API_KEY", "").strip()
                    or os.getenv("OPENAI_API_KEY", "").strip()
                    or os.getenv("GPT_API_KEY", "").strip()
                )
                resolved_base_url = (
                    self.base_url
                    or os.getenv("LICOMEMORY_EMBEDDING_BASE_URL", "").strip()
                    or os.getenv("OPENAI_BASE_URL", "").strip()
                    or os.getenv("GPT_BASE_URL", "").strip()
                    or "https://api.openai.com/v1"
                )
                self.client = AsyncOpenAI(
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    timeout=int(os.getenv("LICOMEMORY_EMBEDDING_TIMEOUT", "120")),
                )
                logger.info(
                    f"OpenAI embedding client initialized with base_url: {resolved_base_url}"
                )
            elif self.api_type == "hf":
                # For HuggingFace models, we might need sentence-transformers
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    msg = (
                        "sentence-transformers is required for api_type=hf. "
                        "Install it in the current environment; fallback is disabled."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg) from e
                # Load model and move to GPU if available.
                # local_files_only avoids HF network churn under high process concurrency.
                load_error = None
                for attempt in range(1, self.hf_load_retry_attempts + 1):
                    try:
                        self.client = SentenceTransformer(
                            self.model_name,
                            cache_folder=self.cache_dir,
                            device=str(self.device),
                            local_files_only=self.hf_local_files_only,
                        )
                        load_error = None
                        break
                    except OSError as e:
                        load_error = e
                        if attempt >= self.hf_load_retry_attempts:
                            raise
                        backoff = min(5.0, 0.5 * attempt) + random.uniform(0, 0.5)
                        logger.warning(
                            f"Failed to load HF embedding model (attempt {attempt}/{self.hf_load_retry_attempts}, "
                            f"local_files_only={self.hf_local_files_only}): {e}. Retry in {backoff:.2f}s"
                        )
                        time.sleep(backoff)
                if load_error is not None:
                    raise RuntimeError(f"Failed to load HF embedding model after retries: {load_error}")
                logger.info(f"HuggingFace model loaded on device: {self.device}")
            else:
                import openai
                self.client = openai

            logger.info(f"Embedding Manager initialized with model: {self.model_name} (type: {self.api_type})")
        except ImportError as e:
            logger.error(f"Failed to initialize embedding client: {e}")
            self.client = None
        except Exception as e:
            logger.error(
                f"Failed to initialize embedding backend (api_type={self.api_type}, model={self.model_name}): {e}. "
                "Falling back to retrieval without embedding backend."
            )
            self.client = None

    async def get_embeddings(self, texts: List[str], need_tensor=False) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if not self.client:
            logger.error("Embedding client not available")
            return []

        try:
            if self.api_type == "hf":
                # HuggingFace model will use GPU automatically if device is cuda
                # convert_to_tensor=True returns torch tensors, which are already on GPU
                embeddings = self.client.encode(texts, convert_to_tensor=True, device=str(self.device))
                # Convert to list for consistency with API
                logger.info(f"self device: {self.device}")
                logger.info(f"embeddings: {embeddings.shape}")
                logger.info(f"embeddings.device: {embeddings.device}")
                if need_tensor:
                    return embeddings
                return embeddings.cpu().tolist() if isinstance(embeddings, torch.Tensor) else embeddings.tolist()
            elif self.api_type == "openai":
                all_embeddings = []
                batch_size = max(1, int(self.embed_batch_size))
                request_kwargs = {}
                if self.dimensions:
                    request_kwargs["dimensions"] = int(self.dimensions)

                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = None
                    last_error = None
                    for attempt in range(1, self.retry_attempts + 1):
                        try:
                            response = await self.client.embeddings.create(
                                model=self.model_name,
                                input=batch,
                                **request_kwargs,
                            )
                            last_error = None
                            break
                        except Exception as e:
                            last_error = e
                            if attempt >= self.retry_attempts:
                                break
                            backoff = min(self.retry_backoff_max, self.retry_backoff * (2 ** (attempt - 1))) + random.uniform(0.0, 0.5)
                            logger.warning(
                                f"Embedding batch failed (attempt {attempt}/{self.retry_attempts}, batch_size={len(batch)}): {e}. "
                                f"Retry in {backoff:.2f}s"
                            )
                            await asyncio.sleep(backoff)
                    if response is None:
                        raise RuntimeError(
                            f"Failed to get embeddings for batch after {self.retry_attempts} attempts: {last_error}"
                        )
                    all_embeddings.extend([data.embedding for data in response.data])

                if need_tensor:
                    return self.transfer_to_tensor(all_embeddings)
                return all_embeddings
            else:
                response = await self.client.embeddings.create(input=texts, model=self.model_name)
                embeddings = [data.embedding for data in response.data]
                if need_tensor:
                    return self.transfer_to_tensor(embeddings)
                return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors using torch on GPU."""
        try:
            # Convert to torch tensors and move to GPU
            v1 = torch.tensor(vec1, dtype=torch.float32, device=self.device)
            v2 = torch.tensor(vec2, dtype=torch.float32, device=self.device)
            
            # Calculate cosine similarity using torch operations
            # cosine_sim = (v1 · v2) / (||v1|| * ||v2||)
            dot_product = torch.dot(v1, v2)
            norm_v1 = torch.linalg.norm(v1)
            norm_v2 = torch.linalg.norm(v2)
            
            similarity = dot_product / (norm_v1 * norm_v2)
            
            # Convert back to Python float
            return similarity.item()
        except Exception as e:
            logger.error(f"Failed to calculate similarity with torch: {e}")
            return 0.0
    
    def cosine_similarity_tensor(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two tensors with shape (n, d) and (m, d) using torch on GPU."""
        try:
            eps = 1e-8
            norm1 = torch.linalg.norm(vec1, dim=1, keepdim=True).clamp_min(eps)
            norm2 = torch.linalg.norm(vec2, dim=1, keepdim=True).clamp_min(eps)
            vec1_normed = vec1 / norm1          
            vec2_normed = vec2 / norm2          
            similarity = torch.matmul(vec1_normed, vec2_normed.T)  
            logger.info(f"similarity shape: {similarity.shape}")
            return similarity
        except Exception as e:
            logger.error(f"Failed to calculate similarity with torch: {e}")
            return 0.0
    

    def transfer_to_tensor(self, embeddings: List[List[float]]) -> torch.Tensor:
        """Transfer a list of embeddings to a tensor on GPU."""
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)
    
    def batch_cosine_similarity(self, query_vec: List[float], candidate_vecs: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarity between one query vector and multiple candidate vectors using torch on GPU.
        This is much more efficient than calling cosine_similarity in a loop.
        
        Args:
            query_vec: Query embedding vector
            candidate_vecs: List of candidate embedding vectors
            
        Returns:
            List of similarity scores
        """
        try:
            # Convert to torch tensors and move to GPU
            query = torch.tensor(query_vec, dtype=torch.float32, device=self.device)
            candidates = torch.tensor(candidate_vecs, dtype=torch.float32, device=self.device)
            
            # Normalize vectors
            query_norm = query / torch.linalg.norm(query)
            candidates_norm = candidates / torch.linalg.norm(candidates, dim=1, keepdim=True)
            
            # Calculate cosine similarity for all candidates at once
            # This is equivalent to: [dot(query, cand) / (||query|| * ||cand||) for cand in candidates]
            similarities = torch.matmul(candidates_norm, query_norm)
            
            # Convert back to Python list
            return similarities.cpu().tolist()
        except Exception as e:
            logger.error(f"Failed to calculate batch similarity with torch: {e}")
            # Fallback to individual calculations
            return [self.cosine_similarity(query_vec, vec) for vec in candidate_vecs]
