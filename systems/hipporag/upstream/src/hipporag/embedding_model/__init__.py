from .base import EmbeddingConfig, BaseEmbeddingModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        from .Contriever import ContrieverModel
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        from .OpenAI import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel
    elif embedding_model_name in {"zylonai/bge-m3", "zylonai/bge-m3:latest", "BAAI/bge-m3"}:
        # Allow OpenAI-compatible local embedding servers such as Ollama to host bge-m3.
        from .OpenAI import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        from .Cohere import CohereEmbeddingModel
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        from .Transformers import TransformersEmbeddingModel
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        from .VLLM import VLLMEmbeddingModel
        return VLLMEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
