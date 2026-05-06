from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"
__all__ = ["Config", "GraphRAG"]


def __getattr__(name: str) -> Any:
    if name == "Config":
        return import_module(".init.config", __name__).Config
    if name == "GraphRAG":
        return import_module(".init.graph_rag", __name__).GraphRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
