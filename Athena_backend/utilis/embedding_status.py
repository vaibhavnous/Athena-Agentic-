from __future__ import annotations

import io
import os
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from typing import Any, Dict

from utilis.azure_embeddings import azure_embeddings_configured, get_azure_embedding_model
from utilis.env import load_backend_env


HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ST_MODEL_NAME = "all-MiniLM-L6-v2"


def _env_enabled(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_embedding_runtime_status(probe_models: bool = True) -> Dict[str, Any]:
    load_backend_env()

    env_enabled = _env_enabled("ATHENA_ENABLE_EMBEDDINGS")
    status: Dict[str, Any] = {
        "env_enabled": env_enabled,
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY")),
        "azure_embedding_configured": azure_embeddings_configured(),
        "sentence_transformer_available": False,
        "langchain_embedding_available": False,
        "azure_embedding_available": False,
        "ready": False,
    }

    if not env_enabled:
        status["reason"] = "Semantic indexing is running in fallback mode"
        return status

    if not probe_models:
        status["ready"] = status["pinecone_configured"] and status["azure_embedding_configured"]
        status["reason"] = (
            "Azure embedding configuration present; semantic model probing is deferred"
            if status["ready"]
            else "Azure embedding or Pinecone configuration is unavailable"
        )
        return status

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = get_azure_embedding_model()
            if model is None:
                raise RuntimeError("Azure embedding configuration is unavailable")
            model.embed_query("athena embedding healthcheck")
        status["azure_embedding_available"] = True
    except Exception as exc:
        status["azure_embedding_error"] = str(exc)

    status["ready"] = (
        status["pinecone_configured"]
        and status["azure_embedding_available"]
    )
    if not status["ready"] and "reason" not in status:
        status["reason"] = "Azure embedding model or Pinecone configuration is unavailable"

    return status


def reset_embedding_runtime_status_cache() -> None:
    get_embedding_runtime_status.cache_clear()
