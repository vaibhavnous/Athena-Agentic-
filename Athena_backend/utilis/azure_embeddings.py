from __future__ import annotations

import os
from typing import List, Sequence

from utilis.env import load_backend_env


def _env_enabled(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_azure_openai_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    suffix = "/openai/v1"
    if endpoint.lower().endswith(suffix):
        return endpoint[: -len(suffix)].rstrip("/")
    return endpoint


def azure_embeddings_configured() -> bool:
    load_backend_env()
    return bool(
        (
            os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        and (
            os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY")
        )
        and (
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
        )
    )


class EmbeddingVector(list):
    def tolist(self) -> List[float]:
        return list(self)


class AzureOpenAIEmbeddings:
    def __init__(self) -> None:
        load_backend_env()
        from openai import AzureOpenAI

        endpoint = _normalize_azure_openai_endpoint(
            os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "").strip()
            or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        )
        api_key = (
            os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "").strip()
            or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        )
        deployment = (
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()
            or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "").strip()
        )
        api_version = (
            os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "").strip()
            or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
        )

        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_EMBEDDING_ENDPOINT is required")
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_EMBEDDING_API_KEY is required")
        if not deployment:
            raise RuntimeError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required")

        self.deployment = deployment
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.deployment,
            input=str(text or ""),
        )
        return list(response.data[0].embedding)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        values = [str(text or "") for text in texts]
        if not values:
            return []
        response = self.client.embeddings.create(
            model=self.deployment,
            input=values,
        )
        return [list(item.embedding) for item in response.data]

    def encode(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(self.embed_query(text))


def get_azure_embedding_model():
    if not _env_enabled("ATHENA_ENABLE_EMBEDDINGS"):
        return None
    if not azure_embeddings_configured():
        return None
    return AzureOpenAIEmbeddings()
