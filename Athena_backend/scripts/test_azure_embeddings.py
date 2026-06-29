from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from openai import AzureOpenAI

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utilis.env import load_backend_env


def _required_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _resolve_deployment(cli_value: str | None) -> str:
    return (
        (cli_value or "").strip()
        or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    )


def _normalize_azure_openai_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    suffix = "/openai/v1"
    if endpoint.lower().endswith(suffix):
        return endpoint[: -len(suffix)].rstrip("/")
    return endpoint


def _build_client() -> AzureOpenAI:
    endpoint = _normalize_azure_openai_endpoint(
        os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "").strip()
        or _required_env("AZURE_OPENAI_ENDPOINT")
    )
    api_key = (
        os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "").strip()
        or _required_env("AZURE_OPENAI_API_KEY")
    )
    api_version = (
        os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "").strip()
        or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
    )

    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


def _summarize_vector(vector: list[float]) -> Dict[str, Any]:
    return {
        "dimensions": len(vector),
        "sample": vector[:8],
        "min": min(vector) if vector else None,
        "max": max(vector) if vector else None,
    }


def main() -> int:
    load_backend_env()

    parser = argparse.ArgumentParser(description="Test Azure OpenAI embedding deployment connectivity.")
    parser.add_argument(
        "--deployment",
        help="Azure OpenAI embedding deployment name. Falls back to AZURE_OPENAI_EMBEDDING_DEPLOYMENT.",
    )
    parser.add_argument(
        "--text",
        default="semantic matching for table nomination in athena",
        help="Text to embed for the connectivity check.",
    )
    args = parser.parse_args()

    deployment = _resolve_deployment(args.deployment)
    if not deployment:
        raise RuntimeError(
            "No embedding deployment provided. Pass --deployment or set "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
        )

    client = _build_client()
    response = client.embeddings.create(
        model=deployment,
        input=[args.text],
    )

    vector = list(response.data[0].embedding)
    payload = {
        "status": "ok",
        "deployment": deployment,
        "endpoint": (
            os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        ),
        "api_version": (
            os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        ),
        "input_preview": args.text[:120],
        "embedding": _summarize_vector(vector),
        "usage": {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
