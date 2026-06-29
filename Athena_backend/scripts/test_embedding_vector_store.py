from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pinecone import Pinecone

from utilis.azure_embeddings import get_azure_embedding_model
from utilis.env import load_backend_env


def _required_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _index_dimension(pc: Pinecone, index_name: str) -> int | None:
    description = pc.describe_index(index_name)
    if hasattr(description, "to_dict"):
        description = description.to_dict()
    if isinstance(description, dict):
        dimension = description.get("dimension")
        return int(dimension) if dimension is not None else None
    return None


def _test_index(pc: Pinecone, index_name: str, vector: List[float]) -> Dict[str, Any]:
    namespace = "athena-embedding-smoke-test"
    vector_id = f"smoke-{uuid.uuid4()}"
    index = pc.Index(index_name)
    dimension = _index_dimension(pc, index_name)

    if dimension is not None and dimension != len(vector):
        raise RuntimeError(
            f"Pinecone index {index_name!r} dimension mismatch: "
            f"index={dimension}, embedding={len(vector)}"
        )

    index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": vector,
                "metadata": {
                    "source": "athena_embedding_smoke_test",
                    "text": "semantic matching for table nomination in athena",
                },
            }
        ],
        namespace=namespace,
    )

    result = index.query(
        vector=vector,
        top_k=1,
        include_metadata=True,
        namespace=namespace,
    )
    matches = getattr(result, "matches", None)
    if matches is None and isinstance(result, dict):
        matches = result.get("matches", [])
    matches = matches or []

    try:
        index.delete(ids=[vector_id], namespace=namespace)
    except Exception:
        pass

    if not matches:
        raise RuntimeError(f"Pinecone index {index_name!r} returned no matches after upsert")

    top_match = matches[0]
    return {
        "index": index_name,
        "dimension": dimension,
        "upserted_id": vector_id,
        "top_match_id": getattr(top_match, "id", None) or top_match.get("id"),
        "top_score": float(getattr(top_match, "score", 0.0) or top_match.get("score", 0.0)),
    }


def main() -> int:
    load_backend_env()

    model = get_azure_embedding_model()
    if model is None:
        raise RuntimeError("Azure embedding model is not configured or ATHENA_ENABLE_EMBEDDINGS is disabled")

    vector = model.embed_query("semantic matching for table nomination in athena")
    if not vector:
        raise RuntimeError("Azure embedding call returned an empty vector")

    pinecone_key = _required_env("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_key)

    index_names = [
        os.getenv("PINECONE_INDEX_NAME", "ai-store-index").strip() or "ai-store-index",
        os.getenv("PINECONE_SCHEMA_INDEX_NAME", "").strip(),
    ]
    index_names = list(dict.fromkeys(name for name in index_names if name))

    results = [_test_index(pc, index_name, vector) for index_name in index_names]
    print(
        json.dumps(
            {
                "status": "ok",
                "embedding": {
                    "deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                    "dimensions": len(vector),
                },
                "pinecone": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
