"""Ollama embeddings and the shared Qdrant client."""

import requests
from fastapi import HTTPException
from qdrant_client import QdrantClient

from .rag_config import EMBED_MODEL, OLLAMA_URL, QDRANT_URL

qdrant = QdrantClient(url=QDRANT_URL)


def embed(text: str) -> list[float]:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Embedding request failed: {exc}"
        ) from exc
