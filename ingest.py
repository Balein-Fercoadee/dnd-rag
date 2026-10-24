from pathlib import Path
import hashlib
import uuid

import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

PDF_DIR = Path.home() / "dnd-rag" / "pdfs"
OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_URL = "http://127.0.0.1:6333"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "dnd35"
CHUNK_SIZE = 1200
OVERLAP = 200
BATCH_SIZE = 32


def split_text(text: str) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)

    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {PDF_DIR}")

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        print(f"Using existing Qdrant collection: {COLLECTION}")
    else:
        first_vector = embed(["dimension check"])[0]
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=len(first_vector),
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created collection {COLLECTION} ({len(first_vector)} dimensions)")

    indexed = 0
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        records = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk_number, chunk in enumerate(split_text(text)):
                records.append((page_number, chunk_number, chunk))

        for offset in range(0, len(records), BATCH_SIZE):
            batch = records[offset : offset + BATCH_SIZE]
            vectors = embed([item[2] for item in batch])
            points = []

            for (page_number, chunk_number, chunk), vector in zip(batch, vectors):
                key = f"{pdf.name}|{page_number}|{chunk_number}|{chunk}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "source": pdf.name,
                            "page": page_number,
                            "text": chunk,
                        },
                    )
                )

            client.upsert(collection_name=COLLECTION, points=points)
            indexed += len(points)

        print(f"Indexed {len(records)} chunks from {pdf.name}")

    print(f"Done. Indexed {indexed} chunks total.")


if __name__ == "__main__":
    main()
