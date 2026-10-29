import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from qdrant_client import QdrantClient, models
from extraction.table_extraction import (
    CLASS_LEVEL_VALUE,
    extract_table_blocks,
    extract_table_data_rows,
    extract_page_text_columns,
    infer_class_spellcasting_ability,
    normalize_bonus_spell_rows,
    normalize_class_progression_rows,
    normalize_spell_table_rows,
    table_class_name,
)

PDF_DIR = Path(os.getenv("PDF_DIR", str(Path.home() / "dnd-rag" / "pdfs")))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("QDRANT_COLLECTION", "dnd35")
PAGE_CHUNK_SIZE = 1400
PAGE_CHUNK_OVERLAP = 180
EMBED_BATCH_SIZE = 32


def split_page_text(text: str) -> list[str]:
    """Split extracted page text into overlapping chunks.

    Overlap preserves context at chunk boundaries during retrieval.
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + PAGE_CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary <= start + PAGE_CHUNK_SIZE // 2:
                boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - PAGE_CHUNK_OVERLAP, start + 1)
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Request embeddings from the configured local Ollama service."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
        )
    return embeddings


def point_id(source: str, page: int, kind: str, ordinal: int, text: str) -> str:
    """Build a stable Qdrant point ID from a record's identifying fields."""
    identity = f"{source}|{page}|{kind}|{ordinal}|{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def collect_pdf(
    pdf_path: Path, preview_page: int | None = None
) -> list[dict[str, Any]]:
    """Collect page chunks, generic table rows, and supported semantic rows."""
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [page.extract_text(layout=True) or "" for page in pdf.pages]
        for page_number, page_text in enumerate(page_texts, start=1):
            if preview_page is not None and page_number != preview_page:
                continue

            chunk_number = 0
            for column_text in extract_page_text_columns(pdf.pages[page_number - 1]):
                for chunk in split_page_text(column_text):
                    records.append(
                        {
                            "source": pdf_path.name,
                            "page": page_number,
                            "kind": "page_text",
                            "ordinal": chunk_number,
                            "text": chunk,
                        }
                    )
                    chunk_number += 1

            for table_number, table_text in enumerate(
                extract_table_blocks(page_text), start=1
            ):
                records.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "kind": "table",
                        "ordinal": table_number,
                        "text": table_text,
                    }
                )
                for row_number, (header, row) in enumerate(
                    extract_table_data_rows(table_text)
                ):
                    records.append(
                        {
                            "source": pdf_path.name,
                            "page": page_number,
                            "kind": "table_data_row",
                            "ordinal": table_number * 10000 + row_number,
                            "table_title": table_text.splitlines()[0].strip(),
                            "text": f"Table columns: {header}. Table row: {row}.",
                        }
                    )
                class_name = table_class_name(table_text)
                class_ability = (
                    infer_class_spellcasting_ability(class_name, page_texts)
                    if class_name
                    else None
                )
                if class_name:
                    for row_number, (
                        class_level,
                        bab,
                        fortitude,
                        reflex,
                        will,
                        original_row,
                    ) in enumerate(normalize_class_progression_rows(table_text)):
                        records.append(
                            {
                                "source": pdf_path.name,
                                "page": page_number,
                                "kind": "class_progression_row",
                                "ordinal": table_number * 10000 + row_number,
                                "class_name": class_name,
                                "class_level": class_level,
                                "base_attack_bonus": bab,
                                "fortitude_save": fortitude,
                                "reflex_save": reflex,
                                "will_save": will,
                                "text": (
                                    f"Table {table_text.splitlines()[0].strip()}. "
                                    f"{class_name} class level {class_level}; "
                                    f"base attack bonus: +{bab}; base saves: "
                                    f"Fortitude +{fortitude}, Reflex +{reflex}, "
                                    f"Will +{will}. "
                                    f"Original table row: {original_row}"
                                ),
                            }
                        )
                for row_number, normalized_row in enumerate(
                    normalize_spell_table_rows(table_text)
                ):
                    level_match = CLASS_LEVEL_VALUE.search(normalized_row)
                    if level_match is None:
                        continue
                    records.append(
                        {
                            "source": pdf_path.name,
                            "page": page_number,
                            "kind": "table_row",
                            "ordinal": table_number * 10000 + row_number,
                            "class_name": class_name,
                            "class_level": int(level_match.group("level")),
                            "class_ability": class_ability,
                            "text": (
                                f"{table_text.splitlines()[0].strip()}. "
                                f"Class spells per day. {normalized_row}"
                            ),
                        }
                    )
            for row_number, normalized_row in enumerate(
                normalize_bonus_spell_rows(page_text)
            ):
                records.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "kind": "bonus_spell_row",
                        "ordinal": 900000 + row_number,
                        "text": (
                            "Table 1-1: Ability Modifiers and Bonus Spells. "
                            f"{normalized_row}"
                        ),
                    }
                )
    return records


def ensure_collection(client: QdrantClient, first_vector: list[float]) -> None:
    """Create the Qdrant collection or validate its existing vector schema."""
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        existing = info.config.params.vectors
        if isinstance(existing, dict):
            raise RuntimeError(
                "This importer expects a single-vector Qdrant collection, "
                "but the existing collection uses named vectors."
            )
        if existing.size != len(first_vector):
            raise RuntimeError(
                f"Collection {COLLECTION!r} has vector size {existing.size}, "
                f"but {EMBED_MODEL!r} returns {len(first_vector)} dimensions. "
                "Choose another collection name or remove/recreate the collection."
            )
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=len(first_vector),
            distance=models.Distance.COSINE,
        ),
    )


def ingest(records: list[dict[str, Any]], client: QdrantClient) -> int:
    """Embed records in batches and upsert them into the configured collection."""
    if not records:
        return 0

    indexed = 0
    for offset in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[offset : offset + EMBED_BATCH_SIZE]
        vectors = embed([record["text"] for record in batch])
        if indexed == 0:
            ensure_collection(client, vectors[0])

        points = [
            models.PointStruct(
                id=point_id(
                    record["source"],
                    record["page"],
                    record["kind"],
                    record["ordinal"],
                    record["text"],
                ),
                vector=vector,
                payload={
                    "source": record["source"],
                    "page": record["page"],
                    "kind": record["kind"],
                    "text": record["text"],
                    **{
                        key: record[key]
                        for key in (
                            "class_name",
                            "class_level",
                            "class_ability",
                            "table_title",
                        )
                        if key in record and record[key] is not None
                    },
                    **(
                        {"base_attack_bonus": record["base_attack_bonus"]}
                        if "base_attack_bonus" in record
                        else {}
                    ),
                    **{
                        key: record[key]
                        for key in ("fortitude_save", "reflex_save", "will_save")
                        if key in record
                    },
                },
            )
            for record, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        indexed += len(points)
    return indexed


def remove_existing_source(client: QdrantClient, source: str) -> None:
    """Remove previously indexed points for one source PDF."""
    if not client.collection_exists(COLLECTION):
        return
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source),
                    )
                ]
            )
        ),
        wait=True,
    )


def main() -> None:
    """Parse CLI options and preview or index PDFs from the configured directory."""
    parser = argparse.ArgumentParser(
        description="Index layout-preserving PDF text and labeled table rows."
    )
    parser.add_argument(
        "--preview-page",
        type=int,
        help="Print extracted chunks for this 1-based PDF page without indexing.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Process one PDF instead of all PDFs in PDF_DIR.",
    )
    args = parser.parse_args()

    if args.preview_page is not None and args.preview_page < 1:
        parser.error("--preview-page must be 1 or greater")

    pdfs = [args.pdf] if args.pdf else sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR}")

    client = None if args.preview_page else QdrantClient(url=QDRANT_URL)
    total = 0
    for pdf_path in pdfs:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        records = collect_pdf(pdf_path, preview_page=args.preview_page)
        if args.preview_page:
            print(f"\n=== {pdf_path.name}, PDF page {args.preview_page} ===")
            for record in records:
                print(f"\n[{record['kind']}]\n{record['text']}")
            continue

        assert client is not None
        if not records:
            raise RuntimeError(
                f"No extractable text or tables found in {pdf_path}; "
                "the existing indexed source was left untouched."
            )
        remove_existing_source(client, pdf_path.name)
        count = ingest(records, client)
        print(f"Indexed {count} text/table chunks from {pdf_path.name}")
        total += count

    if not args.preview_page:
        print(f"Done. Indexed {total} chunks in Qdrant collection {COLLECTION!r}.")


if __name__ == "__main__":
    main()
