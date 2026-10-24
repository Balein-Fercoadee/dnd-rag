import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from qdrant_client import QdrantClient, models

PDF_DIR = Path(os.getenv("PDF_DIR", str(Path.home() / "dnd-rag" / "pdfs")))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("QDRANT_COLLECTION", "dnd35")
PAGE_CHUNK_SIZE = 1400
PAGE_CHUNK_OVERLAP = 180
EMBED_BATCH_SIZE = 32

TABLE_CAPTION = re.compile(r"\bTable\s+\d+\s*[–-]\s*\d+\s*:", re.IGNORECASE)
TABLE_CAPTION_NAME = re.compile(
    r"\bTable\s+\d+\s*[–-]\s*\d+\s*:\s*(?:The\s+)?(?P<name>[^.\n]+)",
    re.IGNORECASE,
)
CLASS_TABLE_SUFFIX = re.compile(
    r"\s+(?:spells?\s+per\s+day|spells?\s+known|class\s+features)\s*$",
    re.IGNORECASE,
)
TABLE_DATA_ROW = re.compile(
    r"^\s*(?:\d+(?:st|nd|rd|th)?|\+\d+(?:/\+\d+)?)\s+\S",
    re.IGNORECASE,
)
CLASS_LEVEL_VALUE = re.compile(
    r"\bClass level (?P<level>\d{1,2})(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
CASTING_ABILITY = re.compile(r"\b(Intelligence|Wisdom|Charisma)\b", re.IGNORECASE)
SPELL_LEVEL_TOKEN = re.compile(
    r"^(?:0(?:th)?|[1-9](?:st|nd|rd|th)?)$", re.IGNORECASE
)
SPELL_SLOT_VALUE = re.compile(r"^(?:\d+|—|–|-)$")
BONUS_SCORE_ROW = re.compile(
    r"^\s*(?P<low>\d{1,2})\s*[–—-]\s*(?P<high>\d{1,2})"
    r"\s+(?P<modifier>[+＋\-−–]?\d+)\s+(?P<slots>.+?)\s*$"
)
BONUS_SPELL_LEVELS = (
    "0th-level",
    "1st-level",
    "2nd-level",
    "3rd-level",
    "4th-level",
    "5th-level",
    "6th-level",
    "7th-level",
    "8th-level",
    "9th-level",
)


def split_page_text(text: str) -> list[str]:
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


def extract_table_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    table_blocks = []
    for start, line in enumerate(lines):
        if not TABLE_CAPTION.search(line):
            continue

        block = [line.rstrip()]
        saw_data_row = False
        remaining_lines = lines[start + 1 :]
        index = 0
        while index < len(remaining_lines):
            candidate = remaining_lines[index]
            if TABLE_DATA_ROW.match(candidate):
                saw_data_row = True
                block.append(candidate.rstrip())
                index += 1
            elif saw_data_row:
                next_row = index + 1
                while (
                    next_row < len(remaining_lines)
                    and next_row <= index + 3
                    and not remaining_lines[next_row].strip()
                ):
                    next_row += 1
                if next_row < len(remaining_lines) and TABLE_DATA_ROW.match(
                    remaining_lines[next_row]
                ):
                    block.extend(
                        row.rstrip() for row in remaining_lines[index:next_row]
                    )
                    index = next_row
                else:
                    break
            else:
                block.append(candidate.rstrip())
                index += 1

        if saw_data_row:
            table_text = "\n".join(block).strip()
            normalized_rows = normalize_spell_table_rows(table_text)
            if normalized_rows:
                table_text += "\n\nNormalized spell-per-day rows:\n"
                table_text += "\n".join(normalized_rows)
            table_blocks.append(table_text)
    return table_blocks


def ordinal_number(value: int) -> str:
    suffix = (
        "th"
        if 10 <= value % 100 <= 13
        else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    )
    return f"{value}{suffix}"


def normalize_spell_table_rows(table_text: str) -> list[str]:
    if "spell" not in table_text.lower():
        return []

    lines = table_text.splitlines()
    spell_levels = []
    for end in range(len(lines)):
        start = max(0, end - 2)
        header_lines = [
            line
            for line in lines[start : end + 1]
            if not TABLE_CAPTION.search(line)
        ]
        header_block = " ".join(header_lines)
        tokens = [
            token
            for token in header_block.split()
            if SPELL_LEVEL_TOKEN.fullmatch(token)
        ]
        if "level" in header_block.lower() and len(tokens) >= 2:
            spell_levels = tokens
            break
    if not spell_levels:
        return []

    normalized = []
    for line in lines:
        if not TABLE_DATA_ROW.match(line):
            continue
        tokens = line.split()
        class_level = tokens[0]
        values = tokens[-len(spell_levels) :]
        if len(values) != len(spell_levels) or not all(
            SPELL_SLOT_VALUE.fullmatch(value) for value in values
        ):
            continue

        entries = []
        for spell_level, value in zip(spell_levels, values, strict=True):
            count = "unavailable" if value in {"—", "–", "-"} else value
            label = (
                "0th"
                if spell_level.lower() in {"0", "0th"}
                else spell_level
                if spell_level[-1].isalpha()
                else ordinal_number(int(spell_level))
            )
            entries.append(f"{label}-level: {count}")
        normalized.append(
            f"Class level {class_level}; spells per day: " + "; ".join(entries) + "."
        )
    return normalized


def normalize_bonus_spell_rows(page_text: str) -> list[str]:
    if "ability modifiers and bonus spells" not in page_text.lower():
        return []

    normalized = []
    for line in page_text.splitlines():
        match = BONUS_SCORE_ROW.match(line)
        if not match:
            continue

        low = int(match.group("low"))
        high = int(match.group("high"))
        modifier = match.group("modifier")
        tokens = match.group("slots").split()
        if low < 10 or len(tokens) < len(BONUS_SPELL_LEVELS):
            continue

        slot_tokens = tokens[: len(BONUS_SPELL_LEVELS)]
        if not all(
            token.isdigit() or token.strip("—–-") == ""
            for token in slot_tokens
        ):
            continue

        bonus_spells = []
        for level, value in zip(BONUS_SPELL_LEVELS, slot_tokens, strict=True):
            count = "0" if not value.isdigit() else value
            bonus_spells.append(f"{level}: {count}")
        normalized.append(
            f"Ability score {low}-{high} (modifier {modifier}); bonus spells per day: "
            + "; ".join(bonus_spells)
            + "."
        )
    return normalized


def table_class_name(table_text: str) -> str | None:
    match = TABLE_CAPTION_NAME.search(table_text.splitlines()[0])
    if match is None:
        return None
    name = CLASS_TABLE_SUFFIX.sub("", match.group("name").strip())
    return name or None


def infer_class_spellcasting_ability(
    class_name: str, page_texts: list[str]
) -> str | None:
    class_pattern = re.compile(rf"\b{re.escape(class_name)}s?\b", re.IGNORECASE)
    candidates = set()
    for page_text in page_texts:
        compact = re.sub(r"\s+", " ", page_text)
        for sentence in re.split(r"(?<=[.!?])\s+", compact):
            if not class_pattern.search(sentence) or "spell" not in sentence.casefold():
                continue
            if not re.search(r"\bdetermines?\b", sentence, re.IGNORECASE):
                continue
            ability = CASTING_ABILITY.search(sentence)
            if ability:
                candidates.add(ability.group(1).lower())
    return next(iter(candidates)) if len(candidates) == 1 else None


def embed(texts: list[str]) -> list[list[float]]:
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
    identity = f"{source}|{page}|{kind}|{ordinal}|{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def collect_pdf(pdf_path: Path, preview_page: int | None = None) -> list[dict[str, Any]]:
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [page.extract_text(layout=True) or "" for page in pdf.pages]
        for page_number, page_text in enumerate(page_texts, start=1):
            if preview_page is not None and page_number != preview_page:
                continue

            for chunk_number, chunk in enumerate(split_page_text(page_text)):
                records.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "kind": "page_text",
                        "ordinal": chunk_number,
                        "text": chunk,
                    }
                )

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
                class_name = table_class_name(table_text)
                class_ability = (
                    infer_class_spellcasting_ability(class_name, page_texts)
                    if class_name
                    else None
                )
                for row_number, normalized_row in enumerate(normalize_spell_table_rows(table_text)):
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
                        for key in ("class_name", "class_level", "class_ability")
                        if key in record and record[key] is not None
                    },
                },
            )
            for record, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        indexed += len(points)
    return indexed


def remove_existing_source(client: QdrantClient, source: str) -> None:
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
