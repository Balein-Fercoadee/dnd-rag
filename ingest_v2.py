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

TABLE_CAPTION = re.compile(
    r"^\s*Table\s+\d+\s*[–-]\s*\d+\s*:",
    re.IGNORECASE,
)
TABLE_CAPTION_NAME = re.compile(
    r"^\s*Table\s+\d+\s*[–-]\s*\d+\s*:\s*(?:The\s+)?"
    r"(?P<name>.*?)(?=\s{2,}|$)",
    re.IGNORECASE,
)
CLASS_TABLE_SUFFIX = re.compile(
    r"\s+(?:spells?\s+per\s+day|spells?\s+known|class\s+features)\s*$",
    re.IGNORECASE,
)
TABLE_DATA_ROW = re.compile(
    r"^\s*(?:\d+(?:st|nd|rd|th)?(?:\s*[–—-]\s*\d+(?:st|nd|rd|th)?)?"
    r"|\+\d+(?:/\+\d+)*)\s+\S",
    re.IGNORECASE,
)
CLASS_LEVEL_VALUE = re.compile(
    r"\bClass level (?P<level>\d{1,2})(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
SPELL_LEVEL_TOKEN = re.compile(
    r"^(?:0(?:th)?|[1-9](?:st|nd|rd|th)?)$", re.IGNORECASE
)
SPELL_SLOT_VALUE = re.compile(r"^(?:\d+|—|–|-)$")
BASE_ATTACK_BONUS_VALUE = re.compile(r"^\+\d+(?:/\+\d+)*$")
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


def extract_table_blocks(text: str) -> list[str]:
    """Extract captioned table regions and add supported spell-table summaries."""
    lines = text.splitlines()
    table_blocks = []
    for start, line in enumerate(lines):
        if not TABLE_CAPTION.search(line):
            continue

        caption_match = TABLE_CAPTION_NAME.match(line)
        caption = (
            f"{line[:caption_match.start('name')]}{caption_match.group('name').strip()}"
            if caption_match
            else line.rstrip()
        )
        block = [caption]
        saw_data_row = False
        text_row_indent = None
        remaining_lines = lines[start + 1 :]
        index = 0
        while index < len(remaining_lines):
            candidate = remaining_lines[index]
            if TABLE_CAPTION.search(candidate):
                break
            if is_table_data_row(candidate):
                saw_data_row = True
                if (
                    text_row_indent is None
                    and not TABLE_DATA_ROW.match(candidate)
                    and len(table_cells(candidate)) >= 3
                ):
                    text_row_indent = len(candidate) - len(candidate.lstrip())
                block.append(candidate.rstrip())
                index += 1
            elif (
                saw_data_row
                and text_row_indent is not None
                and candidate.strip()
                and len(candidate) - len(candidate.lstrip()) >= text_row_indent
                and (
                    len(candidate.split()) >= 3
                    or (
                        len(candidate) - len(candidate.lstrip()) > text_row_indent
                        and len(candidate.split()) >= 1
                    )
                )
            ):
                block.append(candidate.rstrip())
                index += 1
            elif saw_data_row:
                next_row = next(
                    (
                        candidate_index
                        for candidate_index in range(index + 1, min(index + 4, len(remaining_lines)))
                        if TABLE_CAPTION.search(remaining_lines[candidate_index])
                        or is_table_data_row(remaining_lines[candidate_index])
                    ),
                    None,
                )
                if next_row is not None and TABLE_CAPTION.search(
                    remaining_lines[next_row]
                ):
                    break
                if next_row is not None:
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


def table_cells(line: str) -> list[str]:
    """Split a layout-preserved table line at repeated whitespace boundaries."""
    return [cell.strip() for cell in re.split(r"\s{2,}", line.strip()) if cell.strip()]


def is_table_data_row(line: str) -> bool:
    """Return whether a line resembles a numeric or aligned table row."""
    numeric_match = TABLE_DATA_ROW.match(line)
    if numeric_match:
        first_token = line.split(maxsplit=1)[0]
        if re.search(r"(?:st|nd|rd|th)$", first_token, re.IGNORECASE):
            return True
        if re.search(r"[–—-]", first_token) or first_token.startswith("+"):
            return True
        numeric_cells = sum(
            bool(
                re.fullmatch(
                    r"[+＋\-−]?\d+(?:/\+\d+)*|[—–-]",
                    cell,
                )
            )
            for cell in line.split()[1:]
        )
        if numeric_cells >= 2:
            return True
    return len(table_cells(line)) >= 3 and not TABLE_CAPTION.search(line)


def extract_table_data_rows(table_text: str) -> list[tuple[str, str]]:
    """Return raw table rows paired with their caption and available headers."""
    lines = table_text.splitlines()
    numeric_rows = [
        index
        for index, line in enumerate(lines)
        if TABLE_DATA_ROW.match(line) and is_table_data_row(line)
    ]
    if numeric_rows:
        row_indexes = numeric_rows
    else:
        aligned_rows = [
            index for index, line in enumerate(lines) if len(table_cells(line)) >= 3
        ]
        if not aligned_rows:
            return []
        header_index = aligned_rows[0]
        header_lines = [
            line.strip()
            for line in lines[: header_index + 1]
            if line.strip() and not TABLE_CAPTION.search(line)
        ]
        caption = next(
            (line.strip() for line in lines if TABLE_CAPTION.search(line)),
            "Table",
        )
        header = " | ".join(
            [caption, *(" | ".join(table_cells(item)) for item in header_lines)]
        )
        header_indent = len(lines[header_index]) - len(lines[header_index].lstrip())
        text_rows = []
        for line in lines[header_index + 1 :]:
            indent = len(line) - len(line.lstrip())
            if not line.strip() or indent < header_indent:
                continue
            if indent > header_indent and text_rows:
                text_rows[-1] += f" {line.strip()}"
            elif len(line.split()) >= 3:
                text_rows.append(line.strip())
        row_indexes = []
        if text_rows:
            return [(header, " | ".join(table_cells(row))) for row in text_rows]

    if not row_indexes:
        return []

    first_row_index = row_indexes[0]
    header_lines = [
        line.strip()
        for line in lines[:first_row_index]
        if line.strip() and not TABLE_CAPTION.search(line)
    ]
    caption = next(
        (line.strip() for line in lines if TABLE_CAPTION.search(line)),
        "Table",
    )
    header = " | ".join(
        [caption, *(" | ".join(table_cells(line)) for line in header_lines)]
    )
    rows = []
    for row_index in row_indexes:
        row = " | ".join(table_cells(lines[row_index]))
        if row:
            rows.append((header, row))
    return rows


def ordinal_number(value: int) -> str:
    """Format an integer as an English ordinal."""
    suffix = (
        "th"
        if 10 <= value % 100 <= 13
        else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    )
    return f"{value}{suffix}"


def normalize_spell_table_rows(table_text: str) -> list[str]:
    """Normalize class spells-per-day rows while retaining original table text."""
    if "spells per day" not in table_text.lower():
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

    first_data_index = next(
        index for index, line in enumerate(lines) if TABLE_DATA_ROW.match(line)
    )
    original_header = " ".join(
        line.strip() for line in lines[:first_data_index] if line.strip()
    )
    normalized = []
    for line_index, line in enumerate(lines):
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
        original_row_lines = [line.strip()]
        for continuation in lines[line_index + 1 :]:
            if TABLE_CAPTION.search(continuation) or TABLE_DATA_ROW.match(continuation):
                break
            if continuation.strip():
                original_row_lines.append(continuation.strip())
        original_row = " ".join(original_row_lines)
        normalized.append(
            f"Class level {class_level}; spells per day: "
            + "; ".join(entries)
            + f". Original table columns: {original_header}."
            + f" Original table row: {original_row}"
        )
    return normalized


def normalize_class_progression_rows(
    table_text: str,
) -> list[tuple[int, int, int, int, int, str]]:
    """Extract class level, base attack bonus, saves, and original row text."""
    lines = table_text.splitlines()
    first_data_index = next(
        (index for index, line in enumerate(lines) if TABLE_DATA_ROW.match(line)),
        None,
    )
    if first_data_index is None:
        return []

    header = " ".join(line.strip() for line in lines[:first_data_index])
    if (
        not re.search(r"\bbase\b", header, re.IGNORECASE)
        or not re.search(r"\battack bonus\b", header, re.IGNORECASE)
        or re.search(r"\bcharacter level\b", header, re.IGNORECASE)
    ):
        return []

    rows = []
    for line_index, line in enumerate(lines):
        if not TABLE_DATA_ROW.match(line):
            continue
        tokens = line.split()
        if (
            len(tokens) < 2
            or not re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)", tokens[0], re.IGNORECASE)
            or not BASE_ATTACK_BONUS_VALUE.fullmatch(tokens[1])
        ):
            continue
        level = int(re.match(r"\d+", tokens[0]).group())
        base_attack_bonus = int(tokens[1].split("/", maxsplit=1)[0].lstrip("+"))
        continuation_lines = []
        for continuation in lines[line_index + 1 :]:
            if TABLE_CAPTION.search(continuation) or TABLE_DATA_ROW.match(continuation):
                break
            if continuation.strip():
                continuation_lines.append(continuation.strip())
        original_row = " ".join([line.strip(), *continuation_lines])
        if len(tokens) < 5:
            continue
        save_values = tokens[2:5]
        if not all(re.fullmatch(r"\+\d+", value) for value in save_values):
            continue
        fortitude, reflex, will = (int(value[1:]) for value in save_values)
        rows.append((level, base_attack_bonus, fortitude, reflex, will, original_row))
    return rows


def normalize_bonus_spell_rows(page_text: str) -> list[str]:
    """Normalize ability-score bonus spell rows found in page text."""
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
    """Return the class named by a class-progression table caption."""
    match = TABLE_CAPTION_NAME.search(table_text.splitlines()[0])
    if match is None:
        return None
    name = CLASS_TABLE_SUFFIX.sub("", match.group("name").strip())
    return name or None


def infer_class_spellcasting_ability(
    class_name: str, page_texts: list[str]
) -> str | None:
    """Infer a class casting ability only when explicit rules agree uniquely."""
    candidates = set()
    escaped_class = re.escape(class_name)
    class_subject = rf"(?:a|an|the)\s+{escaped_class}s?"
    explicit_class_rules = (
        re.compile(
            rf"\b(?P<ability>Intelligence|Wisdom|Charisma)\s+determines\s+"
            rf"how powerful a spell {class_subject}\s+can cast\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b{class_subject}\s+gains bonus spells based on "
            r"(?:her|his|its|their)\s+"
            r"(?P<ability>Intelligence|Wisdom|Charisma)\s+score\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bminimum\s+(?P<ability>Intelligence|Wisdom|Charisma)\s+"
            rf"score needed to cast {class_subject}\s+spell\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<ability>Intelligence|Wisdom|Charisma)\s+score of "
            rf"\d+\s+or higher is required to cast "
            rf"(?:any|a|an|the)\s+{escaped_class}s?\s+spell\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bto learn, prepare, or cast a spell, {class_subject}\s+"
            r"must have an?\s+"
            r"(?P<ability>Intelligence|Wisdom|Charisma)\s+score\b",
            re.IGNORECASE,
        ),
    )
    for page_text in page_texts:
        compact = re.sub(r"\s+", " ", page_text)
        compact = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", compact)
        for rule_pattern in explicit_class_rules:
            candidates.update(
                match.group("ability").lower()
                for match in rule_pattern.finditer(compact)
            )
    return next(iter(candidates)) if len(candidates) == 1 else None


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


def collect_pdf(pdf_path: Path, preview_page: int | None = None) -> list[dict[str, Any]]:
    """Collect page chunks, generic table rows, and supported semantic rows."""
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
                    ) in enumerate(
                        normalize_class_progression_rows(table_text)
                    ):
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
