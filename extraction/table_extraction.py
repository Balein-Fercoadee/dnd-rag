import re
from typing import Any

from pdfplumber.page import Page

TABLE_CAPTION = re.compile(
    r"^\s*Table\s+\d+\s*[–-]\s*\d+\s*:",
    re.IGNORECASE,
)
TABLE_CAPTION_NAME = re.compile(
    r"^\s*Table\s+\d+\s*[–-]\s*\d+\s*:\s*(?:The\s+)?" r"(?P<name>.*?)(?=\s{2,}|$)",
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
SPELL_LEVEL_TOKEN = re.compile(r"^(?:0(?:th)?|[1-9](?:st|nd|rd|th)?)$", re.IGNORECASE)
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


def page_column_boundaries(
    words: list[dict[str, Any]], page_width: float, page_height: float
) -> list[float]:
    """Find persistent vertical gutters separating page-text columns."""
    occupancy = [False] * int(page_width + 1)
    for word in words:
        if not (
            35 < word["top"] < page_height - 35
            and word["x1"] - word["x0"] < page_width * 0.2
        ):
            continue
        left = max(0, int(word["x0"]))
        right = min(len(occupancy), int(word["x1"]) + 1)
        for x in range(left, right):
            occupancy[x] = True

    gaps = []
    gap_start = None
    for x, occupied in enumerate(occupancy):
        if not occupied and gap_start is None:
            gap_start = x
        elif occupied and gap_start is not None:
            if (
                x - gap_start >= max(8, int(page_width * 0.015))
                and gap_start > page_width * 0.12
                and x < page_width * 0.88
            ):
                gaps.append((gap_start, x))
            gap_start = None

    boundaries = [0.0, *((left + right) / 2 for left, right in gaps), page_width]
    minimum_words = max(20, int(len(words) * 0.05))
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        column_words = sum(
            left <= (word["x0"] + word["x1"]) / 2 < right for word in words
        )
        if column_words < minimum_words:
            return [0.0, page_width]
    return boundaries


def extract_page_text_columns(page: Page) -> list[str]:
    """Extract page text in column order instead of interleaving columns."""
    words = page.extract_words()
    boundaries = page_column_boundaries(words, page.width, page.height)
    if len(boundaries) == 2:
        text = page.extract_text(layout=True) or ""
        return [text] if text else []

    columns = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        text = page.crop((left, 0, right, page.height)).extract_text() or ""
        if text.strip():
            columns.append(text)
    return columns


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
                        for candidate_index in range(
                            index + 1, min(index + 4, len(remaining_lines))
                        )
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
            line for line in lines[start : end + 1] if not TABLE_CAPTION.search(line)
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
                else (
                    spell_level
                    if spell_level[-1].isalpha()
                    else ordinal_number(int(spell_level))
                )
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
            token.isdigit() or token.strip("—–-") == "" for token in slot_tokens
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
            rf"\b(?P<ability>Intelligence|Wisdom|Charisma)\s+determines"
            rf".{{0,180}}?\b(?:a|an|the)\s+{escaped_class}\s+can cast\b",
            re.IGNORECASE,
        ),
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
