from pathlib import Path

import pdfplumber

from extraction.table_extraction import (
    extract_page_text_columns,
    page_column_boundaries,
)


def column_words(starts: list[int], words_per_column: int) -> list[dict[str, float]]:
    return [
        {
            "x0": start + index * 5,
            "x1": start + (index + 1) * 5,
            "top": 50 + (index % 10) * 20,
        }
        for start in starts
        for index in range(words_per_column)
    ]


def test_page_column_boundaries_detect_three_columns() -> None:
    boundaries = page_column_boundaries(
        column_words([40, 210, 380], words_per_column=25),
        page_width=600,
        page_height=800,
    )

    assert len(boundaries) == 4
    assert 180 <= boundaries[1] <= 210
    assert 350 <= boundaries[2] <= 380


def test_page_column_boundaries_keep_continuous_text_in_one_column() -> None:
    words = [
        {
            "x0": 40 + index * 20,
            "x1": 60 + index * 20,
            "top": 50 + (index % 10) * 20,
        }
        for index in range(25)
    ]

    assert page_column_boundaries(
        words,
        page_width=600,
        page_height=800,
    ) == [0.0, 600]


def test_fireball_description_isolated_from_neighboring_spell_damage() -> None:
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "pdfs"
        / "Player's Handbook I. Core Rulebook I v3.5.pdf"
    )
    with pdfplumber.open(pdf_path) as pdf:
        columns = extract_page_text_columns(pdf.pages[231])

    normalized_columns = [" ".join(column.split()) for column in columns]
    fireball_column = next(
        column
        for column in normalized_columns
        if "Fireball Evocation [Fire]" in column
    )

    assert "maximum 10d6" in fireball_column
    assert "maximum 20d6" not in fireball_column
