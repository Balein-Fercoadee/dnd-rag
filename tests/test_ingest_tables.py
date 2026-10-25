from ingest_v2 import extract_table_blocks


WIZARD_TABLE = """\
Table 3-18: The Wizard
Base Attack Bonus Fort Save Ref Save Will Save Special Spells per Day
Level                         0 1st 2nd 3rd 4th 5th 6th 7th 8th 9th
1st   +0   +0   +0   +2   Summon familiar, 3 1 — — — — — — — —
                         Scribe Scroll
2nd   +1   +0   +0   +3   4 2 — — — — — — — —
5th   +2   +1   +1   +4   Bonus feat 4 3 2 1 — — — — — —
"""


def test_spell_table_block_and_rows_preserve_non_spell_columns() -> None:
    table_block = extract_table_blocks(WIZARD_TABLE)[0]
    normalized_row = next(
        line
        for line in table_block.splitlines()
        if line.startswith("Class level 5th;")
    )
    first_level_row = next(
        line
        for line in table_block.splitlines()
        if line.startswith("Class level 1st;")
    )

    assert "Base Attack Bonus" in table_block
    assert "Fort Save Ref Save Will Save" in table_block
    assert "Special Spells per Day" in table_block
    assert "Base Attack Bonus" in normalized_row
    assert "Fort Save Ref Save Will Save" in normalized_row
    assert "Special Spells per Day" in normalized_row
    assert "Bonus feat" in normalized_row
    assert "+2   +1   +1   +4" in normalized_row
    assert "5th-level: unavailable" in normalized_row
    assert "Summon familiar" in first_level_row
    assert "Scribe Scroll" in first_level_row
    assert "0th-level: 3" in first_level_row
    assert "1st-level: 1" in first_level_row
