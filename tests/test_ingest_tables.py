from ingest_v2 import (
    extract_table_blocks,
    extract_table_data_rows,
    infer_class_spellcasting_ability,
    is_table_data_row,
    normalize_class_progression_rows,
    table_class_name,
)


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


def test_sorcerer_ability_rule_does_not_get_assigned_to_wizard() -> None:
    sorcerer_rule = (
        "Abilities: Charisma determines how powerful a spell a sorcerer can cast. "
        "These new spells can be chosen from the sorcerer/wizard spell list."
    )

    assert infer_class_spellcasting_ability("Wizard", [sorcerer_rule]) is None
    assert infer_class_spellcasting_ability("Sorcerer", [sorcerer_rule]) == "charisma"


def test_wizard_ability_is_inferred_from_wizard_specific_rules() -> None:
    wizard_rules = (
        "Intelligence determines how well your character learns and reasons. "
        "This ability is important for wizards because it affects how many spells "
        "they can cast. A wizard gains bonus spells based on her Intelligence score. "
        "The minimum Intelligence score needed to cast a wizard spell is 10 + "
        "the spell's level."
    )

    assert infer_class_spellcasting_ability("Wizard", [wizard_rules]) == "intelligence"


def test_bard_ability_is_inferred_when_pdf_columns_interrupt_the_sentence() -> None:
    bard_rule = (
        "Charisma determines how teachers train them in powerful a spell "
        "a bard can cast, how many spells he can cast per day."
    )

    assert infer_class_spellcasting_ability("Bard", [bard_rule]) == "charisma"


def test_hyphenated_pdf_line_break_does_not_hide_class_ability() -> None:
    cleric_rule = (
        "Abilities: Wisdom determines how pow- erful a spell a cleric can cast, "
        "how many spells he can cast per day."
    )

    assert infer_class_spellcasting_ability("Cleric", [cleric_rule]) == "wisdom"


def test_ranger_casting_requirement_identifies_wisdom() -> None:
    ranger_rule = (
        "A Wisdom score of 11 or higher is required to cast any ranger spell."
    )

    assert infer_class_spellcasting_ability("Ranger", [ranger_rule]) == "wisdom"


def test_class_progression_rows_extract_each_classes_bab() -> None:
    barbarian_table = """\
Table 3-3: The Barbarian
Level Base Attack Bonus Fort Save Ref Save Will Save Special
1st +1 +2 +0 +0 Rage 1/day
2nd +2 +3 +0 +0 Uncanny dodge
3rd +3 +3 +1 +1 Trap sense +1
11th +11/+6/+1 +7 +3 +3 Greater rage
"""

    assert normalize_class_progression_rows(barbarian_table) == [
        (1, 1, 2, 0, 0, "1st +1 +2 +0 +0 Rage 1/day"),
        (2, 2, 3, 0, 0, "2nd +2 +3 +0 +0 Uncanny dodge"),
        (3, 3, 3, 1, 1, "3rd +3 +3 +1 +1 Trap sense +1"),
        (11, 11, 7, 3, 3, "11th +11/+6/+1 +7 +3 +3 Greater rage"),
    ]


def test_character_level_bab_summary_is_not_a_class_progression_table() -> None:
    summary_table = """\
Table 3-1: Base Save and Base Attack Bonuses
Character Level Base Attack Bonus (Good) Base Attack Bonus (Average)
1st +1 +0
2nd +2 +1
"""

    assert normalize_class_progression_rows(summary_table) == []


def test_numeric_range_table_rows_keep_generic_header_context() -> None:
    ability_table = """\
Table 1-1: Ability Modifiers and Bonus Spells
Score  Modifier  Bonus Spells by Spell Level
1      -5        —
2-3    -4        —
20-21  +5        1st
"""

    table_block = extract_table_blocks(ability_table)[0]
    rows = extract_table_data_rows(table_block)

    assert "Normalized spell-per-day rows:" not in table_block
    assert any("20-21" in row and "Modifier" in header for header, row in rows)


def test_text_labeled_table_rows_keep_generic_header_context() -> None:
    deity_table = """\
Table 3-7: Deities
Deity               Alignment       Domains             Typical Worshipers
Heironeous          Lawful good      Good, Law, War      Paladins, fighters
Moradin             Lawful good      Earth, Good         Dwarves
"""

    table_block = extract_table_blocks(deity_table)[0]
    rows = extract_table_data_rows(table_block)

    assert len(rows) == 2
    assert all("Typical Worshipers" in header for header, _ in rows)
    assert "Heironeous" in rows[0][1]
    assert "Moradin" in rows[1][1]
    assert "Typical Worshipers" in rows[0][0]


def test_prose_table_references_do_not_start_table_blocks() -> None:
    prose = """\
Table 1-1: Ability Modifiers and Bonus Spells, page 8). Weapons: Shortspear.
54 gains one or more new spells, as indicated on Table 3-17: Sorcerer Spells Known.
"""

    assert extract_table_blocks(prose) == []


def test_caption_with_adjacent_column_text_keeps_its_title() -> None:
    table = """\
Table 3-5: The Bard Spells Known        yet used up his allotment of spells.
Level 0 1st 2nd
1st   4  —   —
"""

    table_block = extract_table_blocks(table)[0]

    assert table_block.splitlines()[0] == "Table 3-5: The Bard Spells Known"
    assert table_class_name(table_block) == "Bard"


def test_page_number_followed_by_prose_is_not_a_table_row() -> None:
    page_footer = (
        "54 gains one or more new spells, as indicated on Table 3-17: "
        "Sorcerer Spells Known."
    )

    assert not is_table_data_row(page_footer)
