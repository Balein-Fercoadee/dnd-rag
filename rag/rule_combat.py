"""rule combat helpers."""

import re

from typing import Any

from .rag_config import (
    ABILITY_SCORE_PATTERN,
    ABILITY_SCORE_RANGE_PATTERN,
    ARMOR_CLASS_QUESTION,
    ATTACK_BONUS_QUESTION,
    BASE_ATTACK_BONUS_QUESTION,
    BASE_ATTACK_BONUS_VALUE_PATTERN,
    BASE_SAVE_QUESTION,
    BASE_SAVE_VALUE_PATTERN,
)

from .rule_parsing import (
    bonus_score_range,
    citation,
    class_level_descriptors,
    requested_class_levels,
)


def structured_base_armor_class_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if ARMOR_CLASS_QUESTION.search(question) is None:
        return None
    if re.search(
        r"\b(?:armor bonus|wear(?:ing)? armor|shield(?: bonus)?|"
        r"dexterity|dex|size|deflection|natural armor|dodge|cover|concealment)\b",
        question,
        re.IGNORECASE,
    ):
        return None

    formula_pattern = re.compile(
        r"Armor Class.*?your AC:.*?\b10\s*\+",
        re.IGNORECASE | re.DOTALL,
    )
    formula_passage = next(
        (
            passage
            for passage in passages
            if isinstance(passage.get("text"), str)
            and formula_pattern.search(re.sub(r"\s+", " ", passage["text"]))
        ),
        None,
    )
    if formula_passage is None:
        return None

    return (
        f"The base AC is 10 before armor, shield, size, Dexterity, or other "
        f"modifiers; character level alone does not change it. "
        f"{citation(formula_passage)}"
    )


def structured_attack_bonus_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if (
        ATTACK_BONUS_QUESTION.search(question) is None
        or BASE_ATTACK_BONUS_QUESTION.search(question) is not None
    ):
        return None

    attack_type = (
        "ranged" if re.search(r"\branged\b", question, re.IGNORECASE) else "melee"
    )
    ability = "dexterity" if attack_type == "ranged" else "strength"
    ability_match = next(
        (
            match
            for match in ABILITY_SCORE_PATTERN.finditer(question)
            if match.group("ability").casefold() == ability
            and (match.group("score_before") or match.group("score_after"))
        ),
        None,
    )
    if ability_match is None:
        return None
    ability_score = int(
        ability_match.group("score_before") or ability_match.group("score_after")
    )
    expected_range = bonus_score_range(ability_score)
    modifier_passage = next(
        (
            passage
            for passage in passages
            if passage.get("kind") == "bonus_spell_row"
            and isinstance(passage.get("text"), str)
            and (range_match := ABILITY_SCORE_RANGE_PATTERN.search(passage["text"]))
            is not None
            and (
                int(range_match.group("low")),
                int(range_match.group("high")),
            )
            == expected_range
        ),
        None,
    )
    if modifier_passage is None:
        return None
    modifier_match = re.search(
        r"\bmodifier\s+(?P<modifier>[+−\-]?\d+)\b",
        modifier_passage["text"],
        re.IGNORECASE,
    )
    if modifier_match is None:
        return None
    ability_modifier = int(modifier_match.group("modifier").replace("−", "-"))

    class_names = {
        passage["class_name"]
        for passage in passages
        if passage.get("kind") == "class_progression_row"
        and isinstance(passage.get("class_name"), str)
    }
    requested = requested_class_levels(question, class_names)
    if not requested:
        return None

    contributions = []
    for class_name, class_level in requested:
        row = next(
            (
                passage
                for passage in passages
                if passage.get("kind") == "class_progression_row"
                and isinstance(passage.get("class_name"), str)
                and passage["class_name"].casefold() == class_name.casefold()
                and passage.get("class_level") == class_level
            ),
            None,
        )
        if row is None:
            return None
        bab = row.get("base_attack_bonus")
        if not isinstance(bab, int):
            text = row.get("text")
            bab_match = (
                BASE_ATTACK_BONUS_VALUE_PATTERN.search(text)
                if isinstance(text, str)
                else None
            )
            if bab_match is None:
                return None
            bab = int(bab_match.group("value"))
        contributions.append((class_name, class_level, bab, row))

    formula_pattern = re.compile(
        rf"attack bonus with a {attack_type} weapon is:.*?"
        rf"Base attack bonus\s*\+\s*{ability.capitalize()} modifier\s*\+\s*"
        r"size modifier",
        re.IGNORECASE | re.DOTALL,
    )
    formula_passage = next(
        (
            passage
            for passage in passages
            if isinstance(passage.get("text"), str)
            and formula_pattern.search(re.sub(r"\s+", " ", passage["text"]))
        ),
        None,
    )
    if formula_passage is None:
        return None

    size_categories = (
        "Fine",
        "Diminutive",
        "Tiny",
        "Small",
        "Medium",
        "Large",
        "Huge",
        "Gargantuan",
        "Colossal",
    )
    descriptors = class_level_descriptors(question, class_names)
    character_size = None
    size_rule_passage = None
    for descriptor in descriptors:
        descriptor_pattern = re.compile(
            rf"\b{re.escape(descriptor)}s?\b",
            re.IGNORECASE,
        )
        for passage in passages:
            text = passage.get("text")
            if not isinstance(text, str):
                continue
            normalized_text = re.sub(r"\s+", " ", text)
            descriptor_match = descriptor_pattern.search(normalized_text)
            if descriptor_match is None:
                continue
            nearby_text = normalized_text[
                max(0, descriptor_match.start() - 100) : descriptor_match.end() + 100
            ]
            if "creature" not in nearby_text.casefold():
                continue
            character_size = next(
                (
                    size
                    for size in size_categories
                    if re.search(rf"\b{size}\b", nearby_text, re.IGNORECASE)
                ),
                None,
            )
            if character_size is not None:
                size_rule_passage = passage
                break
        if character_size is not None:
            break
    assumed_medium = character_size is None
    character_size = character_size or "Medium"
    size_pattern = re.compile(
        rf"\b{character_size}\s*(?:\|\s*)?(?P<modifier>[+−\-]?\d+)\b",
        re.IGNORECASE,
    )
    size_modifier_passage = next(
        (
            passage
            for passage in passages
            if isinstance(passage.get("text"), str)
            and (size_match := size_pattern.search(passage["text"])) is not None
            and passage.get("kind") in {"table", "table_data_row", "page_text"}
            and "size modifier" in passage["text"].casefold()
        ),
        None,
    )
    if size_modifier_passage is None:
        return None
    size_modifier_match = size_pattern.search(size_modifier_passage["text"])
    if size_modifier_match is None:
        return None
    size_modifier = int(size_modifier_match.group("modifier").replace("−", "-"))

    total_bab = sum(bab for _, _, bab, _ in contributions)
    total = total_bab + ability_modifier + size_modifier
    bab_description = " + ".join(
        f"+{bab} ({class_name} {class_level})"
        for class_name, class_level, bab, _ in contributions
    )
    source_passages = [
        *(row for _, _, _, row in contributions),
        modifier_passage,
        formula_passage,
        size_modifier_passage,
    ]
    if size_rule_passage is not None:
        source_passages.append(size_rule_passage)
    source_pages: dict[str, list[str]] = {}
    for passage in source_passages:
        source = str(passage["source"])
        page = str(passage["page"])
        pages = source_pages.setdefault(source, [])
        if page not in pages:
            pages.append(page)
    citations = " ".join(
        (
            f"[{source}, PDF p. {pages[0]}]"
            if len(pages) == 1
            else f"[{source}, PDF pp. {', '.join(pages)}]"
        )
        for source, pages in source_pages.items()
    )
    size_details = f"{character_size} size {size_modifier:+d}"
    assumption = "Assuming Medium size, " if assumed_medium else ""
    return (
        f"{assumption}the {attack_type} attack bonus is +{total}: BAB "
        f"{bab_description}, {ability.capitalize()} {ability_score} "
        f"({ability_modifier:+d}), {size_details}. {citations}"
    )


def structured_base_attack_bonus_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if BASE_ATTACK_BONUS_QUESTION.search(question) is None:
        return None

    class_names = {
        passage["class_name"]
        for passage in passages
        if passage.get("kind") == "class_progression_row"
        and isinstance(passage.get("class_name"), str)
    }
    requested = requested_class_levels(question, class_names)
    if not requested:
        return None

    contributions = []
    for class_name, class_level in requested:
        row = next(
            (
                passage
                for passage in passages
                if passage.get("kind") == "class_progression_row"
                and isinstance(passage.get("class_name"), str)
                and passage["class_name"].casefold() == class_name.casefold()
                and passage.get("class_level") == class_level
            ),
            None,
        )
        if row is None:
            return None
        bab = row.get("base_attack_bonus")
        if not isinstance(bab, int):
            text = row.get("text")
            match = (
                BASE_ATTACK_BONUS_VALUE_PATTERN.search(text)
                if isinstance(text, str)
                else None
            )
            if match is None:
                return None
            bab = int(match.group("value"))
        contributions.append((class_name, class_level, bab, row))

    total = sum(bab for _, _, bab, _ in contributions)
    breakdown = "; ".join(
        f"{class_name} {class_level}: +{bab}"
        for class_name, class_level, bab, _ in contributions
    )
    citations = " ".join(dict.fromkeys(citation(row) for _, _, _, row in contributions))
    return f"The total base attack bonus is +{total} ({breakdown}) {citations}."


def structured_base_saves_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if BASE_SAVE_QUESTION.search(question) is None:
        return None

    class_names = {
        passage["class_name"]
        for passage in passages
        if passage.get("kind") == "class_progression_row"
        and isinstance(passage.get("class_name"), str)
    }
    requested = requested_class_levels(question, class_names)
    if not requested:
        return None

    save_names = ("fortitude", "reflex", "will")
    totals = dict.fromkeys(save_names, 0)
    citations = []
    breakdowns = {save_name: [] for save_name in save_names}
    for class_name, class_level in requested:
        row = next(
            (
                passage
                for passage in passages
                if passage.get("kind") == "class_progression_row"
                and isinstance(passage.get("class_name"), str)
                and passage["class_name"].casefold() == class_name.casefold()
                and passage.get("class_level") == class_level
            ),
            None,
        )
        if row is None:
            return None

        values = {
            "fortitude": row.get("fortitude_save"),
            "reflex": row.get("reflex_save"),
            "will": row.get("will_save"),
        }
        if not all(isinstance(value, int) for value in values.values()):
            text = row.get("text")
            match = (
                BASE_SAVE_VALUE_PATTERN.search(text) if isinstance(text, str) else None
            )
            if match is None:
                return None
            values = {
                save_name: int(match.group(save_name)) for save_name in save_names
            }

        for save_name, value in values.items():
            totals[save_name] += value
            breakdowns[save_name].append(f"+{value} ({class_name} {class_level})")
        citations.append(citation(row))

    citation_text = " ".join(dict.fromkeys(citations))
    results = "; ".join(
        f"{save_name.capitalize()} +{totals[save_name]} "
        f"({' + '.join(breakdowns[save_name])})"
        for save_name in save_names
    )
    return f"The total base saving throws are {results}. {citation_text}."
