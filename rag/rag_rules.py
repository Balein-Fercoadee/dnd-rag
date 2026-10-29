"""Rule parsing and deterministic answer formatting."""
import re
from typing import Any

from fastapi import HTTPException

from .rag_config import (
    ABILITY_SCORE_PATTERN,
    ABILITY_SCORE_RANGE_PATTERN,
    ARMOR_CLASS_QUESTION,
    ATTACK_BONUS_QUESTION,
    BASE_ATTACK_BONUS_QUESTION,
    BASE_ATTACK_BONUS_VALUE_PATTERN,
    BASE_SAVE_QUESTION,
    BASE_SAVE_VALUE_PATTERN,
    CASTER_LEVEL_IN_QUESTION,
    CHARACTER_CLASS_LEVEL_PATTERN,
    CLASS_LEVEL_ROW_PATTERN,
    CLASS_TABLE_NAME_PATTERN,
    FEAT_QUALIFICATION_QUESTION,
    FEAT_NAME_IN_QUESTION,
    FEAT_SELECTION_LEVEL_PATTERN,
    SPELL_LEVEL_PATTERN,
    SPELL_SLOT_COUNT_PATTERN,
)

def bonus_score_range(ability_score: int) -> tuple[int, int]:
    if ability_score < 1 or ability_score > 99:
        raise HTTPException(status_code=400, detail="Ability score is outside supported range")
    low = ability_score if ability_score == 1 or ability_score % 2 == 0 else ability_score - 1
    high = low if ability_score == 1 else low + 1
    return low, high


def class_name_from_text(text: str) -> str | None:
    match = CLASS_TABLE_NAME_PATTERN.search(text)
    return match.group("name").strip() if match else None


def feat_name_from_question(question: str) -> str | None:
    match = FEAT_NAME_IN_QUESTION.search(question)
    if match is None:
        return None
    name = re.sub(r"\s+feat$", "", match.group("name").strip(), flags=re.IGNORECASE)
    return name or None


def feat_class_level_prerequisite_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if FEAT_QUALIFICATION_QUESTION.search(question) is None:
        return None
    feat_name = feat_name_from_question(question)
    if feat_name is None:
        return None

    character_levels = {
        match.group("class").strip().casefold(): int(match.group("level"))
        for match in CHARACTER_CLASS_LEVEL_PATTERN.finditer(question)
    }
    if not character_levels:
        return None

    character_level = sum(character_levels.values())
    selection_match = FEAT_SELECTION_LEVEL_PATTERN.search(question)
    selection_level = (
        int(selection_match.group("level"))
        if selection_match is not None
        else character_level
    )
    if selection_level < character_level:
        return None
    levels_available = selection_level - character_level

    feat_heading = re.compile(
        rf"\b{re.escape(feat_name)}\s*\[[A-Z][A-Z ]*\]",
        re.IGNORECASE,
    )
    next_feat_heading = re.compile(r"\b[A-Z][A-Z'’ -]{2,}\s+\[[A-Z ]+\]")
    prerequisite_pattern = re.compile(
        r"\bPrerequisites?\s*:\s*(?P<requirements>.*?)"
        r"(?=\.|\b(?:Benefit|Normal|Special)\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    class_requirement_pattern = re.compile(
        r"(?:^|[,;]\s*|\band\s+)"
        r"(?P<class>[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2})\s+level\s+"
        r"(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
        re.IGNORECASE,
    )
    proficiency_requirement_pattern = re.compile(
        r"\bproficiency with (?:the )?selected weapon\b",
        re.IGNORECASE,
    )
    selected_feat_pattern = re.compile(
        r"\balready selected (?:the )?(?P<name>[\w][\w'’-]*(?:\s+[\w][\w'’-]*)*?)\s+feat\b",
        re.IGNORECASE,
    )
    selected_weapon_feat_pattern = re.compile(
        r"\b(?P<name>[A-Z][\w'’-]*\s+[A-Z][\w'’-]*"
        r"(?:\s+[A-Z][\w'’-]*){0,1})"
        r"\s+with selected weapon\b"
    )

    for passage in passages:
        text = passage.get("text")
        if not isinstance(text, str):
            continue
        heading_match = feat_heading.search(text)
        if heading_match is None:
            continue
        entry = text[heading_match.end():]
        next_heading = next_feat_heading.search(entry)
        if next_heading is not None:
            entry = entry[:next_heading.start()]
        prerequisite_match = prerequisite_pattern.search(entry)
        if prerequisite_match is None:
            continue

        requirements = re.sub(
            r"\s+",
            " ",
            prerequisite_match.group("requirements"),
        ).strip().rstrip(".")
        caster_requirement = re.search(
            r"\bcaster\s+level\s*(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
            requirements,
            re.IGNORECASE,
        )
        if caster_requirement is not None:
            required_level = int(caster_requirement.group("level"))
            caster_level_match = CASTER_LEVEL_IN_QUESTION.search(question)
            if caster_level_match is None:
                return (
                    f"Cannot establish from the details given. {feat_name} requires "
                    f"caster level {required_level}{ordinal_suffix(required_level)}; "
                    "the stated class level and character level do not establish a "
                    f"caster level. {citation(passage)}"
                )
            caster_level = int(caster_level_match.group("level"))
            if caster_level < required_level:
                return (
                    f"No. {feat_name} requires caster level "
                    f"{required_level}{ordinal_suffix(required_level)}, but the "
                    f"stated caster level is {caster_level}. {citation(passage)}"
                )

        class_requirements = [
            match
            for match in class_requirement_pattern.finditer(requirements)
            if match.group("class").strip().casefold() != "caster"
        ]
        for requirement_match in class_requirements:
            required_class = requirement_match.group("class").strip().casefold()
            required_level = int(requirement_match.group("level"))
            current_level = character_levels.get(required_class, 0)
            maximum_level = current_level + levels_available
            if maximum_level >= required_level:
                continue

            described_levels = ", ".join(
                f"{level} {class_name.title()}{' level' if level == 1 else ' levels'}"
                for class_name, level in character_levels.items()
            )
            missing_levels = required_level - maximum_level
            listed_requirements = []
            if proficiency_requirement_pattern.search(requirements):
                listed_requirements.append("proficiency with the selected weapon")
            selected_feat_names = {
                match.group("name").strip()
                for match in selected_feat_pattern.finditer(entry)
            }
            selected_feat_names.update(
                match.group("name").strip()
                for match in selected_weapon_feat_pattern.finditer(requirements)
            )
            listed_requirements.extend(
                f"{name} feat" for name in sorted(selected_feat_names)
            )
            listed_requirements.extend(
                f"{match.group('class').strip().title()} level "
                f"{match.group('level')}{ordinal_suffix(int(match.group('level')))}"
                for match in class_requirements
            )
            requirement_summary = ", ".join(listed_requirements)
            other_requirements_note = (
                " The question does not establish whether the character meets the "
                "other listed prerequisites."
                if len(listed_requirements) > 1
                else ""
            )
            return (
                f"No. {feat_name} requires {requirement_summary}. The described "
                f"character has {described_levels}; by character level "
                f"{selection_level}, they could have at most {maximum_level} "
                f"{required_class.title()} "
                f"{'level' if maximum_level == 1 else 'levels'}, which is "
                f"{missing_levels} short of the required {required_class.title()} "
                f"level {required_level}.{other_requirements_note} "
                f"{citation(passage)}"
            )
    return None


def ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def requested_class_level(question: str, class_names: set[str]) -> tuple[str, int] | None:
    for class_name in sorted(class_names, key=len, reverse=True):
        escaped_name = re.escape(class_name)
        patterns = (
            re.compile(
                rf"\b(?P<level>\d{{1,2}})(?:st|nd|rd|th)?[\s-]+level"
                rf"(?:\s+class)?\s+(?:the\s+)?{escaped_name}\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\blevel\s+(?P<level>\d{{1,2}})(?:st|nd|rd|th)?"
                rf"\s+(?:class\s+)?(?:the\s+)?{escaped_name}\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b{escaped_name}\s+(?:at\s+)?(?:class\s+)?level\s+"
                r"(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
                re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            match = pattern.search(question)
            if match:
                return class_name, int(match.group("level"))
    return None


def requested_class_levels(question: str, class_names: set[str]) -> list[tuple[str, int]]:
    requested = []
    for class_name in sorted(class_names, key=len, reverse=True):
        escaped_name = re.escape(class_name)
        patterns = (
            re.compile(
                rf"\b{escaped_name}\s+(?P<level>\d{{1,2}})(?:st|nd|rd|th)?\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b(?P<level>\d{{1,2}})(?:st|nd|rd|th)?[\s-]+"
                rf"(?:level\s+)?{escaped_name}\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b(?P<level>\d{{1,2}})(?:st|nd|rd|th)?[\s-]+level\s+"
                rf"(?:(?:[a-z][a-z'’-]*?)\s+){{1,2}}{escaped_name}\b",
                re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            match = pattern.search(question)
            if match:
                requested.append(
                    (
                        match.start(),
                        class_name,
                        int(match.group("level")),
                    )
                )
                break
    requested.sort(key=lambda item: item[0])
    return [(class_name, level) for _, class_name, level in requested]


def class_level_descriptors(question: str, class_names: set[str]) -> list[str]:
    descriptors = []
    for class_name in sorted(class_names, key=len, reverse=True):
        pattern = re.compile(
            rf"\b\d{{1,2}}(?:st|nd|rd|th)?[\s-]+level\s+"
            rf"(?P<descriptor>(?:[a-z][a-z'’-]*\s+){{1,2}})"
            rf"{re.escape(class_name)}\b",
            re.IGNORECASE,
        )
        for match in pattern.finditer(question):
            descriptor = match.group("descriptor").strip()
            if descriptor and descriptor.casefold() not in {
                item.casefold() for item in descriptors
            }:
                descriptors.append(descriptor)
    return descriptors


def consistent_class_ability(passages: list[dict[str, Any]]) -> str | None:
    abilities = {
        passage["class_ability"].casefold()
        for passage in passages
        if isinstance(passage.get("class_ability"), str)
        and passage["class_ability"].casefold() in {"intelligence", "wisdom", "charisma"}
    }
    return next(iter(abilities)) if len(abilities) == 1 else None


def spell_level_label(level: int) -> str:
    return f"{ordinal_number(level)}-level"


def ordinal_number(value: int) -> str:
    if value == 0:
        return "0th"
    suffix = (
        "th"
        if 10 <= value % 100 <= 13
        else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    )
    return f"{value}{suffix}"


def spell_slot_counts(text: str) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for match in SPELL_SLOT_COUNT_PATTERN.finditer(text):
        value = match.group("count").lower()
        counts[match.group("label").lower()] = (
            None if value == "unavailable" else int(value)
        )
    return counts


def spell_slot_count(text: str, spell_level: int) -> int | None:
    return spell_slot_counts(text).get(spell_level_label(spell_level))


def citation(passage: dict[str, Any]) -> str:
    return f"[{passage['source']}, PDF p. {passage['page']}]"


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

    attack_type = "ranged" if re.search(r"\branged\b", question, re.IGNORECASE) else "melee"
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
            and (
                range_match := ABILITY_SCORE_RANGE_PATTERN.search(passage["text"])
            )
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
            and (
                size_match := size_pattern.search(passage["text"])
            ) is not None
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
        f"[{source}, PDF p. {pages[0]}]"
        if len(pages) == 1
        else f"[{source}, PDF pp. {', '.join(pages)}]"
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
    citations = " ".join(
        dict.fromkeys(citation(row) for _, _, _, row in contributions)
    )
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
            match = BASE_SAVE_VALUE_PATTERN.search(text) if isinstance(text, str) else None
            if match is None:
                return None
            values = {
                save_name: int(match.group(save_name))
                for save_name in save_names
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


def structured_spell_slot_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if "spell" not in question.lower():
        return None

    spell_levels = list(SPELL_LEVEL_PATTERN.finditer(question))
    class_names = {
        passage["class_name"]
        for passage in passages
        if isinstance(passage.get("class_name"), str)
    }
    requested_classes = requested_class_levels(question, class_names)
    if not requested_classes:
        return None

    requested_level = (
        int(spell_levels[-1].group(1) or spell_levels[-1].group(2))
        if spell_levels
        else None
    )
    ability_scores = {}
    mentioned_abilities = set()
    for match in ABILITY_SCORE_PATTERN.finditer(question):
        ability = match.group("ability").lower()
        mentioned_abilities.add(ability)
        score = match.group("score_before") or match.group("score_after")
        if score is not None:
            ability_scores[ability] = int(score)

    class_summaries = []
    for requested_class, requested_class_level_value in requested_classes:
        class_rows = [
            passage
            for passage in passages
            if passage.get("kind") == "table_row"
            and isinstance(passage.get("class_name"), str)
            and passage["class_name"].casefold() == requested_class.casefold()
            and passage.get("class_level") == requested_class_level_value
        ]
        base_passage = next(iter(class_rows), None)
        if base_passage is None:
            return None

        caster_class = str(base_passage["class_name"])
        level = int(base_passage["class_level"])
        caster_ability = consistent_class_ability(class_rows)
        base_counts = spell_slot_counts(base_passage["text"])
        available_levels = {
            label: count for label, count in base_counts.items() if count is not None
        }
        if not available_levels:
            return None

        bonus_counts: dict[str, int | None] = {}
        bonus_citation = ""
        ability_note = ""
        if not isinstance(caster_ability, str):
            ability_note = (
                " The class's spellcasting ability could not be established from "
                "the indexed class description, so bonus spells were not applied."
            )
        elif caster_ability not in ability_scores:
            if mentioned_abilities and caster_ability not in mentioned_abilities:
                mentioned = ", ".join(
                    ability.capitalize() for ability in sorted(mentioned_abilities)
                )
                ability_note = (
                    f" {mentioned} does not grant bonus spells to a {caster_class}; "
                    f"bonus spells depend on {caster_ability.capitalize()}, "
                    "which was not provided."
                )
            else:
                ability_note = (
                    f" Total bonus spells depend on the caster's "
                    f"{caster_ability.capitalize()} score, which was not provided."
                )
        else:
            ability_score = ability_scores[caster_ability]
            expected_range = bonus_score_range(ability_score)
            bonus_passage = next(
                (
                    passage
                    for passage in passages
                    if passage.get("kind") == "bonus_spell_row"
                    and (
                        range_match := ABILITY_SCORE_RANGE_PATTERN.search(
                            passage["text"]
                        )
                    )
                    is not None
                    and (
                        int(range_match.group("low")),
                        int(range_match.group("high")),
                    )
                    == expected_range
                ),
                None,
            )
            if bonus_passage is None:
                ability_note = (
                    f" The bonus-spell row for {caster_ability} "
                    f"{ability_score} was not retrieved, so totals cannot be "
                    "confirmed."
                )
            else:
                bonus_counts = spell_slot_counts(bonus_passage["text"])
                bonus_citation = f" Bonus spells {citation(bonus_passage)}."

        labels = (
            [spell_level_label(requested_level)]
            if requested_level is not None
            else sorted(
                available_levels,
                key=lambda label: int(re.match(r"\d+", label).group()),
            )
        )
        summaries = []
        for label in labels:
            base_count = available_levels.get(label)
            if base_count is None:
                continue
            bonus_count = bonus_counts.get(label, 0)
            total = base_count + (bonus_count or 0)
            summary = f"{label}: {base_count} base"
            if bonus_counts:
                summary += f" + {bonus_count or 0} bonus = {total}"
            summaries.append(summary)
        if not summaries:
            return None

        class_summaries.append(
            f"For a {ordinal_number(level)}-level {caster_class}, spells per day are "
            + "; ".join(summaries)
            + f" {citation(base_passage)}."
            + bonus_citation
            + ability_note
        )

    return " ".join(class_summaries)
