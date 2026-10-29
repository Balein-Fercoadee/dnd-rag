"""rule character helpers."""

import re

from typing import Any

from .rag_config import (
    CASTER_LEVEL_IN_QUESTION,
    CHARACTER_CLASS_LEVEL_PATTERN,
    FEAT_QUALIFICATION_QUESTION,
    FEAT_COUNT_QUESTION,
    FEAT_SELECTION_LEVEL_PATTERN,
    STARTING_MONEY_QUESTION,
)

from .rule_parsing import (
    citation,
    feat_name_from_question,
    ordinal_suffix,
)

CORE_CLASSES = (
    "Barbarian",
    "Bard",
    "Cleric",
    "Druid",
    "Fighter",
    "Monk",
    "Paladin",
    "Ranger",
    "Rogue",
    "Sorcerer",
    "Wizard",
)


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
        entry = text[heading_match.end() :]
        next_heading = next_feat_heading.search(entry)
        if next_heading is not None:
            entry = entry[: next_heading.start()]
        prerequisite_match = prerequisite_pattern.search(entry)
        if prerequisite_match is None:
            continue

        requirements = (
            re.sub(
                r"\s+",
                " ",
                prerequisite_match.group("requirements"),
            )
            .strip()
            .rstrip(".")
        )
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


def structured_fighter_feat_count_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if FEAT_COUNT_QUESTION.search(question) is None:
        return None

    level_patterns = (
        re.compile(
            r"\b(?P<level>\d{1,2})(?:st|nd|rd|th)?[\s-]+level\s+"
            r"(?:[a-z][a-z-]*\s+)?fighter\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:[a-z][a-z-]*\s+)?fighter\s+(?:at\s+)?(?:class\s+)?level\s+"
            r"(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
            re.IGNORECASE,
        ),
    )
    level_match = next(
        (match for pattern in level_patterns if (match := pattern.search(question))),
        None,
    )
    if level_match is None:
        return None

    fighter_level = int(level_match.group("level"))
    if fighter_level < 1:
        return None
    is_human = re.search(r"\bhuman\b", question, re.IGNORECASE) is not None

    passages_by_source: dict[str, list[dict[str, Any]]] = {}
    for passage in passages:
        source = passage.get("source")
        if isinstance(source, str) and isinstance(passage.get("text"), str):
            passages_by_source.setdefault(source, []).append(passage)

    for source, source_passages in passages_by_source.items():
        compact_text = re.sub(
            r"\s+",
            " ",
            " ".join(str(passage["text"]) for passage in source_passages),
        )
        has_first_level_fighter_feat = re.search(
            r"\bAt 1st level, a fighter gets a bonus\b", compact_text, re.IGNORECASE
        )
        has_fighter_progression = re.search(
            r"\bevery two fighter\b", compact_text, re.IGNORECASE
        ) and re.search(r"\blevels thereafter\b", compact_text, re.IGNORECASE)
        has_general_progression = re.search(
            r"\bgets every three levels\b", compact_text, re.IGNORECASE
        )
        has_first_character_feat = re.search(
            r"\bany 1st-level character gets\b",
            compact_text,
            re.IGNORECASE,
        )
        has_human_bonus = re.search(
            r"\bbonus\b.{0,300}\bgranted to a human character\b",
            compact_text,
            re.IGNORECASE | re.DOTALL,
        )
        if not all(
            (
                has_first_level_fighter_feat,
                has_fighter_progression,
                has_general_progression,
                has_first_character_feat,
            )
        ):
            continue
        if is_human and has_human_bonus is None:
            continue

        character_level_feats = 1 + fighter_level // 3
        fighter_bonus_feats = 1 + fighter_level // 2
        human_bonus_feats = int(is_human)
        total_feats = character_level_feats + fighter_bonus_feats + human_bonus_feats
        breakdown = (
            f"{character_level_feats} character-level "
            f"{'feat' if character_level_feats == 1 else 'feats'}, "
            f"{fighter_bonus_feats} Fighter bonus "
            f"{'feat' if fighter_bonus_feats == 1 else 'feats'}"
        )
        if is_human:
            breakdown += ", and 1 Human bonus feat"
        evidence_patterns = (
            r"\bAt 1st level, a fighter gets a bonus",
            r"\bevery two fighter\b",
            r"\bgets every three levels\b",
            r"\bany 1st-level character gets\b",
            r"\bgranted to a human character\b",
        )
        pages = list(
            dict.fromkeys(
                str(passage["page"])
                for passage in source_passages
                if passage.get("page") is not None
                and any(
                    re.search(pattern, str(passage["text"]), re.IGNORECASE)
                    for pattern in evidence_patterns
                )
            )
        )
        page_citation = (
            f"[{source}, PDF p. {pages[0]}]"
            if len(pages) == 1
            else f"[{source}, PDF pp. {', '.join(pages)}]"
        )
        return (
            f"A {fighter_level}{ordinal_suffix(fighter_level)}-level "
            f"{'human ' if is_human else ''}Fighter gets {total_feats} feats: "
            f"{breakdown}. {page_citation}"
        )

    return (
        "Cannot establish the total feat count because the retrieved passages "
        "do not show the character-level and Fighter bonus-feat progressions"
        + (" and the Human bonus feat." if is_human else ".")
    )


def structured_starting_money_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if STARTING_MONEY_QUESTION.search(question) is None:
        return None

    requested_classes = [
        class_name
        for class_name in CORE_CLASSES
        if re.search(rf"\b{class_name}\b", question, re.IGNORECASE)
    ]
    if len(requested_classes) != 1:
        return None
    requested_class = requested_classes[0]
    class_row_pattern = re.compile(
        rf"\b{re.escape(requested_class)}\b[^\d\n]{{0,40}}"
        r"(?P<count>\d+)\s*d\s*(?P<sides>\d+)"
        r"(?:\s*[x×]\s*(?P<multiplier>\d+)\b)?",
        re.IGNORECASE,
    )

    for passage in passages:
        text = passage.get("text")
        if (
            not isinstance(text, str)
            or re.search(r"\bstarting\s+(?:gold|money)\b", text, re.IGNORECASE) is None
        ):
            continue
        match = class_row_pattern.search(text)
        if match is None:
            continue

        count = int(match.group("count"))
        sides = int(match.group("sides"))
        multiplier_match = match.group("multiplier")
        if multiplier_match is None:
            average_match = re.match(
                r"\s*\((?P<gp>\d+)\s*gp" r"(?:\s*,\s*(?P<sp>\d+)\s*sp)?\)",
                text[match.end() :],
                re.IGNORECASE,
            )
            if average_match is None:
                continue
            printed_average = int(average_match.group("gp")) + (
                int(average_match.group("sp") or 0) / 10
            )
            base_average = count * (sides + 1) / 2
            inferred_multiplier = printed_average / base_average
            if not inferred_multiplier.is_integer():
                continue
            multiplier = int(inferred_multiplier)
        else:
            multiplier = int(multiplier_match)

        minimum = count * multiplier
        maximum = count * sides * multiplier
        average = count * (sides + 1) * multiplier / 2
        dice_expression = f"{count}d{sides}"
        if multiplier != 1:
            dice_expression += f" × {multiplier}"
        return (
            f"{requested_class} starting money is {dice_expression} gp: "
            f"{minimum}–{maximum} gp "
            f"(average {average:g} gp). {citation(passage)}"
        )

    return (
        f"Cannot establish the starting money for a {requested_class}; "
        "the matching starting-gold table row was not retrieved."
    )
