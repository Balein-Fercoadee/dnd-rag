"""rule spells helpers."""

import re

from typing import Any

from .rag_config import (
    ABILITY_SCORE_PATTERN,
    ABILITY_SCORE_RANGE_PATTERN,
    MAXIMUM_SPELL_DAMAGE_QUESTION,
    SPELL_DURATION_QUESTION,
    SPELL_LEVEL_PATTERN,
)

from .rule_parsing import (
    bonus_score_range,
    caster_level_for_duration,
    citation,
    consistent_class_ability,
    ordinal_number,
    requested_class_levels,
    spell_level_label,
    spell_name_for_duration,
    spell_name_for_max_damage,
    spell_slot_counts,
)


def structured_spell_maximum_damage_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if MAXIMUM_SPELL_DAMAGE_QUESTION.search(question) is None:
        return None
    spell_name = spell_name_for_max_damage(question)
    if spell_name is None:
        return None

    damage_cap_pattern = re.compile(
        r"maximum\s+(?P<maximum>\d+d\d+)\b" r"(?P<tail>(?:(?!\bmaximum\b)[^.]){0,100})",
        re.IGNORECASE,
    )
    spell_entry_pattern = re.compile(
        rf"\b{re.escape(spell_name)}\b",
        re.IGNORECASE,
    )
    candidates: list[tuple[re.Match[str], dict[str, Any]]] = []
    seen_candidates: set[tuple[int, int]] = set()
    for passage_index, passage in enumerate(passages):
        text = passage.get("text")
        if not isinstance(text, str):
            continue
        normalized_text = re.sub(r"\s+", " ", text)
        for spell_entry in spell_entry_pattern.finditer(normalized_text):
            entry_end = min(spell_entry.end() + 700, len(normalized_text))
            for match in damage_cap_pattern.finditer(
                normalized_text, spell_entry.end(), entry_end
            ):
                candidate_key = (passage_index, match.start())
                if candidate_key not in seen_candidates:
                    candidates.append((match, passage))
                    seen_candidates.add(candidate_key)
    associated_candidates = [
        (match, passage)
        for match, passage in candidates
        if re.search(
            r"\bto\s+(?:every|each|all)\b",
            match.group("tail"),
            re.IGNORECASE,
        )
    ]
    if len(associated_candidates) == 1:
        match, passage = associated_candidates[0]
        return (
            f"{spell_name.title()} can deal a maximum of "
            f"{match.group('maximum')}. {citation(passage)}"
        )
    if not associated_candidates and len(candidates) == 1:
        match, passage = candidates[0]
        return (
            f"{spell_name.title()} can deal a maximum of "
            f"{match.group('maximum')}. {citation(passage)}"
        )

    return (
        f"Cannot establish the maximum damage for {spell_name.title()} from "
        "the retrieved spell description."
    )


def structured_spell_duration_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if SPELL_DURATION_QUESTION.search(question) is None:
        return None
    spell_name = spell_name_for_duration(question)
    if spell_name is None:
        return None

    heading_pattern = re.compile(
        rf"\b{re.escape(spell_name)}\s+"
        r"(?:Abjuration|Conjuration|Divination|Enchantment|Evocation|"
        r"Illusion|Necromancy|Transmutation)\b",
        re.IGNORECASE,
    )
    duration_pattern = re.compile(
        r"\bDuration:\s*(?P<duration>.*?)"
        r"(?=\s+(?:Saving Throw|Spell Resistance):|$)",
        re.IGNORECASE,
    )
    candidates: set[tuple[str, str, int]] = set()
    for passage in passages:
        text = passage.get("text")
        source = passage.get("source")
        page = passage.get("page")
        if (
            not isinstance(text, str)
            or not isinstance(source, str)
            or not isinstance(page, int)
        ):
            continue
        normalized_text = re.sub(r"\s+", " ", text)
        for heading in heading_pattern.finditer(normalized_text):
            duration_match = duration_pattern.search(
                normalized_text, heading.end(), heading.end() + 700
            )
            if duration_match is not None:
                duration = duration_match.group("duration").strip().rstrip(" ,;")
                if duration:
                    candidates.add((duration, source, page))

    if len(candidates) != 1:
        return (
            f"Cannot establish the duration of {spell_name.title()} from "
            "the retrieved spell description."
        )

    duration, source, page = next(iter(candidates))
    caster_level = caster_level_for_duration(question)
    if caster_level is not None:
        per_level_match = re.fullmatch(
            r"(?P<amount>\d+)\s+(?P<unit>rounds?|minutes?|mins?\.?|"
            r"hours?|days?|weeks?|months?|years?)/level"
            r"(?:\s+\([^)]*\))?",
            duration,
            re.IGNORECASE,
        )
        if per_level_match is not None:
            amount = int(per_level_match.group("amount")) * caster_level
            unit = per_level_match.group("unit").lower().rstrip(".")
            unit = {
                "min": "minute",
                "mins": "minute",
            }.get(unit, unit.rstrip("s"))
            duration = f"{amount} {unit}{'' if amount == 1 else 's'}"
    return (
        f"The duration of {spell_name.title()} is {duration}. "
        f"[{source}, PDF p. {page}]"
    )


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
