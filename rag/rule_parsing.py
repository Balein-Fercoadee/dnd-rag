"""rule parsing helpers."""

import re

from typing import Any

from fastapi import HTTPException

from .rag_config import (
    CASTER_LEVEL_IN_QUESTION,
    CLASS_TABLE_NAME_PATTERN,
    FEAT_NAME_IN_QUESTION,
    SPELL_SLOT_COUNT_PATTERN,
)


def bonus_score_range(ability_score: int) -> tuple[int, int]:
    if ability_score < 1 or ability_score > 99:
        raise HTTPException(
            status_code=400, detail="Ability score is outside supported range"
        )
    low = (
        ability_score
        if ability_score == 1 or ability_score % 2 == 0
        else ability_score - 1
    )
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


def ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def requested_class_level(
    question: str, class_names: set[str]
) -> tuple[str, int] | None:
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


def requested_class_levels(
    question: str, class_names: set[str]
) -> list[tuple[str, int]]:
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
        and passage["class_ability"].casefold()
        in {"intelligence", "wisdom", "charisma"}
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


def spell_name_for_max_damage(question: str) -> str | None:
    match = re.search(
        r"\b(?:max(?:imum)?|highest)\s+damage\s+"
        r"(?:(?:that|of|for)\s+)?"
        r"(?P<name>[a-z][a-z'’\-]*(?:\s+[a-z][a-z'’\-]*){0,3}?)"
        r"(?=\s+(?:spell\b|can\b|could\b|does\b|do\b|deal\b|cause\b|"
        r"inflict\b)|[?.!,]|$)",
        question,
        re.IGNORECASE,
    )
    return match.group("name").strip() if match is not None else None


def spell_name_for_duration(question: str) -> str | None:
    match = re.search(
        r"\b(?:duration|how long)\s+(?:of|for)\s+"
        r"(?P<name>[a-z][a-z'’\-]*(?:\s+[a-z][a-z'’\-]*){0,3}?)"
        r"(?=\s+(?:spell\b|last\b|does\b|is\b|for\s+(?:a\s+)?"
        r"\d{1,2}(?:st|nd|rd|th)?[\s-]+level\s+caster\b)|[?.!,]|$)",
        question,
        re.IGNORECASE,
    )
    return match.group("name").strip() if match is not None else None


def caster_level_for_duration(question: str) -> int | None:
    match = CASTER_LEVEL_IN_QUESTION.search(question)
    if match is None:
        match = re.search(
            r"\b(?P<level>\d{1,2})(?:st|nd|rd|th)?[\s-]+level\s+caster\b",
            question,
            re.IGNORECASE,
        )
    return int(match.group("level")) if match is not None else None
