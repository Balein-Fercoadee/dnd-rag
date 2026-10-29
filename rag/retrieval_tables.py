"""Structured class and ability rows retrieved from Qdrant."""

from typing import Any

from qdrant_client import models as qdrant_models

from .rag_config import (
    ABILITY_SCORE_RANGE_PATTERN,
    BONUS_ROW_LIMIT,
    CLASS_LEVEL_ROW_PATTERN,
    COLLECTION,
)
from .rag_rules import bonus_score_range, class_name_from_text, requested_class_levels
from .retrieval_backend import qdrant


def class_name_from_point(point: Any) -> str | None:
    payload = point.payload or {}
    class_name = payload.get("class_name")
    if isinstance(class_name, str) and class_name.strip():
        return class_name.strip()
    text = payload.get("text")
    if isinstance(text, str):
        return class_name_from_text(text)
    return None


def retrieve_class_progression_rows(
    question: str, *, client: Any = qdrant
) -> list[Any]:
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="kind",
                match=qdrant_models.MatchValue(value="class_progression_row"),
            )
        ]
    )
    points, next_offset = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=scroll_filter,
        limit=256,
        with_payload=True,
    )
    progression_rows = []
    while True:
        progression_rows.extend(points)
        if next_offset is None:
            break
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=next_offset,
            with_payload=True,
        )

    class_names = {
        name for point in progression_rows if (name := class_name_from_point(point))
    }
    requested = requested_class_levels(question, class_names)
    matches = []
    for class_name, class_level in requested:
        for point in progression_rows:
            point_name = class_name_from_point(point)
            if (
                point_name is not None
                and point_name.casefold() == class_name.casefold()
                and (point.payload or {}).get("class_level") == class_level
            ):
                matches.append(point)
                break
    return matches


def retrieve_class_spell_row(question: str, *, client: Any = qdrant) -> list[Any]:
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="kind",
                match=qdrant_models.MatchValue(value="table_row"),
            )
        ]
    )
    points, next_offset = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=scroll_filter,
        limit=256,
        with_payload=True,
    )
    matches = []
    class_rows = []
    while True:
        for point in points:
            class_rows.append(point)

        if next_offset is None:
            break
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=next_offset,
            with_payload=True,
        )

    class_names = {
        name for point in class_rows if (name := class_name_from_point(point))
    }
    requested = requested_class_levels(question, class_names)
    if not requested:
        return matches
    for requested_name, expected_level in requested:
        class_matches = 0
        class_sources = set()
        for point in class_rows:
            payload = point.payload or {}
            point_name = class_name_from_point(point)
            if point_name is None or point_name.casefold() != requested_name.casefold():
                continue
            level = payload.get("class_level")
            if not isinstance(level, int):
                text = payload.get("text")
                row_match = (
                    CLASS_LEVEL_ROW_PATTERN.search(text)
                    if isinstance(text, str)
                    else None
                )
                level = int(row_match.group("level")) if row_match else None
            source = payload.get("source")
            if level == expected_level and source not in class_sources:
                matches.append(point)
                class_sources.add(source)
                class_matches += 1
                if class_matches >= BONUS_ROW_LIMIT:
                    break
    return matches


def retrieve_bonus_spell_rows(ability_score: int, *, client: Any = qdrant) -> list[Any]:
    low, high = bonus_score_range(ability_score)
    expected_range = (low, high)
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="kind",
                match=qdrant_models.MatchValue(value="bonus_spell_row"),
            )
        ]
    )
    points, next_offset = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=scroll_filter,
        limit=256,
        with_payload=True,
    )
    matches = []
    seen_sources = set()
    while True:
        for point in points:
            payload = point.payload or {}
            text = payload.get("text")
            source = payload.get("source")
            if not isinstance(text, str) or source in seen_sources:
                continue
            row_match = ABILITY_SCORE_RANGE_PATTERN.search(text)
            if (
                row_match
                and (
                    int(row_match.group("low")),
                    int(row_match.group("high")),
                )
                == expected_range
            ):
                matches.append(point)
                seen_sources.add(source)
                if len(matches) >= BONUS_ROW_LIMIT:
                    return matches

        if next_offset is None:
            break
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=next_offset,
            with_payload=True,
        )
    return matches
