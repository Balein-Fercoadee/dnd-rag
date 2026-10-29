"""Ollama embedding and Qdrant passage retrieval."""
from typing import Any

import requests
from fastapi import HTTPException
from qdrant_client import QdrantClient, models as qdrant_models

from .rag_config import (
    ABILITY_SCORE_PATTERN,
    ABILITY_SCORE_RANGE_PATTERN,
    ARMOR_CLASS_QUESTION,
    ATTACK_BONUS_QUESTION,
    BASE_ATTACK_BONUS_QUESTION,
    BASE_SAVE_QUESTION,
    CLASS_LEVEL_ROW_PATTERN,
    COLLECTION,
    EMBED_MODEL,
    FEAT_QUALIFICATION_QUESTION,
    OLLAMA_URL,
    QDRANT_URL,
    RETRIEVAL_LIMIT,
    BONUS_ROW_LIMIT,
)
from .rag_rules import (
    bonus_score_range,
    class_name_from_text,
    class_level_descriptors,
    consistent_class_ability,
    feat_name_from_question,
    requested_class_levels,
)

qdrant = QdrantClient(url=QDRANT_URL)

def embed(text: str) -> list[float]:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc


def class_name_from_point(point: Any) -> str | None:
    payload = point.payload or {}
    class_name = payload.get("class_name")
    if isinstance(class_name, str) and class_name.strip():
        return class_name.strip()
    text = payload.get("text")
    if isinstance(text, str):
        return class_name_from_text(text)
    return None


def retrieve_class_progression_rows(question: str) -> list[Any]:
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="kind",
                match=qdrant_models.MatchValue(value="class_progression_row"),
            )
        ]
    )
    points, next_offset = qdrant.scroll(
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
        points, next_offset = qdrant.scroll(
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


def retrieve_class_spell_row(question: str) -> list[Any]:
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="kind",
                match=qdrant_models.MatchValue(value="table_row"),
            )
        ]
    )
    points, next_offset = qdrant.scroll(
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
        points, next_offset = qdrant.scroll(
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


def retrieve_bonus_spell_rows(ability_score: int) -> list[Any]:
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
    points, next_offset = qdrant.scroll(
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
            if row_match and (
                int(row_match.group("low")),
                int(row_match.group("high")),
            ) == expected_range:
                matches.append(point)
                seen_sources.add(source)
                if len(matches) >= BONUS_ROW_LIMIT:
                    return matches

        if next_offset is None:
            break
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=next_offset,
            with_payload=True,
        )
    return matches


def retrieve(question: str) -> list[dict[str, Any]]:
    try:
        question_vector = embed(question)
        result = qdrant.query_points(
            collection_name=COLLECTION,
            query=question_vector,
            limit=RETRIEVAL_LIMIT,
            with_payload=True,
        )
        points = list(result.points)

        if FEAT_QUALIFICATION_QUESTION.search(question):
            feat_name = feat_name_from_question(question)
            feat_query = (
                f"{feat_name} feat prerequisites D&D 3.5"
                if feat_name
                else f"{question} D&D 3.5 feat prerequisites and requirements"
            )
            feat_result = qdrant.query_points(
                collection_name=COLLECTION,
                query=embed(feat_query),
                limit=RETRIEVAL_LIMIT,
                with_payload=True,
            )
            feat_points = list(feat_result.points)
            if feat_name:
                feat_points.sort(
                    key=lambda point: feat_name.casefold()
                    not in str((point.payload or {}).get("text", "")).casefold()
                )
            seen_ids = set()
            prioritized_points = []
            for point in [*feat_points, *points]:
                if point.id not in seen_ids:
                    prioritized_points.append(point)
                    seen_ids.add(point.id)
            points = prioritized_points

        has_spell_question = "spell" in question.lower()
        has_armor_class_question = ARMOR_CLASS_QUESTION.search(question) is not None
        has_bab_question = BASE_ATTACK_BONUS_QUESTION.search(question) is not None
        has_attack_bonus_question = ATTACK_BONUS_QUESTION.search(question) is not None
        has_base_save_question = BASE_SAVE_QUESTION.search(question) is not None
        if has_armor_class_question:
            armor_class_rules = qdrant.query_points(
                collection_name=COLLECTION,
                query=embed(
                    "D&D 3.5 Armor Class AC formula base 10 armor bonus "
                    "shield bonus size modifier Dexterity modifier"
                ),
                limit=RETRIEVAL_LIMIT,
                with_payload=True,
            )
            seen_ids = {point.id for point in points}
            for point in armor_class_rules.points:
                if point.id not in seen_ids:
                    points.append(point)
                    seen_ids.add(point.id)

        if has_attack_bonus_question:
            attack_rules = qdrant.query_points(
                collection_name=COLLECTION,
                query=embed(
                    "D&D 3.5 melee attack bonus formula base attack bonus "
                    "Strength modifier size modifier; ranged attack bonus "
                    "Dexterity modifier"
                ),
                limit=RETRIEVAL_LIMIT,
                with_payload=True,
            )
            seen_ids = {point.id for point in points}
            for point in attack_rules.points:
                if point.id not in seen_ids:
                    points.append(point)
                    seen_ids.add(point.id)

        class_points = retrieve_class_spell_row(question) if has_spell_question else []
        if class_points:
            class_point_ids = {point.id for point in class_points}
            points = class_points + [
                point for point in points if point.id not in class_point_ids
            ]

        if has_bab_question or has_attack_bonus_question or has_base_save_question:
            progression_points = retrieve_class_progression_rows(question)
            seen_ids = {point.id for point in points}
            points.extend(
                point for point in progression_points if point.id not in seen_ids
            )

            if has_attack_bonus_question:
                class_names = {
                    point_name
                    for point in progression_points
                    if (point_name := class_name_from_point(point))
                }
                for descriptor in class_level_descriptors(question, class_names):
                    size_rules = qdrant.query_points(
                        collection_name=COLLECTION,
                        query=embed(
                            f"D&D 3.5 {descriptor} creature size category "
                            "size modifier attack rolls"
                        ),
                        limit=RETRIEVAL_LIMIT,
                        with_payload=True,
                    )
                    for point in size_rules.points:
                        if point.id not in seen_ids:
                            points.append(point)
                            seen_ids.add(point.id)

        if has_spell_question and class_points:
            ability_scores = {
                match.group("ability").lower(): int(score)
                for match in ABILITY_SCORE_PATTERN.finditer(question)
                if (
                    score := match.group("score_before")
                    or match.group("score_after")
                )
                is not None
            }
            needed_scores = {
                ability_scores[ability]
                for point in class_points
                if isinstance(
                    ability := (point.payload or {}).get("class_ability"),
                    str,
                )
                and ability.casefold() in ability_scores
            }
            seen_ids = {point.id for point in points}
            for ability_score in needed_scores:
                targeted_points = retrieve_bonus_spell_rows(ability_score)
                points.extend(
                    point
                    for point in targeted_points
                    if point.id not in seen_ids
                )
                seen_ids.update(point.id for point in targeted_points)

        if has_attack_bonus_question:
            ability_scores = {
                int(score)
                for match in ABILITY_SCORE_PATTERN.finditer(question)
                if match.group("ability").casefold()
                in {"strength", "dexterity"}
                and (
                    score := match.group("score_before")
                    or match.group("score_after")
                )
                is not None
            }
            seen_ids = {point.id for point in points}
            for ability_score in ability_scores:
                targeted_points = retrieve_bonus_spell_rows(ability_score)
                points.extend(
                    point
                    for point in targeted_points
                    if point.id not in seen_ids
                )
                seen_ids.update(point.id for point in targeted_points)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant retrieval failed: {exc}") from exc

    passages = []
    for point in points:
        payload = point.payload or {}
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        passages.append(
            {
                "source": str(payload.get("source", "Unknown source")),
                "page": payload.get("page", "unknown"),
                "kind": payload.get("kind"),
                "text": text.strip(),
                "score": getattr(point, "score", None),
                "class_name": payload.get("class_name") or class_name_from_text(text),
                "class_level": payload.get("class_level")
                or (
                    int(match.group("level"))
                    if (match := CLASS_LEVEL_ROW_PATTERN.search(text))
                    else None
                ),
                "class_ability": payload.get("class_ability"),
                "base_attack_bonus": payload.get("base_attack_bonus"),
                "fortitude_save": payload.get("fortitude_save"),
                "reflex_save": payload.get("reflex_save"),
                "will_save": payload.get("will_save"),
            }
        )
    return passages
