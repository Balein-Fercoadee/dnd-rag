"""Question-aware retrieval orchestration."""

import re
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from qdrant_client import models as qdrant_models

from .rag_config import (
    ABILITY_SCORE_PATTERN,
    ARMOR_CLASS_QUESTION,
    ATTACK_BONUS_QUESTION,
    BASE_ATTACK_BONUS_QUESTION,
    BASE_SAVE_QUESTION,
    CLASS_LEVEL_ROW_PATTERN,
    COLLECTION,
    FEAT_QUALIFICATION_QUESTION,
    FEAT_COUNT_QUESTION,
    MAXIMUM_SPELL_DAMAGE_QUESTION,
    SPELL_DURATION_QUESTION,
    RETRIEVAL_LIMIT,
    STARTING_MONEY_QUESTION,
)

from .rag_rules import (
    class_name_from_text,
    class_level_descriptors,
    feat_name_from_question,
    spell_name_for_max_damage,
    spell_name_for_duration,
)

from .retrieval_tables import (
    class_name_from_point,
    retrieve_bonus_spell_rows,
    retrieve_class_progression_rows,
    retrieve_class_spell_row,
)


def retrieve(
    question: str, *, client: Any, embed_fn: Callable[[str], list[float]]
) -> list[dict[str, Any]]:
    try:
        question_vector = embed_fn(question)
        result = client.query_points(
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
            feat_result = client.query_points(
                collection_name=COLLECTION,
                query=embed_fn(feat_query),
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

        if FEAT_COUNT_QUESTION.search(question):
            feat_count_rules = client.query_points(
                collection_name=COLLECTION,
                query=embed_fn(
                    "D&D 3.5 Fighter bonus feats at 1st level and every two "
                    "fighter levels thereafter, plus the feat every three "
                    "character levels and the human bonus feat"
                ),
                limit=RETRIEVAL_LIMIT,
                with_payload=True,
            )
            seen_ids = {point.id for point in points}
            for point in feat_count_rules.points:
                if point.id not in seen_ids:
                    points.append(point)
                    seen_ids.add(point.id)

            fighter_rule_pages = {
                (payload["source"], payload["page"])
                for point in points
                if isinstance(payload := (point.payload or {}), dict)
                and payload.get("kind") == "page_text"
                and isinstance(payload.get("source"), str)
                and isinstance(payload.get("page"), int)
                and re.search(
                    r"\bfighter\b.{0,80}\bbonus feats?\b|"
                    r"\bbonus feats?\b.{0,80}\bfighter\b",
                    str(payload.get("text", "")),
                    re.IGNORECASE | re.DOTALL,
                )
            }
            for source, page in fighter_rule_pages:
                page_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="source",
                            match=qdrant_models.MatchValue(value=source),
                        ),
                        qdrant_models.FieldCondition(
                            key="page",
                            match=qdrant_models.MatchValue(value=page),
                        ),
                        qdrant_models.FieldCondition(
                            key="kind",
                            match=qdrant_models.MatchValue(value="page_text"),
                        ),
                    ]
                )
                page_points, next_offset = client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=page_filter,
                    limit=256,
                    with_payload=True,
                )
                while True:
                    for point in page_points:
                        if point.id not in seen_ids:
                            points.append(point)
                            seen_ids.add(point.id)
                    if next_offset is None:
                        break
                    page_points, next_offset = client.scroll(
                        collection_name=COLLECTION,
                        scroll_filter=page_filter,
                        limit=256,
                        offset=next_offset,
                        with_payload=True,
                    )

        if MAXIMUM_SPELL_DAMAGE_QUESTION.search(question):
            spell_name = spell_name_for_max_damage(question)
            if spell_name is not None:
                spell_damage_rules = client.query_points(
                    collection_name=COLLECTION,
                    query=embed_fn(
                        f"D&D 3.5 {spell_name} spell damage per caster level "
                        "maximum damage cap"
                    ),
                    limit=RETRIEVAL_LIMIT,
                    with_payload=True,
                )
                seen_ids = {point.id for point in points}
                for point in spell_damage_rules.points:
                    if point.id not in seen_ids:
                        points.append(point)
                        seen_ids.add(point.id)

        if SPELL_DURATION_QUESTION.search(question):
            spell_name = spell_name_for_duration(question)
            if spell_name is not None:
                spell_duration_rules = client.query_points(
                    collection_name=COLLECTION,
                    query=embed_fn(
                        f"D&D 3.5 {spell_name} spell duration Duration field"
                    ),
                    limit=RETRIEVAL_LIMIT,
                    with_payload=True,
                )
                seen_ids = {point.id for point in points}
                for point in spell_duration_rules.points:
                    if point.id not in seen_ids:
                        points.append(point)
                        seen_ids.add(point.id)

                spell_name_pattern = re.compile(
                    rf"\b{re.escape(spell_name)}\b", re.IGNORECASE
                )
                spell_pages = {
                    (payload["source"], payload["page"])
                    for point in spell_duration_rules.points
                    if isinstance(payload := (point.payload or {}), dict)
                    and payload.get("kind") == "page_text"
                    and isinstance(payload.get("source"), str)
                    and isinstance(payload.get("page"), int)
                    and spell_name_pattern.search(str(payload.get("text", "")))
                }
                for source, page in spell_pages:
                    page_filter = qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="source",
                                match=qdrant_models.MatchValue(value=source),
                            ),
                            qdrant_models.FieldCondition(
                                key="page",
                                match=qdrant_models.MatchValue(value=page),
                            ),
                            qdrant_models.FieldCondition(
                                key="kind",
                                match=qdrant_models.MatchValue(value="page_text"),
                            ),
                        ]
                    )
                    page_points, next_offset = client.scroll(
                        collection_name=COLLECTION,
                        scroll_filter=page_filter,
                        limit=256,
                        with_payload=True,
                    )
                    while True:
                        for point in page_points:
                            if point.id not in seen_ids:
                                points.append(point)
                                seen_ids.add(point.id)
                        if next_offset is None:
                            break
                        page_points, next_offset = client.scroll(
                            collection_name=COLLECTION,
                            scroll_filter=page_filter,
                            limit=256,
                            offset=next_offset,
                            with_payload=True,
                        )

        has_spell_question = "spell" in question.lower()
        has_armor_class_question = ARMOR_CLASS_QUESTION.search(question) is not None
        has_bab_question = BASE_ATTACK_BONUS_QUESTION.search(question) is not None
        has_attack_bonus_question = ATTACK_BONUS_QUESTION.search(question) is not None
        has_base_save_question = BASE_SAVE_QUESTION.search(question) is not None
        has_starting_money_question = (
            STARTING_MONEY_QUESTION.search(question) is not None
        )
        if has_starting_money_question:
            starting_money_rules = client.query_points(
                collection_name=COLLECTION,
                query=embed_fn(
                    f"{question} D&D 3.5 Random Starting Gold table "
                    "class starting gold dice roll"
                ),
                limit=RETRIEVAL_LIMIT,
                with_payload=True,
            )
            seen_ids = {point.id for point in points}
            for point in starting_money_rules.points:
                if point.id not in seen_ids:
                    points.append(point)
                    seen_ids.add(point.id)

        if has_armor_class_question:
            armor_class_rules = client.query_points(
                collection_name=COLLECTION,
                query=embed_fn(
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
            attack_rules = client.query_points(
                collection_name=COLLECTION,
                query=embed_fn(
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

        class_points = (
            retrieve_class_spell_row(question, client=client)
            if has_spell_question
            else []
        )
        if class_points:
            class_point_ids = {point.id for point in class_points}
            points = class_points + [
                point for point in points if point.id not in class_point_ids
            ]

        if has_bab_question or has_attack_bonus_question or has_base_save_question:
            progression_points = retrieve_class_progression_rows(
                question, client=client
            )
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
                    size_rules = client.query_points(
                        collection_name=COLLECTION,
                        query=embed_fn(
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
                if (score := match.group("score_before") or match.group("score_after"))
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
                targeted_points = retrieve_bonus_spell_rows(
                    ability_score, client=client
                )
                points.extend(
                    point for point in targeted_points if point.id not in seen_ids
                )
                seen_ids.update(point.id for point in targeted_points)

        if has_attack_bonus_question:
            ability_scores = {
                int(score)
                for match in ABILITY_SCORE_PATTERN.finditer(question)
                if match.group("ability").casefold() in {"strength", "dexterity"}
                and (score := match.group("score_before") or match.group("score_after"))
                is not None
            }
            seen_ids = {point.id for point in points}
            for ability_score in ability_scores:
                targeted_points = retrieve_bonus_spell_rows(
                    ability_score, client=client
                )
                points.extend(
                    point for point in targeted_points if point.id not in seen_ids
                )
                seen_ids.update(point.id for point in targeted_points)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Qdrant retrieval failed: {exc}"
        ) from exc

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
