from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as rag_app
from rag import rag_retrieval, rag_rules

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"

FIGHTER_FEAT_RULES = {
    "kind": "page_text",
    "source": SOURCE,
    "page": 39,
    "text": (
        "Bonus Feats: At 1st level, a fighter gets a bonus combat-oriented "
        "feat in addition to the feat that any 1st-level character gets and "
        "the bonus feat granted to a human character. The fighter gains an "
        "additional bonus feat at 2nd level and every two fighter levels "
        "thereafter (4th, 6th, 8th, 10th, 12th, 14th, 16th, 18th, and 20th). "
        "These bonus feats are in addition to the feat that a character of "
        "any class gets every three levels."
    ),
}

FIGHTER_FEAT_RULE_CHUNKS = [
    {
        "kind": "page_text",
        "source": SOURCE,
        "page": 39,
        "text": (
            "Bonus Feats:At 1st level, a fighter gets a bonus combat-oriented "
            "feat in addition to the feat that are typically members of warrior "
            "families who any 1st-level character gets and the bonus trace their "
            "lineages feat granted to a human character. The fighter gains an "
            "additional bonus feat fighters at 2nd level and every two fighter "
            "skilled with swords levels thereafter."
        ),
    },
    {
        "kind": "page_text",
        "source": SOURCE,
        "page": 39,
        "text": (
            "These bonus feats are in addition to the feat that a character of "
            "any class gets every three levels."
        ),
    },
]

WIZARD_FEAT_RULES = [
    {
        "kind": "page_text",
        "source": SOURCE,
        "page": 57,
        "text": (
            "At 1st level, a wizard gains Scribe Scroll as a bonus feat. "
            "At 5th level, a wizard gains a bonus feat, and gains another "
            "bonus feat every five levels thereafter."
        ),
    },
    {
        "kind": "page_text",
        "source": SOURCE,
        "page": 90,
        "text": (
            "A character gains a feat at 1st level and another feat every "
            "three levels thereafter."
        ),
    },
    {
        "kind": "page_text",
        "source": SOURCE,
        "page": 12,
        "text": "Every human gets one extra feat at 1st level.",
    },
]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    with TestClient(rag_app.app) as test_client:
        yield test_client
    rag_app.app.dependency_overrides.clear()


def ask(
    client: TestClient,
    question: str,
    passages: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    monkeypatch.setattr(rag_app, "retrieve", lambda _question: passages)
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "The supported feat progression should be counted deterministically"
        ),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": question}],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]["content"]


def test_first_level_human_fighter_counts_all_three_feat_sources(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = ask(
        client,
        "How many feats does a 1st level human fighter get?",
        FIGHTER_FEAT_RULE_CHUNKS,
        monkeypatch,
    )

    assert "gets 3 feats" in answer
    assert "1 character-level feat" in answer
    assert "1 Fighter bonus feat" in answer
    assert "1 Human bonus feat" in answer
    assert f"{SOURCE}, PDF p. 39" in answer


def test_ninth_level_human_wizard_counts_general_class_and_human_feats(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = ask(
        client,
        "How many feats does a 9th level human wizard get?",
        WIZARD_FEAT_RULES,
        monkeypatch,
    )

    assert "gets 7 feats" in answer
    assert "4 character-level feats" in answer
    assert "2 Wizard bonus feats" in answer
    assert "1 Human bonus feat" in answer
    assert f"{SOURCE}, PDF pp. 57, 90, 12" in answer


def test_multiclass_human_wizard_fighter_counts_each_progression() -> None:
    answer = rag_rules.structured_feat_count_answer(
        "How many feats does a human, 5th level wizard/4th level fighter get?",
        [
            {
                **WIZARD_FEAT_RULES[0],
                "page": 56,
                "text": (
                    "Table 3-18: The Wizard. 1st Summon familiar, Scribe Scroll; "
                    "5th Bonus feat; 10th Bonus feat; 15th Bonus feat; "
                    "20th Bonus feat."
                ),
            },
            FIGHTER_FEAT_RULES,
            WIZARD_FEAT_RULES[1],
            WIZARD_FEAT_RULES[2],
        ],
    )

    assert answer is not None
    assert "gets 10 feats" in answer
    assert "4 character-level feats" in answer
    assert "2 Wizard bonus feats" in answer
    assert "3 Fighter bonus feats" in answer
    assert "1 Human bonus feat" in answer


def test_multiclass_feat_retrieval_targets_each_class_progression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wizard_levels = (1, 5, 10, 15, 20)
    wizard_points = [
        SimpleNamespace(
            id=f"wizard-level-{level}",
            payload={
                "kind": "class_progression_row",
                "source": SOURCE,
                "page": 56,
                "text": (
                    "Table 3–18: The Wizard. Wizard class level "
                    f"{level}; Original table row: {level}{'st' if level == 1 else 'th'} "
                    f"{'Scribe Scroll' if level == 1 else 'Bonus feat'}."
                ),
            },
            score=1.0,
        )
        for level in wizard_levels
    ]
    fighter_point = SimpleNamespace(
        id="fighter-feat-rules",
        payload=FIGHTER_FEAT_RULES,
        score=1.0,
    )
    general_point = SimpleNamespace(
        id="character-feat-rules",
        payload=WIZARD_FEAT_RULES[1],
        score=1.0,
    )
    human_point = SimpleNamespace(
        id="human-feat-rules",
        payload=WIZARD_FEAT_RULES[2],
        score=1.0,
    )
    query_calls = 0

    def fake_query_points(**_kwargs: object) -> SimpleNamespace:
        nonlocal query_calls
        query_calls += 1
        results = {
            1: [],
            2: [wizard_points[1]],
            3: [fighter_point],
            4: [general_point],
            5: [human_point],
        }
        return SimpleNamespace(points=results.get(query_calls, []))

    embedded_queries: list[str] = []
    monkeypatch.setattr(
        rag_retrieval,
        "embed",
        lambda query: embedded_queries.append(query) or [0.0],
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "query_points",
        fake_query_points,
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "scroll",
        lambda **_kwargs: ([*wizard_points, fighter_point], None),
    )

    passages = rag_retrieval.retrieve(
        "How many feats does a human, 5th level wizard/4th level fighter get?"
    )
    answer = rag_rules.structured_feat_count_answer(
        "How many feats does a human, 5th level wizard/4th level fighter get?",
        passages,
    )

    assert len(embedded_queries) == 5
    assert any("Wizard Scribe Scroll" in query for query in embedded_queries)
    assert any("Fighter bonus feats" in query for query in embedded_queries)
    assert any("every three character levels" in query for query in embedded_queries)
    assert any("human race one extra feat" in query for query in embedded_queries)
    assert answer is not None
    assert "gets 10 feats" in answer


def test_ninth_level_human_wizard_counts_from_extracted_table_wording() -> None:
    answer = rag_rules.structured_feat_count_answer(
        "How many feats does a 9th level human wizard get?",
        [
            {
                "source": SOURCE,
                "page": 56,
                "kind": "table",
                "text": (
                    "Table 3-18: The Wizard Special Level. 1st +0 Summon "
                    "familiar Scribe Scroll 2nd +1 5th +2 Bonus feat 6th +3 "
                    "10th +5 Bonus feat 11th +5 15th +7 Bonus feat 16th +8 "
                    "20th +10 Bonus feat."
                ),
            },
            {
                "source": SOURCE,
                "page": 23,
                "kind": "table_data_row",
                "text": (
                    "Table columns: Table 3-2: Experience and Level-Dependent "
                    "Benefits | Character Level | Feats | Increases. Table row: "
                    "9th +6 +3 +9/+4 | +6/+1 | +4 | 9th 36,000 12 | 6 | 4th | —."
                ),
            },
            {
                "source": SOURCE,
                "page": 14,
                "text": "Humans are adaptable. 1 extra feat at 1st level.",
            },
        ],
    )

    assert answer is not None
    assert "gets 7 feats" in answer
    assert "4 character-level feats" in answer
    assert "2 Wizard bonus feats" in answer
    assert "1 Human bonus feat" in answer
    assert "PDF pp. 56, 23, 14" in answer


def test_wizard_feat_count_uses_level_and_race_from_question() -> None:
    answer = rag_rules.structured_feat_count_answer(
        "How many feats does a wizard at class level 5 get?",
        WIZARD_FEAT_RULES[:2],
    )

    assert answer is not None
    assert "gets 4 feats" in answer
    assert "2 character-level feats" in answer
    assert "2 Wizard bonus feats" in answer
    assert "Human bonus feat" not in answer


@pytest.mark.parametrize(
    ("level", "expected_total", "character_feats", "fighter_feats"),
    [
        (2, 4, 1, 2),
        (4, 6, 2, 3),
        (6, 8, 3, 4),
    ],
)
def test_higher_level_human_fighter_counts_both_progressions(
    level: int,
    expected_total: int,
    character_feats: int,
    fighter_feats: int,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = ask(
        client,
        f"How many feats does a human fighter at level {level} get?",
        [FIGHTER_FEAT_RULES],
        monkeypatch,
    )

    assert f"gets {expected_total} feats" in answer
    assert f"{character_feats} character-level" in answer
    assert f"{fighter_feats} Fighter bonus" in answer
    assert "1 Human bonus feat" in answer


def test_nonhuman_fighter_count_does_not_add_human_bonus_feat() -> None:
    answer = rag_rules.structured_fighter_feat_count_answer(
        "How many feats does a 1st level elf fighter get?",
        [FIGHTER_FEAT_RULES],
    )

    assert answer is not None
    assert "gets 2 feats" in answer
    assert "Human bonus feat" not in answer


def test_missing_feat_progression_evidence_does_not_guess() -> None:
    answer = rag_rules.structured_fighter_feat_count_answer(
        "How many feats does a 1st level human fighter get?",
        [
            {
                "source": SOURCE,
                "page": 39,
                "text": "Fighters are proficient with all simple weapons.",
            }
        ],
    )

    assert answer is not None
    assert answer.startswith("Cannot establish")
    assert "Human bonus feat" in answer


def test_feat_count_retrieval_targets_all_three_feat_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved_point = SimpleNamespace(
        id="fighter-feat-rules",
        payload=FIGHTER_FEAT_RULE_CHUNKS[0],
        score=1.0,
    )
    sibling_point = SimpleNamespace(
        id="fighter-feat-rule-sibling",
        payload=FIGHTER_FEAT_RULE_CHUNKS[1],
        score=1.0,
    )
    query_calls = 0

    def fake_query_points(**_kwargs: object) -> SimpleNamespace:
        nonlocal query_calls
        query_calls += 1
        return SimpleNamespace(points=[retrieved_point] if query_calls == 1 else [])

    embedded_queries: list[str] = []
    monkeypatch.setattr(
        rag_retrieval,
        "embed",
        lambda query: embedded_queries.append(query) or [0.0],
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "query_points",
        fake_query_points,
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "scroll",
        lambda **_kwargs: ([retrieved_point, sibling_point], None),
    )

    passages = rag_retrieval.retrieve(
        "How many feats does a 1st level human fighter get?"
    )

    assert len(embedded_queries) == 4
    assert "Fighter bonus feats at 1st level" in embedded_queries[1]
    assert "feat at 1st level" in embedded_queries[2]
    assert "every three character levels" in embedded_queries[2]
    assert "Table 3-2" in embedded_queries[2]
    assert "human race one extra feat" in embedded_queries[3]
    assert len(passages) == 2
    assert {passage["text"] for passage in passages} == {
        chunk["text"] for chunk in FIGHTER_FEAT_RULE_CHUNKS
    }


def test_feat_count_retrieval_targets_wizard_progression_and_page_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wizard_point = SimpleNamespace(
        id="wizard-feat-rules",
        payload=WIZARD_FEAT_RULES[0],
        score=1.0,
    )
    sibling_point = SimpleNamespace(
        id="wizard-feat-rule-sibling",
        payload={
            **WIZARD_FEAT_RULES[0],
            "text": "A wizard gains another bonus feat every five levels thereafter.",
        },
        score=1.0,
    )
    query_calls = 0
    general_point = SimpleNamespace(
        id="character-feat-rules",
        payload={
            **WIZARD_FEAT_RULES[1],
            "text": (
                "A character gains a feat at 1st level and another feat every "
                "three levels thereafter."
            ),
        },
        score=1.0,
    )
    human_point = SimpleNamespace(
        id="human-feat-rules",
        payload=WIZARD_FEAT_RULES[2],
        score=1.0,
    )

    def fake_query_points(**_kwargs: object) -> SimpleNamespace:
        nonlocal query_calls
        query_calls += 1
        points_by_call = {
            1: [],
            2: [wizard_point],
            3: [general_point],
            4: [human_point],
        }
        return SimpleNamespace(points=points_by_call.get(query_calls, []))

    embedded_queries: list[str] = []
    monkeypatch.setattr(
        rag_retrieval,
        "embed",
        lambda query: embedded_queries.append(query) or [0.0],
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "query_points",
        fake_query_points,
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "scroll",
        lambda **_kwargs: ([wizard_point, sibling_point], None),
    )

    passages = rag_retrieval.retrieve(
        "How many feats does a 9th level human wizard get?"
    )

    assert len(embedded_queries) == 4
    assert "Scribe Scroll" in embedded_queries[1]
    assert "every five levels" in embedded_queries[1]
    assert "every three character levels" in embedded_queries[2]
    assert "human race one extra feat" in embedded_queries[3]
    assert {passage["text"] for passage in passages} >= {
        wizard_point.payload["text"],
        sibling_point.payload["text"],
        general_point.payload["text"],
        human_point.payload["text"],
    }


def test_wizard_feat_count_expands_all_class_progression_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels = (1, 5, 10, 15, 20)
    ordinals = {1: "1st", 5: "5th", 10: "10th", 15: "15th", 20: "20th"}
    wizard_points = [
        SimpleNamespace(
            id=f"wizard-level-{level}",
            payload={
                "kind": "class_progression_row",
                "source": SOURCE,
                "page": 56,
                "text": (
                    "Table Table 3–18: The Wizard. Wizard class level "
                    f"{level}; Original table row: {ordinals[level]} "
                    f"{'Scribe Scroll' if level == 1 else 'Bonus feat'}."
                ),
            },
            score=1.0,
        )
        for level in levels
    ]
    general_point = SimpleNamespace(
        id="character-feat-rules",
        payload={
            "kind": "page_text",
            "source": SOURCE,
            "page": 88,
            "text": (
                "A character gets a feat at 1st level and another feat every "
                "three levels thereafter."
            ),
        },
        score=1.0,
    )
    human_point = SimpleNamespace(
        id="human-feat-rules",
        payload={
            "kind": "page_text",
            "source": SOURCE,
            "page": 14,
            "text": "Humans have 1 extra feat at 1st level.",
        },
        score=1.0,
    )
    query_calls = 0

    def fake_query_points(**_kwargs: object) -> SimpleNamespace:
        nonlocal query_calls
        query_calls += 1
        results = {
            1: [],
            2: [wizard_points[1]],
            3: [general_point],
            4: [human_point],
        }
        return SimpleNamespace(points=results.get(query_calls, []))

    monkeypatch.setattr(
        rag_retrieval,
        "embed",
        lambda _query: [0.0],
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "query_points",
        fake_query_points,
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "scroll",
        lambda **_kwargs: ([*wizard_points, wizard_points[0]], None),
    )

    passages = rag_retrieval.retrieve(
        "How many feats does a 9th level human wizard get?"
    )
    answer = rag_rules.structured_feat_count_answer(
        "How many feats does a 9th level human wizard get?",
        passages,
    )

    assert {passage["text"] for passage in passages} >= {
        point.payload["text"] for point in wizard_points
    }
    assert answer is not None
    assert "gets 7 feats" in answer
    assert "2 Wizard bonus feats" in answer


def test_split_feat_rule_chunks_are_combined_for_counting() -> None:
    answer = rag_rules.structured_fighter_feat_count_answer(
        "How many feats does a 1st level human fighter get?",
        [
            {
                **chunk,
                "score": 1.0,
            }
            for chunk in FIGHTER_FEAT_RULE_CHUNKS
        ],
    )

    assert answer is not None
    assert "gets 3 feats" in answer
    assert "PDF p. 39" in answer


def test_feat_count_citation_omits_unrelated_retrieved_pages() -> None:
    answer = rag_rules.structured_fighter_feat_count_answer(
        "How many feats does a 1st level human fighter get?",
        [
            FIGHTER_FEAT_RULES,
            {
                "source": SOURCE,
                "page": 88,
                "text": "Fighters may select combat feats listed in this chapter.",
            },
        ],
    )

    assert answer is not None
    assert "PDF p. 39" in answer
    assert "PDF p. 88" not in answer
