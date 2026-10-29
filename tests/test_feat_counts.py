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

    assert len(embedded_queries) == 2
    assert "Fighter bonus feats at 1st level" in embedded_queries[1]
    assert "human bonus feat" in embedded_queries[1]
    assert len(passages) == 2
    assert {passage["text"] for passage in passages} == {
        chunk["text"] for chunk in FIGHTER_FEAT_RULE_CHUNKS
    }


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
