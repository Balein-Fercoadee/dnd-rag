from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as rag_app
from rag import rag_retrieval, rag_rules

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"

FIREBALL_AND_FLAME_STRIKE = {
    "kind": "page_text",
    "source": SOURCE,
    "page": 232,
    "text": (
        "Fireball Evocation [Fire] Level: Sor/Wiz 3. "
        "A fireball spell is an explosion of flame and deals 1d6 points of "
        "fire damage per caster level (maximum 10d6) to every creature in "
        "the area. "
        "Flame Strike Evocation [Fire]. A flame strikeproduces a vertical "
        "column of divine fire. The spell deals 1d6 points of damage per "
        "caster level (maximum 15d6)."
    ),
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    with TestClient(rag_app.app) as test_client:
        yield test_client
    rag_app.app.dependency_overrides.clear()


def test_fireball_max_damage_uses_its_own_cap_not_flame_strikes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rag_app,
        "retrieve",
        lambda _question: [FIREBALL_AND_FLAME_STRIKE],
    )
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "The named spell's retrieved damage cap should be answered deterministically"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the max damage that Fireball can do?",
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert "maximum of 10d6" in answer
    assert "15d6" not in answer
    assert f"{SOURCE}, PDF p. 232" in answer


def test_flame_strike_uses_its_own_fifteen_d6_cap_with_extracted_wording() -> None:
    answer = rag_rules.structured_spell_maximum_damage_answer(
        "what is the max damage for flame strike?",
        [FIREBALL_AND_FLAME_STRIKE],
    )

    assert answer is not None
    assert "maximum of 15d6" in answer


def test_spell_name_is_extracted_from_maximum_damage_question() -> None:
    assert (
        rag_rules.spell_name_for_max_damage(
            "What is the max damage that Fireball can do?"
        )
        == "Fireball"
    )
    assert (
        rag_rules.spell_name_for_max_damage(
            "What is the maximum damage of Flame Strike?"
        )
        == "Flame Strike"
    )


def test_missing_spell_specific_cap_does_not_guess() -> None:
    answer = rag_rules.structured_spell_maximum_damage_answer(
        "What is the maximum damage that Fireball can do?",
        [
            {
                "source": SOURCE,
                "page": 232,
                "text": "Fireball is a spell that creates an explosion.",
            }
        ],
    )

    assert answer is not None
    assert answer.startswith("Cannot establish")


def test_max_damage_retrieval_targets_the_named_spell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = SimpleNamespace(
        id="fireball-description",
        payload=FIREBALL_AND_FLAME_STRIKE,
        score=1.0,
    )
    query_calls = 0
    embedded_queries: list[str] = []

    def fake_query_points(**_kwargs: object) -> SimpleNamespace:
        nonlocal query_calls
        query_calls += 1
        return SimpleNamespace(points=[] if query_calls == 1 else [point])

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

    passages = rag_retrieval.retrieve(
        "What is the max damage that Fireball can do?"
    )

    assert query_calls == 2
    assert "Fireball spell damage per caster level" in embedded_queries[1]
    assert len(passages) == 1
    assert passages[0]["text"] == FIREBALL_AND_FLAME_STRIKE["text"]
