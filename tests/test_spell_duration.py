from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as rag_app
from rag import rag_retrieval, rag_rules

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"
HASTE_DESCRIPTION = {
    "kind": "page_text",
    "source": SOURCE,
    "page": 177,
    "text": (
        "Haste Transmutation Level:Brd 3, Sor/Wiz 3 Components:V, S, M "
        "Casting Time:1 standard action Range:Close (25 ft. + 5 ft./2 levels) "
        "Targets:One creature/level, no two of which can be more than 30 ft. "
        "apart Duration: 1 round/level Saving Throw:Fortitude negates "
        "(harmless) Spell Resistance:Yes (harmless)"
    ),
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    with TestClient(rag_app.app) as test_client:
        yield test_client
    rag_app.app.dependency_overrides.clear()


def test_haste_duration_is_reported_as_one_round_per_level() -> None:
    answer = rag_rules.structured_spell_duration_answer(
        "What is the duration of haste?",
        [HASTE_DESCRIPTION],
    )

    assert answer == (
        "The duration of Haste is 1 round/level. "
        f"[{SOURCE}, PDF p. 177]"
    )


def test_haste_duration_is_scaled_to_explicit_caster_level() -> None:
    answer = rag_rules.structured_spell_duration_answer(
        "What is the duration of haste for a 10th level caster?",
        [HASTE_DESCRIPTION],
    )

    assert answer == (
        "The duration of Haste is 10 rounds. "
        f"[{SOURCE}, PDF p. 177]"
    )


def test_duration_question_with_missing_rule_does_not_guess() -> None:
    answer = rag_rules.structured_spell_duration_answer(
        "What is the duration of haste?",
        [{"source": SOURCE, "page": 177, "text": "Haste is a transmutation spell."}],
    )

    assert answer is not None
    assert answer.startswith("Cannot establish the duration of Haste")


def test_api_answers_haste_duration_without_calling_chat_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_app, "retrieve", lambda _question: [HASTE_DESCRIPTION])
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "The explicit duration field should be answered deterministically"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [
                {"role": "user", "content": "What is the duration of haste?"}
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert "1 round/level" in answer
    assert "typically 1 round" not in answer


def test_api_scales_haste_duration_for_caster_level_without_calling_chat_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_app, "retrieve", lambda _question: [HASTE_DESCRIPTION])
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "The explicit duration field should be scaled deterministically"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is the duration of haste for a 10th level caster?"
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert "10 rounds" in answer
    assert "hours" not in answer


def test_duration_retrieval_targets_named_spell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_point = SimpleNamespace(
        id="haste-body",
        payload={
            "kind": "page_text",
            "source": SOURCE,
            "page": 240,
            "text": "Multiple haste effects don't stack. Haste dispels and counters slow.",
        },
        score=1.0,
    )
    duration_point = SimpleNamespace(
        id="haste-duration",
        payload={**HASTE_DESCRIPTION, "page": 240},
        score=1.0,
    )
    query_calls = 0
    scroll_calls = 0
    embedded_queries: list[str] = []

    def fake_query_points(**_kwargs: object) -> SimpleNamespace:
        nonlocal query_calls
        query_calls += 1
        return SimpleNamespace(points=[] if query_calls == 1 else [body_point])

    def fake_scroll(**_kwargs: object) -> tuple[list[SimpleNamespace], None]:
        nonlocal scroll_calls
        scroll_calls += 1
        return [duration_point], None

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
    monkeypatch.setattr(rag_retrieval.qdrant, "scroll", fake_scroll)

    passages = rag_retrieval.retrieve("What is the duration of haste?")

    assert query_calls == 2
    assert scroll_calls == 1
    assert "haste spell duration" in embedded_queries[1].casefold()
    assert len(passages) == 2
    answer = rag_rules.structured_spell_duration_answer(
        "What is the duration of haste?",
        passages,
    )
    assert answer == (
        "The duration of Haste is 1 round/level. "
        f"[{SOURCE}, PDF p. 240]"
    )
