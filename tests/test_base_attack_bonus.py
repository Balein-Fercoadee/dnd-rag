import os
from collections.abc import Iterator

import pytest
import requests
from fastapi.testclient import TestClient

import app as rag_app
from rag import rag_rules

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"


def progression_row(
    class_name: str, level: int, bab: int, page: int
) -> dict[str, object]:
    return {
        "kind": "class_progression_row",
        "class_name": class_name,
        "class_level": level,
        "base_attack_bonus": bab,
        "source": SOURCE,
        "page": page,
        "text": f"{class_name} class level {level}; base attack bonus: +{bab}.",
    }


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
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": question}],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]["content"]


def test_multiclass_bab_adds_each_class_progression(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = ask(
        client,
        "What is the BAB of a barbarian 3/fighter 4 character?",
        [
            progression_row("Barbarian", 3, 3, 26),
            progression_row("Fighter", 4, 4, 40),
        ],
        monkeypatch,
    )

    assert "total base attack bonus is +7" in answer
    assert "Barbarian 3: +3; Fighter 4: +4" in answer


def test_multiclass_bab_query_finds_all_class_level_pairs() -> None:
    question = "What is the BAB of a barbarian 3/fighter 4 character?"

    assert rag_rules.requested_class_levels(
        question,
        {"Barbarian", "Fighter", "Wizard"},
    ) == [("Barbarian", 3), ("Fighter", 4)]


@pytest.mark.live
def test_live_api_multiclass_base_attack_bonus() -> None:
    base_url = os.getenv("RAG_TEST_BASE_URL")
    api_key = os.getenv("RAG_TEST_API_KEY")
    if not base_url or not api_key:
        pytest.skip("Set RAG_TEST_BASE_URL and RAG_TEST_API_KEY to enable live tests")

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the BAB of a barbarian 3/fighter 4 character?",
                }
            ],
            "stream": False,
        },
        timeout=300,
    )
    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert "total base attack bonus is +7" in answer.casefold(), answer
    assert "barbarian 3: +3; fighter 4: +4" in answer.casefold(), answer
