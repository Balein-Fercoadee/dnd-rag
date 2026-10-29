import os
from collections.abc import Iterator

import pytest
import requests
from fastapi.testclient import TestClient

import app as rag_app

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"


def progression_row(
    class_name: str,
    level: int,
    bab: int,
    page: int,
    fortitude: int,
    reflex: int,
    will: int,
) -> dict[str, object]:
    return {
        "kind": "class_progression_row",
        "class_name": class_name,
        "class_level": level,
        "base_attack_bonus": bab,
        "fortitude_save": fortitude,
        "reflex_save": reflex,
        "will_save": will,
        "source": SOURCE,
        "page": page,
        "text": (
            f"{class_name} class level {level}; base attack bonus: +{bab}; "
            f"base saves: Fortitude +{fortitude}, Reflex +{reflex}, Will +{will}."
        ),
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    with TestClient(rag_app.app) as test_client:
        yield test_client
    rag_app.app.dependency_overrides.clear()


def test_multiclass_saving_throws_add_each_class_contribution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    barbarian = progression_row("Barbarian", 3, 3, 26, 3, 1, 1)
    fighter = progression_row("Fighter", 4, 4, 40, 4, 1, 1)
    monkeypatch.setattr(
        rag_app,
        "retrieve",
        lambda _question: [barbarian, fighter],
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What are the base saving throws for a 3rd level "
                        "barbarian/4th level fighter character?"
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert "Fortitude +7" in answer
    assert "Reflex +2" in answer
    assert "Will +2" in answer


@pytest.mark.live
def test_live_api_multiclass_base_saving_throws() -> None:
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
                    "content": (
                        "What are the base saving throws for a 3rd level "
                        "barbarian/4th level fighter character?"
                    ),
                }
            ],
            "stream": False,
        },
        timeout=300,
    )
    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert "fortitude +7" in answer.casefold(), answer
    assert "reflex +2" in answer.casefold(), answer
    assert "will +2" in answer.casefold(), answer
