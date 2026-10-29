from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app as rag_app
from rag import rag_rules

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    with TestClient(rag_app.app) as test_client:
        yield test_client
    rag_app.app.dependency_overrides.clear()


def test_base_ac_uses_ten_when_no_modifiers_are_provided(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    armor_class_formula = {
        "kind": "page_text",
        "source": SOURCE,
        "page": 7,
        "text": (
            "Armor Class (AC): Add the following numbers together to get your AC: "
            "10 + your armor bonus + your shield bonus + your size modifier + "
            "your Dexterity modifier."
        ),
    }
    monkeypatch.setattr(
        rag_app,
        "retrieve",
        lambda _question: [armor_class_formula],
    )
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "The base AC should be answered from the retrieved rule"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [
                {"role": "user", "content": "What is the AC of a 20th level wizard?"}
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert "base AC is 10" in answer
    assert "character level alone does not change it" in answer
    assert f"{SOURCE}, PDF p. 7" in answer


def test_ac_with_modifier_details_is_not_reduced_to_base_ac() -> None:
    assert (
        rag_rules.structured_base_armor_class_answer(
            "What is the AC of a wizard with 18 Dexterity?",
            [
                {
                    "source": SOURCE,
                    "page": 7,
                    "text": (
                        "Armor Class (AC): Add the following numbers together to "
                        "get your AC: 10 + your armor bonus + your shield bonus + "
                        "your size modifier + your Dexterity modifier."
                    ),
                }
            ],
        )
        is None
    )
