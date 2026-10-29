from collections.abc import Iterator

import pytest
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


BONUS_18 = {
    "kind": "bonus_spell_row",
    "source": SOURCE,
    "page": 9,
    "text": (
        "Ability score 18-19 (modifier +4); bonus spells per day: "
        "0th-level: 0; 1st-level: 1."
    ),
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


def test_attack_bonus_adds_strength_modifier_to_bab(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fighter = progression_row("Fighter", 5, 5, 40)
    melee_formula = {
        "kind": "page_text",
        "source": SOURCE,
        "page": 135,
        "text": (
            "Your attack bonus with a melee weapon is: Base attack bonus + "
            "Strength modifier + size modifier."
        ),
    }
    medium_size = {
        "kind": "table_data_row",
        "source": SOURCE,
        "page": 135,
        "text": "Table 8-1: Size Modifiers. Table row: Medium | +0.",
    }
    answer = ask(
        client,
        "What is the attack bonus of a 5th level fighter with an 18 strength?",
        [fighter, BONUS_18, melee_formula, medium_size],
        monkeypatch,
    )

    assert "attack bonus is +9" in answer
    assert "BAB +5 (Fighter 5)" in answer
    assert "Strength 18 (+4)" in answer
    assert "Medium size +0" in answer
    assert "Assuming Medium size" in answer
    assert "PDF pp. 40, 9, 135" in answer


def test_halfling_attack_bonus_is_concise_and_uses_small_size(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fighter = progression_row("Fighter", 5, 5, 40)
    halfling_size = {
        "kind": "page_text",
        "source": SOURCE,
        "page": 21,
        "text": (
            "Halflings are Small creatures. A Small character gets +1 size "
            "bonus on attack rolls."
        ),
    }
    size_table = {
        "kind": "table_data_row",
        "source": SOURCE,
        "page": 135,
        "text": "Table 8-1: Size Modifiers. Table row: Small | +1.",
    }
    melee_formula = {
        "kind": "page_text",
        "source": SOURCE,
        "page": 135,
        "text": (
            "Your attack bonus with a melee weapon is: Base attack bonus + "
            "Strength modifier + size modifier."
        ),
    }

    answer = ask(
        client,
        "What is the attack bonus of a 5th level halfling fighter with an 18 strength?",
        [fighter, BONUS_18, halfling_size, size_table, melee_formula],
        monkeypatch,
    )

    assert "attack bonus is +10" in answer
    assert "BAB +5 (Fighter 5)" in answer
    assert "Strength 18 (+4)" in answer
    assert "Small size +1" in answer
    assert "Assuming Medium size" not in answer
    assert len(answer.split()) < 40
    assert "PDF pp. 40, 9, 135, 21" in answer


def test_ranged_attack_does_not_use_strength_as_attack_modifier() -> None:
    fighter = progression_row("Fighter", 5, 5, 40)
    passages = [
        fighter,
        BONUS_18,
        {
            "kind": "page_text",
            "source": SOURCE,
            "page": 135,
            "text": (
                "With a ranged weapon, your attack bonus is: Base attack bonus + "
                "Dexterity modifier + size modifier."
            ),
        },
        {
            "kind": "table_data_row",
            "source": SOURCE,
            "page": 135,
            "text": "Table 8-1: Size Modifiers. Table row: Medium | +0.",
        },
    ]

    assert (
        rag_rules.structured_attack_bonus_answer(
            "What is the ranged attack bonus of a 5th level fighter with an 18 strength?",
            passages,
        )
        is None
    )
