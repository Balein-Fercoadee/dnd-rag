import os
from collections.abc import Iterator

import pytest
import requests
from fastapi.testclient import TestClient

import app as rag_app

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"


def class_row(
    caster_class: str,
    class_level: int,
    class_ability: str,
    counts: str,
    page: int,
) -> dict[str, object]:
    return {
        "kind": "table_row",
        "class_name": caster_class,
        "class_level": class_level,
        "class_ability": class_ability,
        "source": SOURCE,
        "page": page,
        "text": (
            f"Class level {rag_app.ordinal_number(class_level)}; "
            f"spells per day: {counts}"
        ),
    }


def bonus_row(
    score_range: str,
    counts: str,
    page: int = 9,
    modifier: int = 5,
) -> dict[str, object]:
    return {
        "kind": "bonus_spell_row",
        "source": SOURCE,
        "page": page,
        "text": (
            f"Ability score {score_range} (modifier +{modifier}); "
            f"bonus spells per day: {counts}"
        ),
    }


CLERIC_5 = class_row(
    "Cleric",
    5,
    "wisdom",
    "1st-level: 4; 2nd-level: 3; 3rd-level: 2; 4th-level: 1.",
    33,
)
WIZARD_11 = class_row(
    "Wizard",
    11,
    "intelligence",
    "0th-level: 4; 1st-level: 4; 2nd-level: 4; 3rd-level: 4; "
    "4th-level: 3; 5th-level: 2; 6th-level: 1; "
    "7th-level: unavailable; 8th-level: unavailable; 9th-level: unavailable.",
    56,
)
RANGER_11 = class_row(
    "Ranger",
    11,
    "wisdom",
    "1st-level: 1; 2nd-level: 1; 3rd-level: 0; 4th-level: unavailable.",
    47,
)
BONUS_18 = bonus_row(
    "18-19",
    "0th-level: 0; 1st-level: 1; 2nd-level: 1; 3rd-level: 1; "
    "4th-level: 1; 5th-level: 0; 6th-level: 0; 7th-level: 0; "
    "8th-level: 0; 9th-level: 0.",
    modifier=4,
)
BONUS_20 = bonus_row(
    "20-21",
    "0th-level: 0; 1st-level: 1; 2nd-level: 1; 3rd-level: 1; "
    "4th-level: 1; 5th-level: 1; 6th-level: 0; 7th-level: 0; "
    "8th-level: 0; 9th-level: 0.",
)


def progression_row(
    caster_class: str, level: int, bab: int, page: int
) -> dict[str, object]:
    return {
        "kind": "class_progression_row",
        "class_name": caster_class,
        "class_level": level,
        "base_attack_bonus": bab,
        "fortitude_save": {"Barbarian": 3, "Fighter": 4}.get(caster_class, 0),
        "reflex_save": {"Barbarian": 1, "Fighter": 1}.get(caster_class, 0),
        "will_save": {"Barbarian": 1, "Fighter": 1}.get(caster_class, 0),
        "source": SOURCE,
        "page": page,
        "text": (
            f"{caster_class} class level {level}; base attack bonus: +{bab}."
        ),
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    monkeypatch.setattr(rag_app, "retrieve", lambda _question: [])
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


def test_cleric_level_five_lists_base_spells_without_assuming_bonus_score(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = ask(
        client,
        "How many spells does a 5th level cleric get?",
        [CLERIC_5],
        monkeypatch,
    )

    assert "1st-level: 4 base" in answer
    assert "2nd-level: 3 base" in answer
    assert "3rd-level: 2 base" in answer
    assert "4th-level: 1 base" in answer
    assert "Wisdom score, which was not provided" in answer


def test_wizard_level_eleven_adds_intelligence_bonus_to_fifth_level(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = ask(
        client,
        "How many 5th level spells does an 11th level wizard with a 20 intelligence get?",
        [WIZARD_11, BONUS_20],
        monkeypatch,
    )

    assert "5th-level: 2 base + 1 bonus = 3" in answer


def test_wizard_uses_intelligence_not_wisdom_for_bonus_spells(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = ask(
        client,
        "How many 5th level spells does an 11th level wizard with a 20 wisdom get?",
        [WIZARD_11, BONUS_20],
        monkeypatch,
    )

    assert "5th-level: 2 base" in answer
    assert "Wisdom does not grant bonus spells to a Wizard" in answer
    assert "5th-level: 2 base + 1 bonus" not in answer


def test_ranger_level_eleven_adds_wisdom_bonus_only_to_available_levels(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = ask(
        client,
        "How many spells does an 11th level ranger with an 18 wisdom get?",
        [RANGER_11, BONUS_18],
        monkeypatch,
    )

    assert "1st-level: 1 base + 1 bonus = 2" in answer
    assert "2nd-level: 1 base + 1 bonus = 2" in answer
    assert "3rd-level: 0 base + 1 bonus = 1" in answer
    assert "4th-level:" not in answer


def test_unlisted_multiword_caster_name_is_resolved_from_indexed_table(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    dread_necromancer = class_row(
        "Dread Necromancer",
        11,
        "charisma",
        "1st-level: 5; 2nd-level: 4; 3rd-level: 3.",
        87,
    )
    answer = ask(
        client,
        "How many 2nd-level spells does an 11th-level Dread Necromancer get?",
        [dread_necromancer],
        monkeypatch,
    )

    assert "2nd-level: 4 base" in answer
    assert "Charisma score, which was not provided" in answer


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

    assert rag_app.requested_class_levels(
        question,
        {"Barbarian", "Fighter", "Wizard"},
    ) == [("Barbarian", 3), ("Fighter", 4)]


def test_multiclass_saving_throws_add_each_class_contribution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    barbarian = progression_row("Barbarian", 3, 3, 26)
    barbarian.update(
        {"fortitude_save": 3, "reflex_save": 1, "will_save": 1}
    )
    fighter = progression_row("Fighter", 4, 4, 40)
    fighter.update(
        {"fortitude_save": 4, "reflex_save": 1, "will_save": 1}
    )
    answer = ask(
        client,
        "What are the base saving throws for a 3rd level barbarian/4th level fighter character?",
        [barbarian, fighter],
        monkeypatch,
    )

    assert "Fortitude +7" in answer
    assert "Reflex +2" in answer
    assert "Will +2" in answer


@pytest.mark.live
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "How many spells does a 5th level cleric get?",
            ("1st-level: 4 base", "2nd-level: 3 base"),
        ),
        (
            "How many 5th level spells does an 11th level wizard with a 20 intelligence get?",
            ("5th-level: 2 base + 1 bonus = 3",),
        ),
        (
            "How many 5th level spells does an 11th level wizard with a 20 wisdom get?",
            ("5th-level: 2 base", "Wisdom does not grant bonus spells"),
        ),
        (
            "How many spells does an 11th level ranger with an 18 wisdom get?",
            ("1st-level: 1 base + 1 bonus = 2", "3rd-level: 0 base + 1 bonus = 1"),
        ),
        (
            "What is the BAB of a barbarian 3/fighter 4 character?",
            ("total base attack bonus is +7", "Barbarian 3: +3; Fighter 4: +4"),
        ),
        (
            "What are the base saving throws for a 3rd level barbarian/4th level fighter character?",
            ("Fortitude +7", "Reflex +2", "Will +2"),
        ),
    ],
)
def test_live_api_spell_question(
    question: str, expected: tuple[str, ...]
) -> None:
    base_url = os.getenv("RAG_TEST_BASE_URL")
    api_key = os.getenv("RAG_TEST_API_KEY")
    if not base_url or not api_key:
        pytest.skip("Set RAG_TEST_BASE_URL and RAG_TEST_API_KEY to enable live tests")

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        },
        timeout=300,
    )
    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    for expected_phrase in expected:
        assert expected_phrase.casefold() in answer.casefold(), answer
