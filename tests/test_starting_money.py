from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as rag_app
from rag import rag_retrieval, rag_rules

SOURCE = "Player's Handbook I. Core Rulebook I v3.5.pdf"

STARTING_GOLD_TABLE = {
    "kind": "table",
    "source": SOURCE,
    "page": 112,
    "text": (
        "Table 7-1: Random Starting Gold\n"
        "Class Amount (average)\n"
        "Barbarian 4d4 × 10 (100 gp)\n"
        "Bard 4d4 × 10 (100 gp)\n"
        "Cleric 5d4 × 10 (125 gp)\n"
        "Druid 2d4 × 10 (50 gp)\n"
        "Fighter 6d4 × 10 (150 gp)\n"
        "Monk 5d4 (12 gp, 5 sp)\n"
        "Paladin 6d4 × 10 (150 gp)\n"
        "Ranger 6d4 × 10 (150 gp)\n"
        "Rogue 5d4 (125 gp)\n"
        "Sorcerer 3d4 × 10 (75 gp)\n"
        "Wizard 3d4 × 10 (75 gp)"
    ),
}

CLASS_STARTING_MONEY = [
    ("Barbarian", "4d4 × 10", "40–160", "100"),
    ("Bard", "4d4 × 10", "40–160", "100"),
    ("Cleric", "5d4 × 10", "50–200", "125"),
    ("Druid", "2d4 × 10", "20–80", "50"),
    ("Fighter", "6d4 × 10", "60–240", "150"),
    ("Monk", "5d4", "5–20", "12.5"),
    ("Paladin", "6d4 × 10", "60–240", "150"),
    ("Ranger", "6d4 × 10", "60–240", "150"),
    ("Rogue", "5d4 × 10", "50–200", "125"),
    ("Sorcerer", "3d4 × 10", "30–120", "75"),
    ("Wizard", "3d4 × 10", "30–120", "75"),
]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    with TestClient(rag_app.app) as test_client:
        yield test_client
    rag_app.app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("class_name", "dice", "gold_range", "average"),
    CLASS_STARTING_MONEY,
)
def test_starting_money_returns_dice_range_and_average_for_every_core_class(
    class_name: str,
    dice: str,
    gold_range: str,
    average: str,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_app,
        "retrieve",
        lambda _question: [STARTING_GOLD_TABLE],
    )
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "Starting money should be calculated from the retrieved table"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": f"What is the starting money for a {class_name.lower()}?",
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert f"{class_name} starting money is {dice} gp" in answer
    assert f"{gold_range} gp" in answer
    assert f"average {average} gp" in answer
    assert f"{SOURCE}, PDF p. 112" in answer


def test_starting_money_does_not_guess_when_the_class_row_is_missing() -> None:
    answer = rag_rules.structured_starting_money_answer(
        "What is the starting money for a fighter?",
        [
            {
                "kind": "table",
                "source": SOURCE,
                "page": 112,
                "text": (
                    "Table 7-1: Random Starting Gold\n" "Barbarian 4d4 × 10 (100 gp)"
                ),
            }
        ],
    )

    assert answer is not None
    assert "Cannot establish" in answer
    assert "matching starting-gold table row was not retrieved" in answer


def test_starting_money_retrieval_targets_the_random_starting_gold_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starting_gold_point = SimpleNamespace(
        id="starting-gold",
        payload=STARTING_GOLD_TABLE,
    )
    query_calls = 0

    def fake_query_points(**_kwargs: object) -> SimpleNamespace:
        nonlocal query_calls
        query_calls += 1
        return SimpleNamespace(points=[] if query_calls == 1 else [starting_gold_point])

    monkeypatch.setattr(rag_retrieval, "embed", lambda _text: [0.0])
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "query_points",
        fake_query_points,
    )

    passages = rag_retrieval.retrieve("What is the starting money for a fighter?")

    assert query_calls == 2
    assert len(passages) == 1
    assert passages[0]["text"] == STARTING_GOLD_TABLE["text"]
