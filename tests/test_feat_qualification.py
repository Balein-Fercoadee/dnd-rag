from types import SimpleNamespace
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app as rag_app
from rag import rag_retrieval, rag_rules


def feat_point(point_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id,
        payload={
            "kind": "page_text",
            "source": "Player's Handbook.pdf",
            "page": 96,
            "text": text,
        },
        score=1.0,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    rag_app.app.dependency_overrides[rag_app.require_api_key] = lambda: None
    with TestClient(rag_app.app) as test_client:
        yield test_client
    rag_app.app.dependency_overrides.clear()


def test_feat_qualification_retrieval_searches_prerequisites_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = feat_point("initial", "Weapon Specialization")
    general_prerequisites = feat_point(
        "general",
        "Feats have prerequisites which a character must meet.",
    )
    prerequisite = feat_point(
        "prerequisite",
        "Weapon Specialization. Prerequisites: Fighter level 4th, Weapon Focus.",
    )
    embedded_queries: list[str] = []
    query_results = iter(([initial], [general_prerequisites, prerequisite, initial]))

    monkeypatch.setattr(
        rag_retrieval,
        "embed",
        lambda query: embedded_queries.append(query) or [len(embedded_queries)],
    )
    monkeypatch.setattr(
        rag_retrieval.qdrant,
        "query_points",
        lambda **_kwargs: SimpleNamespace(points=next(query_results)),
    )

    passages = rag_retrieval.retrieve(
        "Does a 5th level wizard qualify to take Weapon Specialization at 6th level?"
    )

    assert len(embedded_queries) == 2
    assert embedded_queries[1] == "Weapon Specialization feat prerequisites D&D 3.5"
    assert [passage["text"] for passage in passages] == [
        "Weapon Specialization. Prerequisites: Fighter level 4th, Weapon Focus.",
        "Weapon Specialization",
        "Feats have prerequisites which a character must meet.",
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Does a 5th level wizard qualify to take Weapon Specialization at 6th level?",
            "Weapon Specialization",
        ),
        (
            "Can a character select Improved Two-Weapon Fighting as a feat?",
            "Improved Two-Weapon Fighting",
        ),
        ("Can I take the Dodge feat?", "Dodge"),
    ],
)
def test_feat_name_is_extracted_from_selection_questions(
    question: str, expected: str
) -> None:
    assert rag_rules.feat_name_from_question(question) == expected


def test_feat_qualification_prompt_compares_prerequisites_with_character(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prerequisite_text = (
        "DODGE [GENERAL] You are trained at dodging attacks. "
        "Prerequisite: Dex 13. Benefit: You gain a bonus to Armor Class."
    )
    monkeypatch.setattr(
        rag_app,
        "retrieve",
        lambda _question: [
            {
                "source": "Player's Handbook.pdf",
                "page": 96,
                "text": prerequisite_text,
            }
        ],
    )
    system_prompts: list[str] = []

    def fake_chat(*_args: object, **kwargs: object) -> SimpleNamespace:
        request_data = kwargs["json"]
        system_prompts.append(request_data["messages"][0]["content"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {
                    "content": (
                        "No. Weapon Specialization requires Fighter level 4 and "
                        "Weapon Focus, which a 5th-level wizard does not establish."
                    )
                }
            },
        )

    monkeypatch.setattr(rag_app.requests, "post", fake_chat)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Does a 5th level wizard qualify to take Dodge at 6th level?"
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert "compare each one with the character details" in system_prompts[0]
    assert "requires levels in that specific class" in system_prompts[0]
    assert (
        "Caster level is distinct from character level and class level"
        in system_prompts[0]
    )
    assert "do not assume unmentioned feats" in system_prompts[0].casefold()


def test_class_level_prerequisite_is_deterministically_checked(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feat_passage = {
        "source": "Player's Handbook.pdf",
        "page": 103,
        "text": (
            "WEAPON SPECIALIZATION [GENERAL] Choose a weapon. "
            "Prerequisites: Proficiency with selected weapon, Weapon Focus "
            "with selected weapon, fighter level 4th. "
            "Benefit: You deal extra damage."
        ),
    }
    monkeypatch.setattr(rag_app, "retrieve", lambda _question: [feat_passage])
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "The explicit class-level prerequisite should be evaluated deterministically"
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
                        "Does a 5th level wizard qualify to take Weapon "
                        "Specialization at 6th level?"
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert answer.startswith("No.")
    assert "proficiency with the selected weapon" in answer
    assert "Weapon Focus feat" in answer
    assert "Fighter level 4th" in answer
    assert "at most 1 Fighter level" in answer
    assert "3 short of the required Fighter level 4" in answer
    assert "Player's Handbook.pdf, PDF p. 103" in answer


def test_class_level_prerequisite_evaluator_is_not_feat_specific() -> None:
    answer = rag_rules.feat_class_level_prerequisite_answer(
        "Can a 5th level wizard take Arcane Mastery at 6th level?",
        [
            {
                "source": "Rules.pdf",
                "page": 42,
                "text": (
                    "ARCANE MASTERY [GENERAL] Benefit description. "
                    "Prerequisites: Knowledge (arcana) 4 ranks, wizard level 8th. "
                    "Benefit: Benefit description."
                ),
            }
        ],
    )

    assert answer is not None
    assert answer.startswith("No. Arcane Mastery requires")
    assert "Wizard level 8th" in answer


def test_caster_level_is_not_inferred_from_fighter_or_character_level(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forge_ring = {
        "source": "Player's Handbook.pdf",
        "page": 95,
        "text": (
            "FORGE RING [ITEM CREATION] You can create magic rings. "
            "Prerequisite: Caster level 12th. "
            "Benefit: You can create magic rings."
        ),
    }
    monkeypatch.setattr(rag_app, "retrieve", lambda _question: [forge_ring])
    monkeypatch.setattr(
        rag_app.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "Caster level must not be inferred from character or fighter level"
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
                        "Can a 17th level fighter take Forge Ring at 19th level?"
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert answer.startswith("Cannot establish")
    assert "requires caster level 12th" in answer
    assert "class level and character level do not establish a caster level" in answer
    assert "Player's Handbook.pdf, PDF p. 95" in answer
