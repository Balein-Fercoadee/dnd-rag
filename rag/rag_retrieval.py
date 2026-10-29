"""Public retrieval facade preserving the original RAG module interface."""

from typing import Any

from .retrieval_backend import embed, qdrant
from .retrieval_pipeline import retrieve as _retrieve
from .retrieval_tables import (
    class_name_from_point,
    retrieve_bonus_spell_rows,
    retrieve_class_progression_rows,
    retrieve_class_spell_row,
)

__all__ = [
    "class_name_from_point",
    "embed",
    "qdrant",
    "retrieve",
    "retrieve_bonus_spell_rows",
    "retrieve_class_progression_rows",
    "retrieve_class_spell_row",
]


def retrieve(question: str) -> list[dict[str, Any]]:
    return _retrieve(question, client=qdrant, embed_fn=embed)
