"""Runtime configuration and parsing patterns for the D&D rules API."""
import os
import re

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")


CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")


EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


COLLECTION = os.getenv("QDRANT_COLLECTION", "dnd35")


API_KEY = os.getenv("RAG_API_KEY")


RETRIEVAL_LIMIT = int(os.getenv("RETRIEVAL_LIMIT", "8"))


BONUS_ROW_LIMIT = int(os.getenv("BONUS_ROW_LIMIT", "2"))


ABILITY_SCORE_PATTERN = re.compile(
    r"\b(?:(?P<score_before>\d{1,2})\s+)?"
    r"(?P<ability>intelligence|wisdom|charisma)"
    r"(?:\s+score)?(?:\s+(?:is|of))?"
    r"(?:\s+(?P<score_after>\d{1,2}))?\b",
    re.IGNORECASE,
)


SPELL_LEVEL_PATTERN = re.compile(
    r"\b(?:(\d{1,2})(?:st|nd|rd|th)?[\s-]+level(?:\s+of)?\s+spells?"
    r"|spells?\s+(?:of\s+)?(\d{1,2})(?:st|nd|rd|th)?[\s-]+level)\b",
    re.IGNORECASE,
)


SPELL_SLOT_COUNT_PATTERN = re.compile(
    r"\b(?P<label>\d+(?:st|nd|rd|th)-level):\s*"
    r"(?P<count>\d+|unavailable)\b",
    re.IGNORECASE,
)


ABILITY_SCORE_RANGE_PATTERN = re.compile(
    r"\bAbility score (?P<low>\d{1,2})-(?P<high>\d{1,2})\b",
    re.IGNORECASE,
)


CLASS_LEVEL_ROW_PATTERN = re.compile(
    r"\bClass level (?P<level>\d{1,2})(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)


CLASS_TABLE_NAME_PATTERN = re.compile(
    r"\bTable\s+\d+\s*[–-]\s*\d+\s*:\s*(?:The\s+)?(?P<name>[^.\n]+)",
    re.IGNORECASE,
)


BASE_ATTACK_BONUS_QUESTION = re.compile(r"\b(?:BAB|base attack bonus)\b", re.IGNORECASE)


BASE_ATTACK_BONUS_VALUE_PATTERN = re.compile(
    r"\bbase attack bonus:\s*\+(?P<value>\d+)\b",
    re.IGNORECASE,
)


BASE_SAVE_QUESTION = re.compile(
    r"\b(?:base\s+)?(?:saving throws?|saves?)\b",
    re.IGNORECASE,
)


BASE_SAVE_VALUE_PATTERN = re.compile(
    r"\bbase saves:\s*Fortitude\s+\+(?P<fortitude>\d+),\s*"
    r"Reflex\s+\+(?P<reflex>\d+),\s*Will\s+\+(?P<will>\d+)",
    re.IGNORECASE,
)
