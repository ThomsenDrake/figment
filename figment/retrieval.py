"""Protocol-card retrieval with SQLite FTS5 and JSON fallback."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARD_DIR = PROJECT_ROOT / "data" / "protocol_cards"
DEFAULT_INDEX_PATH = DEFAULT_CARD_DIR / "protocol_cards.sqlite"


def load_protocol_cards(card_dir: str | Path = DEFAULT_CARD_DIR) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path in sorted(Path(card_dir).glob("*.json")):
        cards.append(json.loads(path.read_text(encoding="utf-8")))
    return cards


def known_card_ids(card_dir: str | Path = DEFAULT_CARD_DIR) -> set[str]:
    return {str(card.get("card_id")) for card in load_protocol_cards(card_dir) if card.get("card_id")}


def query_from_intake(intake: Mapping[str, Any]) -> str:
    fields = ("setting", "patient_age", "pregnancy_status", "chief_concern", "symptoms", "vitals", "available_supplies", "responder_note")
    return " ".join(str(intake.get(field, "")) for field in fields if intake.get(field))


def _tokens(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", query) if len(token) > 1]


def _card_text(card: Any) -> str:
    if isinstance(card, str):
        return card
    if isinstance(card, list):
        return " ".join(_card_text(item) for item in card)
    if isinstance(card, dict):
        return " ".join(_card_text(value) for value in card.values())
    return ""


def _result(card: dict[str, Any], score: float, source: str) -> dict[str, Any]:
    return {
        "card_id": card.get("card_id", ""),
        "title": card.get("title", ""),
        "score": score,
        "source": source,
        "card": card,
    }


def _memory_search(query: str, card_dir: str | Path, limit: int) -> list[dict[str, Any]]:
    terms = _tokens(query)
    cards = load_protocol_cards(card_dir)
    if not terms:
        return [_result(card, 0.0, "json_fallback") for card in cards[:limit]]

    scored: list[tuple[float, dict[str, Any]]] = []
    for card in cards:
        title = str(card.get("title", "")).lower()
        body = _card_text(card).lower()
        score = 0.0
        for term in terms:
            score += 4.0 * title.count(term)
            score += body.count(term)
        if score > 0:
            scored.append((score, card))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("card_id", ""))))
    return [_result(card, score, "json_fallback") for score, card in scored[:limit]]


def _fts_search(query: str, index_path: str | Path, limit: int) -> list[dict[str, Any]]:
    terms = _tokens(query)
    if not terms:
        return []
    fts_query = " OR ".join(terms)
    connection = sqlite3.connect(Path(index_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT cards.payload, bm25(cards_fts) AS rank
            FROM cards_fts
            JOIN cards ON cards_fts.card_id = cards.card_id
            WHERE cards_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    finally:
        connection.close()
    return [_result(json.loads(row["payload"]), float(-row["rank"]), "sqlite_fts") for row in rows]


def search_protocol_cards(
    query: str,
    *,
    card_dir: str | Path = DEFAULT_CARD_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Search protocol cards, preferring local SQLite FTS/BM25 when present."""
    capped_limit = max(1, min(limit, 10))
    if Path(index_path).exists():
        try:
            results = _fts_search(query, index_path, capped_limit)
        except sqlite3.Error:
            results = []
        if results:
            return results
    return _memory_search(query, card_dir, capped_limit)
