#!/usr/bin/env python3
"""Build a local SQLite FTS5 index for Figment protocol cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARD_DIR = PROJECT_ROOT / "data" / "protocol_cards"
DEFAULT_INDEX_PATH = DEFAULT_CARD_DIR / "protocol_cards.sqlite"


def _card_text(card: Any) -> str:
    if isinstance(card, str):
        return card
    if isinstance(card, list):
        return " ".join(_card_text(item) for item in card)
    if isinstance(card, dict):
        return " ".join(_card_text(value) for value in card.values())
    return ""


def _load_cards(card_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    cards: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(card_dir.glob("*.json")):
        cards.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return cards


def build_index(card_dir: Path = DEFAULT_CARD_DIR, index_path: Path = DEFAULT_INDEX_PATH) -> int:
    cards = _load_cards(card_dir)
    if not cards:
        raise ValueError(f"no protocol cards found in {card_dir}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    connection = sqlite3.connect(tmp_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("CREATE TABLE cards (card_id TEXT PRIMARY KEY, path TEXT NOT NULL, title TEXT NOT NULL, payload TEXT NOT NULL)")
        connection.execute("CREATE VIRTUAL TABLE cards_fts USING fts5(card_id UNINDEXED, title, body)")
        for path, card in cards:
            card_id = str(card.get("card_id", "")).strip()
            title = str(card.get("title", "")).strip()
            if not card_id or not title:
                raise ValueError(f"{path} must contain card_id and title")
            payload = json.dumps(card, sort_keys=True)
            body = _card_text(card)
            connection.execute(
                "INSERT INTO cards (card_id, path, title, payload) VALUES (?, ?, ?, ?)",
                (card_id, str(path), title, payload),
            )
            connection.execute(
                "INSERT INTO cards_fts (card_id, title, body) VALUES (?, ?, ?)",
                (card_id, title, body),
            )
        connection.commit()
    finally:
        connection.close()

    tmp_path.replace(index_path)
    return len(cards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figment protocol-card SQLite FTS index.")
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARD_DIR, help="Directory containing protocol card JSON files.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help="SQLite index path to write.")
    args = parser.parse_args()

    count = build_index(args.cards, args.index)
    print(f"indexed {count} protocol cards into {args.index}")


if __name__ == "__main__":
    main()
