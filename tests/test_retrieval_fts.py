from pathlib import Path

from figment.retrieval import search_protocol_cards
from scripts.build_fts import build_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = PROJECT_ROOT / "data" / "protocol_cards"


def test_builds_temp_fts_index_and_retrieves_relevant_cards_with_bm25(tmp_path: Path) -> None:
    index_path = tmp_path / "protocol_cards.sqlite"

    indexed_count = build_index(card_dir=str(CARD_DIR), index_path=str(index_path))

    assert indexed_count == 10
    results = search_protocol_cards(
        "child lethargy sunken eyes unable to keep fluids down no urine",
        card_dir=CARD_DIR,
        index_path=index_path,
        limit=3,
    )

    assert [result["source"] for result in results] == ["sqlite_fts"] * len(results)
    assert results[0]["card_id"] == "PED-DEHYD-RED-FLAGS-v1"
    assert [result["score"] for result in results] == sorted(
        (result["score"] for result in results),
        reverse=True,
    )


def test_search_uses_json_fallback_when_fts_index_is_absent(tmp_path: Path) -> None:
    results = search_protocol_cards(
        "spreading redness fever wound drainage",
        card_dir=CARD_DIR,
        index_path=tmp_path / "missing.sqlite",
        limit=3,
    )

    assert [result["source"] for result in results] == ["json_fallback"] * len(results)
    assert results[0]["card_id"] == "WOUND-INFECTION-ESCALATION-v1"
