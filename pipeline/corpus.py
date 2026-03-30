"""
Corpus loader — single entry point for decklist data.

Interface contract: get_corpus() -> list[list[str]]
Each inner list is a deck represented as card names (with repeats for counts).
get_corpus_with_meta() returns richer deck objects for the visualizer.
"""

import json
from pathlib import Path
from typing import TypedDict

MOCK_PATH = Path(__file__).parent.parent / "data" / "corpus" / "mock" / "sample_decklists.json"
LIMITLESS_PATH = Path(__file__).parent.parent / "data" / "corpus" / "limitless_decklists.json"


def _default_path() -> Path:
    """Use Limitless data if available, otherwise fall back to mock."""
    if LIMITLESS_PATH.exists():
        return LIMITLESS_PATH
    return MOCK_PATH


class DeckMeta(TypedDict):
    id: str
    name: str
    format: str
    placement: int
    tournament: str
    cards: list[str]  # flattened with repeats


def _load_raw(path: Path | None = None) -> list[dict]:
    path = path or _default_path()
    with open(path) as f:
        return json.load(f)


def _flatten_cards(deck: dict) -> list[str]:
    """Expand card entries into a flat list respecting counts."""
    return [card["name"] for card in deck["cards"] for _ in range(card["count"])]


def get_corpus(path: Path | None = None) -> list[list[str]]:
    """Return corpus as list of decklists, each a list of card names (with repeats)."""
    return [_flatten_cards(d) for d in _load_raw(path)]


def get_corpus_unique(path: Path | None = None) -> list[set[str]]:
    """Return corpus as list of decklists, each a set of unique card names."""
    return [set(_flatten_cards(d)) for d in _load_raw(path)]


def get_corpus_with_meta(path: Path | None = None) -> list[DeckMeta]:
    """Return deck metadata alongside flattened card lists."""
    raw = _load_raw(path)
    out = []
    for d in raw:
        out.append(DeckMeta(
            id=d.get("id", d.get("tournament_id", "")),
            name=d.get("name", d.get("deck_archetype", "")),
            format=d.get("format", "standard"),
            placement=d.get("placement", 0),
            tournament=d.get("tournament", d.get("tournament_name", "")),
            cards=_flatten_cards(d),
        ))
    return out
