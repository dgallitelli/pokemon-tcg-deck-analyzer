"""
Next Best Card — given a partial deck, score every candidate card.

Scoring: for each candidate card c, compute the average PMI between c
and every card already in the partial deck. This captures how well c
fits with the existing card package.

score(c | deck) = mean(PMI(c, d) for d in deck if PMI(c,d) > -inf)

Cards with no positive co-occurrence with any deck card score -inf.

Evolution lines: when collapse_evolutions=True, pre-evolution cards
(e.g. Charcadet for Armarouge) are hidden from recommendations since
they're implied by the highest evolution.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from pipeline.cooccurrence import CooccurrenceResult

EVO_MAP_PATH = Path(__file__).parent.parent / "data" / "evolution_map.json"

_evo_data = None

def _load_evo_map() -> dict:
    global _evo_data
    if _evo_data is None:
        if EVO_MAP_PATH.exists():
            with open(EVO_MAP_PATH) as f:
                _evo_data = json.load(f)
        else:
            _evo_data = {"evolves_from": {}, "implies": {}}
    return _evo_data


@dataclass
class Recommendation:
    card: str
    score: float
    co_occurring_with: int  # how many deck cards it co-occurs with


def score_candidates(
    cooc: CooccurrenceResult,
    partial_deck: list[str],
    top_k: int = 15,
    exclude: set[str] | None = None,
    collapse_evolutions: bool = True,
) -> list[Recommendation]:
    """
    Score all cards not in partial_deck by average PMI with deck cards.
    When collapse_evolutions=True, pre-evolutions implied by a recommended
    card (or already in the deck) are excluded from results.
    Returns top_k recommendations sorted by score descending.
    """
    exclude = exclude or set()
    deck_set = set(partial_deck)

    # Build set of cards to skip: pre-evolutions of cards already in deck
    evo_map = _load_evo_map() if collapse_evolutions else {"implies": {}, "evolves_from": {}}
    evolves_from = evo_map.get("evolves_from", {})

    # Cards in deck that are pre-evolutions of other deck cards — already implied
    implied_by_deck = set()
    for card in partial_deck:
        for other in evo_map.get("implies", {}).get(card, []):
            if other in deck_set:
                implied_by_deck.add(other)

    # Get indices of deck cards that exist in our corpus
    deck_idxs = []
    for card in partial_deck:
        if card in cooc.card_to_idx:
            deck_idxs.append(cooc.card_to_idx[card])

    if not deck_idxs:
        return []

    pmi = cooc.pmi_matrix
    results = []

    for i, card_name in enumerate(cooc.cards):
        if card_name in deck_set or card_name in exclude:
            continue

        # Skip if this card is a pre-evolution of something already in deck
        # A card is a pre-evo if a deck card's evolution chain passes through it
        if collapse_evolutions:
            is_pre_evo = False
            for deck_card in deck_set:
                # Walk the deck card's evolution chain downward
                ancestor = evolves_from.get(deck_card)
                while ancestor:
                    if ancestor == card_name:
                        is_pre_evo = True
                        break
                    ancestor = evolves_from.get(ancestor)
                if is_pre_evo:
                    break
            if is_pre_evo:
                continue

        # PMI scores between this candidate and each deck card
        scores = pmi[i, deck_idxs]
        valid = scores[scores > -np.inf]

        if len(valid) == 0:
            continue

        avg_score = float(np.mean(valid))
        results.append(Recommendation(
            card=card_name,
            score=avg_score,
            co_occurring_with=len(valid),
        ))

    results.sort(key=lambda r: r.score, reverse=True)

    if not collapse_evolutions:
        return results[:top_k]

    # Post-filter: if we recommend card X, remove its pre-evolutions from results
    filtered = []
    implied_cards = set()
    for r in results:
        if r.card in implied_cards:
            continue
        filtered.append(r)
        # Mark this card's pre-evolutions as implied
        for implied in evo_map.get("implies", {}).get(r.card, []):
            if implied in evolves_from or implied != r.card:
                # Only mark actual pre-evolutions, not the whole family
                implied_cards.add(implied)
        if len(filtered) >= top_k:
            break

    return filtered


def build_deck_iteratively(
    cooc: CooccurrenceResult,
    seed_cards: list[str],
    target_size: int = 20,
    top_k_per_step: int = 1,
) -> list[str]:
    """
    Starting from seed cards, greedily add the best card one at a time.
    Returns the full deck (seed + added cards).
    """
    deck = list(seed_cards)
    while len(deck) < target_size:
        recs = score_candidates(cooc, deck, top_k=top_k_per_step)
        if not recs:
            break
        deck.append(recs[0].card)
    return deck
