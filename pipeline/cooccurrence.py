"""
Co-occurrence matrix with PMI normalization.

Given a corpus of decklists (each a set of card names), builds:
  - Raw co-occurrence counts
  - PMI-normalized affinity scores

PMI(a, b) = log2(P(a,b) / (P(a) * P(b)))
where P(a,b) = fraction of decks containing both a and b,
      P(a)   = fraction of decks containing a.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from pipeline.corpus import get_corpus_unique


@dataclass
class CooccurrenceResult:
    """Holds the PMI matrix and card-to-index mapping."""
    cards: list[str]
    card_to_idx: dict[str, int]
    raw_counts: np.ndarray    # (n_cards, n_cards) symmetric int matrix
    pmi_matrix: np.ndarray    # (n_cards, n_cards) symmetric float matrix
    deck_count: int


def build_cooccurrence(corpus: list[set[str]] | None = None) -> CooccurrenceResult:
    """Build PMI co-occurrence matrix from a decklist corpus."""
    if corpus is None:
        corpus = get_corpus_unique()

    # Vocabulary and frequency
    card_freq: Counter[str] = Counter()
    for deck in corpus:
        for card in deck:
            card_freq[card] += 1

    cards = sorted(card_freq.keys())
    card_to_idx = {c: i for i, c in enumerate(cards)}
    n = len(cards)
    n_decks = len(corpus)

    # Raw co-occurrence counts
    raw = np.zeros((n, n), dtype=np.int32)
    for deck in corpus:
        idxs = [card_to_idx[c] for c in deck]
        for i, a in enumerate(idxs):
            for b in idxs[i + 1:]:
                raw[a, b] += 1
                raw[b, a] += 1

    # PMI calculation
    pmi = np.full((n, n), -np.inf, dtype=np.float64)
    for i in range(n):
        p_i = card_freq[cards[i]] / n_decks
        for j in range(i + 1, n):
            p_j = card_freq[cards[j]] / n_decks
            p_ij = raw[i, j] / n_decks
            if p_ij > 0:
                score = math.log2(p_ij / (p_i * p_j))
                pmi[i, j] = score
                pmi[j, i] = score
        pmi[i, i] = 0.0  # self-PMI is meaningless, zero it out

    return CooccurrenceResult(
        cards=cards,
        card_to_idx=card_to_idx,
        raw_counts=raw,
        pmi_matrix=pmi,
        deck_count=n_decks,
    )


def query_card(result: CooccurrenceResult, seed: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Return top-k cards by PMI affinity to a seed card."""
    if seed not in result.card_to_idx:
        raise KeyError(f"Card '{seed}' not found in corpus")
    idx = result.card_to_idx[seed]
    scores = result.pmi_matrix[idx]
    ranked = np.argsort(scores)[::-1]
    out = []
    for i in ranked:
        if result.cards[i] == seed:
            continue
        if scores[i] == -np.inf:
            break
        out.append((result.cards[i], float(scores[i])))
        if len(out) >= top_k:
            break
    return out
