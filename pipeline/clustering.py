"""
Cluster cards into core packages using UMAP + HDBSCAN on PMI vectors.

Each card's PMI row vector captures its co-occurrence profile across the corpus.
UMAP reduces these to 2D for visualization, HDBSCAN finds dense clusters
that represent functional card packages (e.g. "Charizard engine", "Lost Zone core").
"""

import numpy as np
import pandas as pd
import umap
import hdbscan

from pipeline.cooccurrence import CooccurrenceResult


def cluster_cards(
    cooc: CooccurrenceResult,
    min_cluster_size: int = 3,
    umap_neighbors: int = 5,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster cards and return a DataFrame with columns:
      card, cluster, umap_x, umap_y, deck_appearances, win_rate_proxy
    """
    pmi = cooc.pmi_matrix.copy()
    # Replace -inf with a large negative for UMAP
    pmi[pmi == -np.inf] = -10.0

    n_cards = len(cooc.cards)
    # Adjust UMAP params for small datasets
    effective_neighbors = min(umap_neighbors, n_cards - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, effective_neighbors),
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
        init="random",
    )
    embedding = reducer.fit_transform(pmi)

    # HDBSCAN clustering — scale min_cluster_size with corpus
    effective_min_cluster = max(min_cluster_size, min(8, n_cards // 10))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding)

    # Deck appearance count per card (how many decks it shows up in)
    appearances = np.diag(cooc.raw_counts).copy()
    # raw_counts diagonal is 0 (self-pairs), use freq from column sums instead
    appearances = (cooc.raw_counts > 0).sum(axis=1)

    df = pd.DataFrame({
        "card": cooc.cards,
        "cluster": labels,
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1],
        "deck_appearances": appearances,
    })

    return df
