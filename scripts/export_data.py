"""
Export pipeline data to JSON for the web frontend.

Four view modes, each providing different X/Y coordinates:
  1. PMI — synergy (top-5 mean PMI) vs specificity (top-5 PMI concentration)
  2. CCoE — conditional co-occurrence excess (P(j|i) - P(j))
  3. Archetype — archetype count vs win rate
  4. PCA — first two principal components of PMI vectors
"""

import math
import json
import sys
import os
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cooccurrence import build_cooccurrence
from pipeline.clustering import cluster_cards
from pipeline.next_best_card import score_candidates

# Load raw decklist data for win/loss stats
CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "corpus", "limitless_decklists.json",
)

with open(CORPUS_PATH) as f:
    raw_decks = json.load(f)

n_decks = len(raw_decks)

# Load set code -> API set ID mapping for card images
SET_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "set_code_mapping.json",
)
with open(SET_MAP_PATH) as f:
    set_code_map = json.load(f)

# Per-card stats: appearances, total wins, total games, archetypes
card_wins = defaultdict(float)
card_games = defaultdict(float)
card_appearances = defaultdict(int)
card_archetypes = defaultdict(set)  # card -> set of archetype names

# Track set/number for image URLs (keep first occurrence per card name)
card_image_info: dict[str, dict] = {}

for deck in raw_decks:
    wins = deck.get("wins", 0)
    losses = deck.get("losses", 0)
    total = wins + losses
    archetype = deck.get("deck_archetype", "Unknown")
    card_names = set(c["name"] for c in deck["cards"])
    for card in card_names:
        if total > 0:
            card_wins[card] += wins
            card_games[card] += total
        card_appearances[card] += 1
        if archetype != "Unknown":
            card_archetypes[card].add(archetype)
    # Collect set/number for image URLs
    for c in deck["cards"]:
        name = c["name"]
        if name not in card_image_info and c.get("set") and c.get("number"):
            api_set_id = set_code_map.get(c["set"])
            if api_set_id:
                card_image_info[name] = {
                    "img": f"https://images.pokemontcg.io/{api_set_id}/{c['number']}.png",
                    "img_lg": f"https://images.pokemontcg.io/{api_set_id}/{c['number']}_hires.png",
                }

# Build co-occurrence and clusters
cooc = build_cooccurrence()
df = cluster_cards(cooc)

# Per-card base stats
card_meta_share = {}
card_win_rate = {}

for card in cooc.cards:
    appearances = card_appearances.get(card, 0)
    card_meta_share[card] = appearances / n_decks if n_decks > 0 else 0
    games = card_games.get(card, 0)
    card_win_rate[card] = card_wins.get(card, 0) / games if games > 0 else 0.5


# === Normalization helper ===

def normalize_to_range(values_dict):
    """Normalize to [-1, 1] centered on median."""
    vals = sorted(values_dict.values())
    median = vals[len(vals) // 2]
    above = max(v - median for v in vals) or 1.0
    below = max(median - v for v in vals) or 1.0
    return {k: round((v - median) / (above if v >= median else below), 4)
            for k, v in values_dict.items()}


# === VIEW 1: PMI (synergy vs specificity) ===

TOP_K = 5

pmi_syn_raw = {}
pmi_spec_raw = {}

for i, card in enumerate(cooc.cards):
    row = cooc.pmi_matrix[i]
    positive = row[(row > -np.inf) & (row > 0)]
    positive = np.sort(positive)[::-1]

    if len(positive) == 0:
        pmi_syn_raw[card] = 0.0
        pmi_spec_raw[card] = 0.0
        continue

    top = positive[:TOP_K]
    pmi_syn_raw[card] = float(np.mean(top))
    total_mass = float(positive.sum())
    pmi_spec_raw[card] = float(top.sum()) / total_mass if total_mass > 0 else 0.0

pmi_syn = normalize_to_range(pmi_syn_raw)
pmi_spec = normalize_to_range(pmi_spec_raw)


# === VIEW 2: CCoE (conditional co-occurrence excess) ===

ccoe_syn_raw = {}
ccoe_spec_raw = {}

# P(j) = base rate for each card
p_base = {card: card_appearances.get(card, 0) / n_decks for card in cooc.cards}

for i, card in enumerate(cooc.cards):
    app_i = card_appearances.get(card, 0)
    if app_i == 0:
        ccoe_syn_raw[card] = 0.0
        ccoe_spec_raw[card] = 0.0
        continue

    # P(j|i) - P(j) for all j
    excesses = []
    for j, other in enumerate(cooc.cards):
        if i == j:
            continue
        co_count = cooc.raw_counts[i, j]
        p_j_given_i = co_count / app_i
        excess = p_j_given_i - p_base[other]
        if excess > 0:
            excesses.append(excess)

    excesses.sort(reverse=True)

    if len(excesses) == 0:
        ccoe_syn_raw[card] = 0.0
        ccoe_spec_raw[card] = 0.0
        continue

    top = excesses[:TOP_K]
    ccoe_syn_raw[card] = float(np.mean(top))
    total = sum(excesses)
    ccoe_spec_raw[card] = sum(top) / total if total > 0 else 0.0

ccoe_syn = normalize_to_range(ccoe_syn_raw)
ccoe_spec = normalize_to_range(ccoe_spec_raw)


# === VIEW 3: Archetype count vs Win rate ===

arch_count_raw = {}
arch_wr_raw = {}

for card in cooc.cards:
    arch_count_raw[card] = float(len(card_archetypes.get(card, set())))
    arch_wr_raw[card] = card_win_rate.get(card, 0.5) * 100  # percentage

arch_count = normalize_to_range(arch_count_raw)
arch_wr = normalize_to_range(arch_wr_raw)


# === VIEW 4: PCA of PMI matrix ===

pmi_clean = cooc.pmi_matrix.copy()
pmi_clean[pmi_clean == -np.inf] = -10.0

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(pmi_clean)

pca_x_raw = {card: float(pca_coords[i, 0]) for i, card in enumerate(cooc.cards)}
pca_y_raw = {card: float(pca_coords[i, 1]) for i, card in enumerate(cooc.cards)}

pca_x = normalize_to_range(pca_x_raw)
pca_y = normalize_to_range(pca_y_raw)

print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")


# === Build Pokemon name set for cluster labeling ===
# Prefer Pokemon names over Trainer/Energy for package labels

import re

EVO_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "evolution_map.json",
)
_pokemon_names = set()
if os.path.exists(EVO_MAP_PATH):
    with open(EVO_MAP_PATH) as f:
        evo_data = json.load(f)
    _pokemon_names.update(evo_data.get("evolves_from", {}).keys())
    _pokemon_names.update(evo_data.get("evolves_from", {}).values())
    for card, others in evo_data.get("implies", {}).items():
        _pokemon_names.add(card)
        _pokemon_names.update(others)

# Heuristic: names ending with ex/V/VMAX/VSTAR/GX are always Pokemon
_POKEMON_SUFFIX = re.compile(r'\s+(ex|V|VMAX|VSTAR|GX|EX)$')

def is_pokemon(name):
    if name in _pokemon_names:
        return True
    if _POKEMON_SUFFIX.search(name):
        return True
    return False


# === Build cluster metadata ===

clusters = {}
for cid in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == cid]
    cards_in_cluster = subset["card"].tolist()

    avg_share = sum(card_meta_share.get(c, 0) for c in cards_in_cluster) / len(cards_in_cluster)
    avg_wr = sum(card_win_rate.get(c, 0) for c in cards_in_cluster) / len(cards_in_cluster)

    cards_in_cluster.sort(key=lambda c: card_meta_share.get(c, 0), reverse=True)

    recs = score_candidates(cooc, cards_in_cluster, top_k=8)

    # Pick a Pokemon name for the label if one exists in the cluster
    pokemon_in_cluster = [c for c in cards_in_cluster if is_pokemon(c)]
    top_card = pokemon_in_cluster[0] if pokemon_in_cluster else cards_in_cluster[0]
    label = "Unclustered" if cid == -1 else f"{top_card} package"

    clusters[int(cid)] = {
        "label": label,
        "cards": cards_in_cluster,
        "avg_meta_share": round(avg_share * 100, 1),
        "avg_win_rate": round(avg_wr * 100, 1),
        "recommendations": [
            {"card": r.card, "score": round(r.score, 3), "cooccurs": r.co_occurring_with}
            for r in recs
        ],
    }


# === Build per-card points with all four views ===

points = []
for _, row in df.iterrows():
    card = row["card"]
    ms = card_meta_share.get(card, 0)
    wr = card_win_rate.get(card, 0.5)
    points.append({
        "card": card,
        "cluster": int(row["cluster"]),
        "views": {
            "pmi": {"x": pmi_syn.get(card, 0), "y": pmi_spec.get(card, 0)},
            "ccoe": {"x": ccoe_syn.get(card, 0), "y": ccoe_spec.get(card, 0)},
            "archetype": {"x": arch_count.get(card, 0), "y": arch_wr.get(card, 0)},
            "pca": {"x": pca_x.get(card, 0), "y": pca_y.get(card, 0)},
        },
        "meta_share": round(ms * 100, 2),
        "win_rate": round(wr * 100, 2),
        "appearances": card_appearances.get(card, 0),
        "archetype_count": len(card_archetypes.get(card, set())),
        "total_games": int(card_games.get(card, 0)),
        "img": card_image_info.get(card, {}).get("img", ""),
        "img_lg": card_image_info.get(card, {}).get("img_lg", ""),
    })


# === PMI pair lookup (top-15 per card) ===

TOP_PMI_PER_CARD = 15
_card_top = defaultdict(list)
for i, card_a in enumerate(cooc.cards):
    for j, card_b in enumerate(cooc.cards):
        if i < j and cooc.pmi_matrix[i, j] > float("-inf"):
            score = round(float(cooc.pmi_matrix[i, j]), 3)
            _card_top[card_a].append((score, card_b))
            _card_top[card_b].append((score, card_a))

_kept_pairs = set()
for card, partners in _card_top.items():
    partners.sort(reverse=True)
    for score, partner in partners[:TOP_PMI_PER_CARD]:
        key = f"{min(card, partner)}||{max(card, partner)}"
        _kept_pairs.add((key, score))

pmi_pairs = {k: s for k, s in _kept_pairs}


# === Write output ===

payload = {
    "points": points,
    "clusters": clusters,
    "pmi_pairs": pmi_pairs,
    "total_decks": n_decks,
    "total_cards": len(cooc.cards),
    "views": {
        "pmi": {
            "label": "PMI Synergy",
            "x_label": "SYNERGY (top-5 mean PMI)",
            "y_label": "SPECIFICITY (PMI concentration)",
            "quadrants": {
                "tr": "Engine pieces", "tl": "Niche techs",
                "br": "Flexible glue", "bl": "Generic staples",
            },
        },
        "ccoe": {
            "label": "Co-occurrence Excess",
            "x_label": "SYNERGY (conditional excess)",
            "y_label": "SPECIFICITY (excess concentration)",
            "quadrants": {
                "tr": "Engine pieces", "tl": "Niche techs",
                "br": "Flexible glue", "bl": "Generic staples",
            },
        },
        "archetype": {
            "label": "Archetype vs WR",
            "x_label": "ARCHETYPE COUNT",
            "y_label": "WIN RATE",
            "quadrants": {
                "tr": "Versatile winners", "tl": "Specialist winners",
                "br": "Versatile losers", "bl": "Specialist losers",
            },
        },
        "pca": {
            "label": "PCA",
            "x_label": f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var) — staple vs archetype",
            "y_label": f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var) — deck family axis",
            "quadrants": {
                "tr": "Core staples", "tl": "Archetype family A",
                "br": "Broad tech cards", "bl": "Archetype family B",
            },
        },
    },
}

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.json")
with open(out_path, "w") as f:
    json.dump(payload, f)

print(f"Exported {len(points)} cards, {len(clusters)} clusters, 4 views")
print(f"Written to: {out_path}")
