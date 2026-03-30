# AGENTS.md — `pokemon-tcg-ai-deck-builder`

This document describes the architecture, goals, and module structure of this repository for any agent (human or AI) contributing to it.

---

## What This Project Is

`pokemon-tcg-ai-deck-builder` is a research project exploring the application of **composite world models** to the Pokémon Trading Card Game. The goal is not to build a simple card recommender — it is to build a system that genuinely *understands* the game: the rules, the metagame, the deck construction problem, and the decision-making during a match.

The project is structured as a progressive set of components that can be developed independently and composed together. The current focus is on **deck construction intelligence**, specifically the problem of finding which cards belong together in a deck given a seed card, using co-occurrence mining and graph-based representations.

---

## Core Hypothesis

A large language model alone cannot play or build Pokémon TCG decks optimally — it lacks the ability to simulate consequences, maintain game state, or search combinatorial spaces. However, a **composite system** of specialized ML models, each designed for its sub-problem, can.

The architecture is inspired by:
- **MuZero** (DeepMind) — planning in latent space without a hand-coded rules model
- **AlphaZero** — self-play reinforcement learning for game mastery
- **Next Best Action systems** — recommendation under constraints, adapted to sequential deck construction
- **Graph Neural Networks** — representing decks and card relationships as heterogeneous graphs

---

## System Components

The system is divided into two layers: an **online layer** (used during a live match) and an **offline layer** (used for deck analysis and construction).

### Offline Layer

#### 1. Card Encoder
Transforms each card's structured data (name, HP, attacks, abilities, type, retreat cost) into a dense embedding vector. Trained with masked prediction and contrastive learning on card text.

- Input: card JSON from the official card database
- Output: `h_card ∈ ℝ^256`
- Key property: the same encoder is reused across all downstream components; retraining it on new expansions propagates improvements everywhere

#### 2. Co-occurrence Miner
Builds a PMI-normalized co-occurrence matrix from a corpus of top-cut tournament decklists (source: Limitless TCG). Given a seed card, produces a ranked list of cards by affinity — not by textual similarity, but by actual co-presence in competitive decks.

- Input: corpus of decklists as `list[list[str]]`, seed card name
- Output: PMI matrix, per-card co-occurrence scores relative to the seed
- Key distinction: PMI corrects for ubiquitous cards (e.g. basic Energy) that would otherwise dominate raw co-occurrence counts

#### 3. Deck Graph Builder
Represents a deck as a heterogeneous graph where nodes are cards and edges encode three types of relationships:
- **Structural synergy**: high PMI co-occurrence in historical top-cut data (static prior)
- **Functional synergy**: cosine similarity in card embedding space, filtered by role compatibility (semi-static)
- **Causal dependency**: conditional probability of playing card B given card A is in hand (dynamic, derived from game logs)

Architecture: Relational Graph Convolutional Network (R-GCN) or Heterogeneous Graph Transformer (HGT).

#### 4. Next Best Card
Given a partial deck (N < 60 cards) and a meta context vector, scores each candidate card from the legal card pool on its marginal value to the deck. Analogous to Next Best Action systems in recommendation.

- Query tower: deck graph → `z_deck ∈ ℝ^512` via GNN readout
- Item tower: card embedding `h_j ∈ ℝ^256`
- Scoring: `score(c_j | D, M) = MLP([z_deck ; h_j ; z_meta ; Δ_consistency])`
- Training: masked card prediction (self-supervised) + contrastive ranking on fitness delta pairs

#### 5. Deck Builder (Evolutionary Optimizer)
Constructs complete 60-card decklists optimized for the current meta. Not a neural network — an evolutionary search guided by the Value Net as a fitness oracle.

- Population: seeded from recent top-cut decklists + mutations
- Fitness: `w1 × E[win_rate] + w2 × consistency + w3 × matchup_coverage - w4 × constraint_violations`
- Mutation operators: semantic swap (same-role card substitution via card embeddings), structural mutation (category rebalancing), exploratory mutation (out-of-role insertion)
- Key component: the Synergy Graph constrains mutations to respect known co-occurrence patterns

#### 6. Meta Analyst
Tracks the current competitive metagame. Embeds decklists as aggregated card vectors, clusters them into archetypes, and estimates matchup percentages.

- Input: tournament results with placements from Limitless TCG
- Output: archetype tier list, matchup matrix, meta distribution vector `z_meta`
- Used by: Deck Builder (fitness), Next Best Card (meta context), Game Agent (opponent modeling)

### Online Layer (Game Agent)

These components operate during a live match and are not the current development focus, but are documented here for architectural completeness.

#### 7. Board State Encoder
Encodes the full game state (active Pokémon, bench, hand, discard, prize count, turn number) as a latent vector `z_t`. Uses a GNN over the board graph, where nodes are cards in play and edges encode game relationships (attached energy, evolution chains, targeting).

#### 8. Belief Model
Maintains a probabilistic distribution over the opponent's hidden hand. Uses particle filtering updated by observed actions, compressed via a VAE into `(μ, σ)` appended to the board state. This is the component that separates a strong player from a naive one.

#### 9. Transition Model
Predicts the next latent state given the current state and an action: `z_{t+1} = f(z_t, a)`. Enables the MCTS planner to search in latent space without invoking the rules engine at every node.

#### 10. Policy and Value Networks
- Policy net: `P(action | z_t)` — distribution over legal actions
- Value net: `V(z_t)` — estimated win probability from state `z_t`
- Trained jointly via supervised pre-training on top-player games, then improved through self-play RL (AlphaZero-style)

#### 11. MCTS Planner
Combines Policy, Value, and Transition Model to plan ahead in latent space. The rules engine is only invoked for `legal_actions()`, not for simulation. This mirrors the MuZero approach.

---

## Current Development Focus

The active work is on the **Offline Layer**, specifically:

1. **Co-occurrence mining pipeline** — building the PMI matrix from a decklist corpus and validating that it surfaces genuine card affinities, not just popularity
2. **Cluster visualization** — reducing card co-occurrence vectors via UMAP and clustering with HDBSCAN to identify functional card groups relative to a seed card
3. **Next Best Card prototype** — a two-tower scoring model for sequential deck slot recommendation

The game agent (online layer) is a future milestone, dependent on having a strong board state representation first.

---

## Data Sources

| Source | Usage | Notes |
|--------|-------|-------|
| Limitless TCG | Top-cut decklists, tournament results | Primary corpus for co-occurrence and meta analysis |
| Pokémon TCG API / Pokellector | Card data (text, stats, images) | Input to Card Encoder |
| PTCGL / TCGOne replay logs | Game state transitions | Required for Transition Model and Belief Model training; currently unavailable at scale |

---

## Repository Structure

```
pokemon-tcg-ai-deck-builder/
│
├── data/
│   ├── corpus/          # Decklist corpus (JSON)
│   ├── cards/           # Card database (JSON per set)
│   └── meta/            # Tournament results and tier snapshots
│
├── pipeline/
│   ├── cooccurrence.py  # PMI matrix construction
│   ├── embeddings.py    # Card embedding extraction
│   ├── reduction.py     # UMAP dimensionality reduction
│   └── clustering.py    # HDBSCAN cluster assignment
│
├── models/
│   ├── card_encoder/    # Transformer-based card encoder
│   ├── deck_gnn/        # R-GCN / HGT deck graph model
│   ├── next_best_card/  # Two-tower scoring model
│   ├── meta_analyst/    # Archetype clustering and matchup regression
│   └── game_agent/      # Board encoder, belief model, MCTS (future)
│
├── optimizer/
│   └── deck_builder.py  # Evolutionary search with fitness oracle
│
├── viz/
│   └── cluster_viz.jsx  # Interactive React cluster visualizer
│
├── scripts/
│   ├── scrape_limitless.py
│   └── build_card_db.py
│
├── notebooks/           # Exploratory analysis and validation
├── tests/
├── AGENTS.md            # This file
└── README.md
```

---

## Design Principles

**Separation of concerns.** Each component solves one sub-problem using the ML paradigm most suited to it. The Card Encoder uses contrastive learning. The Deck Builder uses evolutionary search. The Game Agent uses RL. No single model is asked to do everything.

**Replaceable data ingestion.** The corpus loading interface (`get_corpus() -> list[list[str]]`) is a single function. Swapping from mock data to live Limitless scraping requires changing only this function.

**Offline-first.** The Deck Builder and Meta Analyst provide standalone value without requiring the Game Agent. The project is useful before the hardest components are built.

**Explainability is structural.** Attention weights in the GNN and PMI scores in the co-occurrence matrix are directly interpretable. Explanations for card recommendations emerge from the architecture, not from post-hoc prompting of a language model.

**LLMs as interface, not engine.** A language model (e.g. Claude) may be used as a natural language interface to explain recommendations, narrate deck archetypes, or generate human-readable summaries of the optimizer's output. It does not make card selection decisions.

---

## Agent Instructions

If you are an AI agent working in this repository:

- **Do not attempt to use a language model as the primary card recommendation engine.** Route all card scoring through the `next_best_card` model or the evolutionary optimizer.
- **The rules engine is authoritative.** For any question about legal moves, card interactions, or deck validity, consult the rules engine, not the ML models.
- **PMI, not similarity.** When the task involves finding cards that belong together, use co-occurrence PMI scores. Card embedding similarity is for finding *alternatives*, not *companions*.
- **Mock data is in `data/corpus/mock/`.** Use it for unit tests and pipeline development. Do not commit real scraping output to the repository without stripping personally identifiable tournament player data.
- **Component interfaces are stable contracts.** If you modify the output shape of any model, update all downstream consumers and the relevant section of this document.
