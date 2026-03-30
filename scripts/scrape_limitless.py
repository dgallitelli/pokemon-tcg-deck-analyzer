"""
Scrape decklists and matchup data from Limitless TCG API.

Fetches recent PTCG Standard tournaments and extracts:
  - Top-cut decklists with placement, win/loss record, and deck archetype
  - Round-by-round pairings cross-referenced with deck archetypes
  - Archetype-vs-archetype matchup win rates

Usage:
    python scripts/scrape_limitless.py [--tournaments 100] [--min-players 40] [--top-n 32]

Output:
    data/corpus/limitless_decklists.json   — decklists
    data/corpus/limitless_matchups.json    — archetype matchup matrix
    data/corpus/limitless_pairings.json    — raw pairings with deck info
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

API_BASE = "https://play.limitlesstcg.com/api"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "corpus"


def api_get(path: str, retries: int = 3) -> list | dict:
    url = f"{API_BASE}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "pokemon-tcg-deck-builder/0.1")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited on {path}, waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    return []
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return []
    return []


def flatten_decklist(decklist: dict) -> list[dict]:
    """Flatten pokemon/trainer/energy sections into a single card list, preserving type."""
    cards = []
    for section in ["pokemon", "trainer", "energy"]:
        for card in decklist.get(section, []):
            cards.append({
                "name": card["name"],
                "count": card["count"],
                "set": card.get("set", ""),
                "number": card.get("number", ""),
                "type": section,
            })
    return cards


def fetch_tournaments(n_tournaments: int, min_players: int) -> list[dict]:
    """Fetch tournament list, paginating if needed."""
    all_tournaments = []
    page = 1
    while len(all_tournaments) < n_tournaments * 2:
        batch = api_get(f"/tournaments?game=PTCG&format=STANDARD&limit=50&page={page}")
        if not batch:
            break
        all_tournaments.extend(batch)
        page += 1
        time.sleep(0.2)

    filtered = [t for t in all_tournaments if t.get("players", 0) >= min_players]
    return filtered[:n_tournaments]


def scrape_tournament(t: dict, top_n: int):
    """Scrape a single tournament: standings + pairings."""
    tid = t["id"]

    standings = api_get(f"/tournaments/{tid}/standings")
    if not standings:
        return [], [], {}

    # Build player -> deck archetype map
    player_deck = {}
    for p in standings:
        player_id = p.get("player", "")
        deck = p.get("deck", {})
        if player_id and deck:
            player_deck[player_id] = deck.get("name", "Unknown")

    # Extract top-N decklists
    decklists = []
    for player in standings:
        placing = player.get("placing")
        if placing is None or placing > top_n:
            continue
        decklist = player.get("decklist")
        if not decklist:
            continue

        record = player.get("record", {})
        deck_type = player.get("deck", {})
        decklists.append({
            "tournament_id": tid,
            "tournament_name": t.get("name", ""),
            "tournament_date": t.get("date", ""),
            "tournament_players": t.get("players", 0),
            "player": player.get("name", "Anonymous"),
            "placing": placing,
            "wins": record.get("wins", 0),
            "losses": record.get("losses", 0),
            "ties": record.get("ties", 0),
            "deck_archetype": deck_type.get("name", "Unknown"),
            "cards": flatten_decklist(decklist),
        })

    # Get pairings
    time.sleep(0.15)
    pairings_raw = api_get(f"/tournaments/{tid}/pairings")
    pairings = []
    for m in pairings_raw:
        p1 = m.get("player1", "")
        p2 = m.get("player2", "")
        winner = m.get("winner")
        if not p1 or not p2:
            continue

        d1 = player_deck.get(p1, "Unknown")
        d2 = player_deck.get(p2, "Unknown")
        if d1 == "Unknown" or d2 == "Unknown":
            continue

        if winner == p1:
            result = "p1"
        elif winner == p2:
            result = "p2"
        elif winner == 0:
            result = "tie"
        else:
            result = "other"

        pairings.append({
            "tournament_id": tid,
            "round": m.get("round", 0),
            "phase": m.get("phase", 0),
            "deck1": d1,
            "deck2": d2,
            "result": result,
        })

    return decklists, pairings, player_deck


def build_matchup_matrix(all_pairings: list[dict]) -> dict:
    """Build archetype-vs-archetype matchup stats from pairings."""
    matchups = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))
    archetype_stats = defaultdict(lambda: {"games": 0, "wins": 0})

    for p in all_pairings:
        d1, d2, result = p["deck1"], p["deck2"], p["result"]

        if result == "p1":
            matchups[d1][d2]["wins"] += 1
            matchups[d2][d1]["losses"] += 1
            archetype_stats[d1]["wins"] += 1
        elif result == "p2":
            matchups[d1][d2]["losses"] += 1
            matchups[d2][d1]["wins"] += 1
            archetype_stats[d2]["wins"] += 1
        elif result == "tie":
            matchups[d1][d2]["ties"] += 1
            matchups[d2][d1]["ties"] += 1

        archetype_stats[d1]["games"] += 1
        archetype_stats[d2]["games"] += 1

    matrix = {}
    for deck_a, opponents in matchups.items():
        matrix[deck_a] = {}
        for deck_b, stats in opponents.items():
            total = stats["wins"] + stats["losses"] + stats["ties"]
            matrix[deck_a][deck_b] = {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total": total,
                "win_rate": round(stats["wins"] / total * 100, 1) if total > 0 else 0,
            }

    overall = {}
    for arch, stats in archetype_stats.items():
        overall[arch] = {
            "games": stats["games"],
            "wins": stats["wins"],
            "win_rate": round(stats["wins"] / stats["games"] * 100, 1) if stats["games"] > 0 else 0,
        }

    return {"matchups": matrix, "archetypes": overall}


def main():
    parser = argparse.ArgumentParser(description="Scrape Limitless TCG decklists + matchups")
    parser.add_argument("--tournaments", type=int, default=100,
                        help="Number of tournaments to fetch")
    parser.add_argument("--min-players", type=int, default=40,
                        help="Minimum tournament size")
    parser.add_argument("--top-n", type=int, default=32,
                        help="Only keep top N decklists per tournament")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for tournament scraping")
    args = parser.parse_args()

    # Load existing data for incremental scraping
    existing_tids = set()
    existing_decks = []
    existing_pairings = []
    decks_path = OUTPUT_DIR / "limitless_decklists.json"
    pairings_path = OUTPUT_DIR / "limitless_pairings.json"
    if decks_path.exists():
        existing_decks = json.loads(decks_path.read_text())
        existing_tids = set(d["tournament_id"] for d in existing_decks)
        print(f"Loaded {len(existing_decks)} existing decklists from {len(existing_tids)} tournaments")
    if pairings_path.exists():
        existing_pairings = json.loads(pairings_path.read_text())

    print(f"Fetching tournament list...")
    tournaments = fetch_tournaments(args.tournaments, args.min_players)
    new_tournaments = [t for t in tournaments if t["id"] not in existing_tids]
    print(f"Found {len(tournaments)} tournaments, {len(new_tournaments)} new")

    if not new_tournaments:
        print("No new tournaments to scrape.")
        return

    all_decks = list(existing_decks)
    all_pairings = list(existing_pairings)
    completed = 0

    def process_tournament(t):
        return t, scrape_tournament(t, args.top_n)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_tournament, t): t for t in new_tournaments}
        for future in as_completed(futures):
            t, (decklists, pairings, _) = future.result()
            all_decks.extend(decklists)
            all_pairings.extend(pairings)
            completed += 1
            tname = t["name"][:50]
            players = t.get("players", 0)
            print(f"  [{completed}/{len(new_tournaments)}] {tname} ({players}p) — {len(decklists)} decks, {len(pairings)} matches")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "limitless_decklists.json", "w") as f:
        json.dump(all_decks, f, indent=2)

    with open(OUTPUT_DIR / "limitless_pairings.json", "w") as f:
        json.dump(all_pairings, f, indent=2)

    matchup_data = build_matchup_matrix(all_pairings)
    with open(OUTPUT_DIR / "limitless_matchups.json", "w") as f:
        json.dump(matchup_data, f, indent=2)

    archetypes = {}
    for d in all_decks:
        arch = d["deck_archetype"]
        archetypes[arch] = archetypes.get(arch, 0) + 1

    print(f"\n=== Summary ===")
    print(f"Decklists: {len(all_decks)}")
    print(f"Pairings:  {len(all_pairings)}")
    print(f"Matchups:  {len(matchup_data['archetypes'])} archetypes")
    print(f"\nTop archetypes (by decklist count):")
    for arch, count in sorted(archetypes.items(), key=lambda x: -x[1])[:15]:
        stats = matchup_data["archetypes"].get(arch, {})
        wr = stats.get("win_rate", 0)
        games = stats.get("games", 0)
        print(f"  {arch}: {count} lists, {games} games, {wr}% WR")


if __name__ == "__main__":
    main()
