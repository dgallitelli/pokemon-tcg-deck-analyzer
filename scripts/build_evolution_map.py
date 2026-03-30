"""
Build evolution line map by querying pokemontcg.io for each unique card name.

Output: data/evolution_map.json
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
CORPUS_PATH = DATA_DIR / "corpus" / "limitless_decklists.json"
OUTPUT_PATH = DATA_DIR / "evolution_map.json"
API = "https://api.pokemontcg.io/v2"


def api_get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "pokemon-tcg-deck-builder/0.1")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
            elif attempt < retries - 1:
                time.sleep(1)
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def main():
    with open(CORPUS_PATH) as f:
        decks = json.load(f)

    card_names = sorted(set(c["name"] for d in decks for c in d["cards"]))
    print(f"Querying evolution data for {len(card_names)} unique cards...")

    evolves_from = {}
    batch_size = 10

    # Query in batches using OR queries: name:"X" OR name:"Y"
    for i in range(0, len(card_names), batch_size):
        batch = card_names[i:i + batch_size]
        # Build OR query for exact name matches
        name_queries = " OR ".join(f'name:"{n}"' for n in batch)
        q = urllib.parse.quote(f"supertype:Pokémon ({name_queries})")
        url = f"{API}/cards?q={q}&select=name,evolvesFrom&pageSize=250"

        data = api_get(url)
        if data and "data" in data:
            for card in data["data"]:
                name = card.get("name", "")
                ef = card.get("evolvesFrom", "")
                if name and ef and name in set(card_names):
                    evolves_from[name] = ef

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {i + batch_size}/{len(card_names)} queried, {len(evolves_from)} evolutions found")
        time.sleep(0.3)

    print(f"Found {len(evolves_from)} evolvesFrom relationships")

    # Build family graph
    children = defaultdict(set)
    for child, parent in evolves_from.items():
        children[parent].add(child)

    corpus_set = set(card_names)

    def find_root(name):
        visited = set()
        while name in evolves_from and name not in visited:
            visited.add(name)
            name = evolves_from[name]
        return name

    # Build implies map
    seen_roots = {}
    implies = {}

    for card_name in card_names:
        root = find_root(card_name)
        if root not in seen_roots:
            members = set()
            queue = [root]
            while queue:
                current = queue.pop(0)
                if current in corpus_set:
                    members.add(current)
                for child in children.get(current, []):
                    queue.append(child)
            seen_roots[root] = members

        family = seen_roots[root]
        if len(family) > 1:
            others = sorted(m for m in family if m != card_name)
            if others:
                implies[card_name] = others

    output = {
        "evolves_from": {k: v for k, v in evolves_from.items()},
        "implies": implies,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  {len(evolves_from)} evolvesFrom entries")
    print(f"  {len(implies)} cards in evolution families")

    for name in ["Armarouge", "Gardevoir ex", "Dragapult ex", "Gholdengo ex", "Froslass"]:
        if name in implies:
            print(f"  {name} implies: {implies[name]}")
        elif name in card_names:
            print(f"  {name}: no evolution family found")


if __name__ == "__main__":
    import urllib.parse
    main()
