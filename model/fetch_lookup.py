"""Download Pokémon Showdown's pokedex.json into lookup/ (run once).

The file is only ever parsed as JSON, never executed.
"""
import json
import urllib.request

from config import POKEDEX, POKEDEX_URL


def main() -> None:
    req = urllib.request.Request(POKEDEX_URL, headers={"User-Agent": "gts-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    dex = json.loads(raw)  # fail loudly if it isn't JSON
    if not isinstance(dex, dict) or "garchomp" not in dex:
        raise SystemExit("Downloaded file doesn't look like a Showdown pokedex")
    POKEDEX.parent.mkdir(parents=True, exist_ok=True)
    POKEDEX.write_text(json.dumps(dex, indent=1, sort_keys=True))
    print(f"Saved {len(dex)} entries to {POKEDEX}")


if __name__ == "__main__":
    main()
