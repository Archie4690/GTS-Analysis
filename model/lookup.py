"""Reference data: the Showdown Pokédex, the nature table, roles, and the GTS floor rules."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from config import GTS_CONFIG, POKEDEX

STATS = ("hp", "atk", "def", "spa", "spd", "spe")

# nature -> (raised stat, lowered stat); None for the five neutral natures.
NATURES: dict[str, tuple[str, str] | None] = {
    "hardy": None, "docile": None, "serious": None, "bashful": None, "quirky": None,
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa"),
}

LEGENDARY_TAGS = {"Sub-Legendary", "Restricted Legendary", "Mythical"}
FORM_ALIASES = {"galarian": "galar", "alolan": "alola", "hisuian": "hisui", "paldean": "paldea", "female": "f"}

# Breeding Pokémon: every IV matters and a nature is as good as its best use.
BREEDING_SPECIES = {"ditto"}


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# --------------------------------------------------------------------------- roles

@dataclass(frozen=True)
class Role:
    """How a (fully evolved) Pokémon uses its stats, as weights from 0 to 1."""
    name: str                 # dex id of the final form this role was worked out from
    attack: str               # physical / special / mixed
    speed: str                # fast / medium / slow
    wall: bool
    weights: dict = field(hash=False)   # stat -> weight, including hp (used for IVs/EVs)
    atk_spa_ratio: float = 1.0
    base_spe: float = 0.0
    bulk_ratio: float = 1.0

    def nature_fit(self, nature: str | None) -> float:
        if nature is None or nature not in NATURES:
            return float("nan")
        effect = NATURES[nature]
        if effect is None:
            return 0.0
        up, down = effect
        return self.weights[up] - self.weights[down]


def role_from_stats(name: str, bs: dict) -> Role:
    atk, spa, spe = bs["atk"], bs["spa"], bs["spe"]
    attack = "physical" if atk >= 1.2 * spa else "special" if spa >= 1.2 * atk else "mixed"
    speed = "fast" if spe >= 80 else "slow" if spe <= 50 else "medium"
    best_attack = max(atk, spa)
    bulk = (bs["hp"] + bs["def"] + bs["spd"]) / 3
    wall = bulk >= 1.1 * best_attack
    main, other = ("atk", "spa") if atk >= spa else ("spa", "atk")
    w: dict[str, float] = {}
    if wall:
        # Walls mostly attack with one stat; if it isn't clear which, neither matters much.
        if attack == "mixed":
            w[main] = w[other] = 0.1
        else:
            w[main], w[other] = 0.4, 0.0
        w["def"] = w["spd"] = 1.0
        w["spe"] = {"fast": 0.6, "medium": 0.3, "slow": 0.0}[speed]
        w["hp"] = 1.0
    else:
        w[main] = 1.0
        w[other] = 0.5 if attack == "mixed" else 0.0
        w["def"] = w["spd"] = 0.3
        w["spe"] = {"fast": 1.0, "medium": 0.6, "slow": 0.0}[speed]
        w["hp"] = 0.7
    return Role(name, attack, speed, wall, w,
                atk_spa_ratio=atk / max(spa, 1), base_spe=spe, bulk_ratio=bulk / max(best_attack, 1))


# Stand-in stat lines for a breeding Pokémon: its nature is as good as the best of these.
_ARCHETYPES = {
    "fast physical": dict(hp=80, atk=130, def_=80, spa=60, spd=80, spe=110),
    "fast special": dict(hp=80, atk=60, def_=80, spa=130, spd=80, spe=110),
    "slow physical": dict(hp=100, atk=130, def_=90, spa=60, spd=80, spe=40),
    "slow special": dict(hp=100, atk=60, def_=90, spa=130, spd=80, spe=40),
    "physical wall": dict(hp=100, atk=80, def_=130, spa=50, spd=100, spe=60),
    "special wall": dict(hp=100, atk=50, def_=130, spa=80, spd=100, spe=60),
}


def _breeding_roles() -> list[Role]:
    roles = []
    for name, s in _ARCHETYPES.items():
        bs = {"hp": s["hp"], "atk": s["atk"], "def": s["def_"], "spa": s["spa"], "spd": s["spd"], "spe": s["spe"]}
        roles.append(role_from_stats(name, bs))
    return roles


# Breeders want all six IVs, whatever the eventual use.
BREEDING_IV_WEIGHTS = {s: 1.0 for s in STATS}


# --------------------------------------------------------------------------- pokédex

class Dex:
    def __init__(self, path=POKEDEX):
        if not path.exists():
            raise SystemExit(f"{path} is missing: run `python model/fetch_lookup.py` first")
        raw = json.loads(path.read_text())
        # Positive dex numbers only (drops CAP / fan-made entries).
        self.e = {k: v for k, v in raw.items() if v.get("num", 0) > 0 and "baseStats" in v}

    def resolve(self, species: str, form: str | None) -> tuple[str | None, str | None]:
        """Cobblemon species + form -> (Showdown id or None, regional bias for evolutions)."""
        sp = to_id(species)
        form = (form or "normal").lower()
        bias = None
        if form.endswith("-bias"):                    # e.g. goomy "hisui-bias" evolves into Hisuian Goodra
            region = form[: -len("-bias")]
            bias = FORM_ALIASES.get(region, region)
            form = "normal"
        if form not in ("normal", ""):
            cand = sp + to_id(FORM_ALIASES.get(form, form))
            entry = self.e.get(cand)
            if entry and to_id(entry.get("baseSpecies", "")) == sp:
                return cand, bias
        if sp in self.e:
            return sp, bias
        return None, bias

    def tags(self, dex_id: str) -> set[str]:
        entry = self.e[dex_id]
        tags = set(entry.get("tags", []))
        base = to_id(entry.get("baseSpecies", ""))
        if base and base in self.e:
            tags |= set(self.e[base].get("tags", []))
        return tags

    def _evo_ids(self, dex_id: str, bias: str | None) -> list[str]:
        out = []
        for name in self.e[dex_id].get("evos", []):
            eid = to_id(name)
            if bias and eid + bias in self.e:
                eid = eid + bias
            if eid in self.e and eid not in out:
                out.append(eid)
        return out

    def final_forms(self, dex_id: str, bias: str | None = None) -> list[str]:
        evos = self._evo_ids(dex_id, bias)
        if not evos:
            return [dex_id]
        leaves: list[str] = []
        for eid in evos:
            for leaf in self.final_forms(eid, bias):
                if leaf not in leaves:
                    leaves.append(leaf)
        return leaves

    def line_root(self, dex_id: str) -> str:
        seen = set()
        cur = dex_id
        while cur not in seen:
            seen.add(cur)
            entry = self.e[cur]
            prevo = to_id(entry.get("prevo", ""))
            if prevo and prevo in self.e:
                cur = prevo
                continue
            base = to_id(entry.get("baseSpecies", ""))   # regional/mega forms -> base species line
            if base and base != cur and base in self.e:
                cur = base
                continue
            break
        return cur

    def base_stats(self, dex_id: str) -> dict:
        return self.e[dex_id]["baseStats"]


@dataclass
class SpeciesInfo:
    dex_id: str | None
    found: bool
    base_stats: dict | None
    types: tuple[str, str]
    egg_group: str
    legendary: bool
    mythical: bool
    restricted: bool
    ultra_beast: bool
    paradox: bool
    fully_evolved: bool
    line: str
    roles: list[Role]
    breeding: bool


@lru_cache(maxsize=None)
def species_info(species: str, form: str) -> SpeciesInfo:
    dex = get_dex()
    dex_id, bias = dex.resolve(species, form)
    breeding = to_id(species) in BREEDING_SPECIES
    if dex_id is None:
        return SpeciesInfo(None, False, None, ("unknown", "none"), "unknown", False, False, False, False,
                           False, False, to_id(species), [], breeding)
    entry = dex.e[dex_id]
    tags = dex.tags(dex_id)
    types = tuple(entry.get("types", ["unknown"])) + ("none",)
    finals = dex.final_forms(dex_id, bias)
    roles = _breeding_roles() if breeding else [role_from_stats(f, dex.base_stats(f)) for f in finals]
    return SpeciesInfo(
        dex_id=dex_id, found=True, base_stats=dex.base_stats(dex_id),
        types=(types[0], types[1]),
        egg_group=(entry.get("eggGroups") or ["unknown"])[0],
        legendary=bool(tags & LEGENDARY_TAGS), mythical="Mythical" in tags,
        restricted="Restricted Legendary" in tags, ultra_beast="Ultra Beast" in tags,
        paradox="Paradox" in tags, fully_evolved=not dex._evo_ids(dex_id, bias),
        line=dex.line_root(dex_id), roles=roles, breeding=breeding,
    )


_DEX: Dex | None = None


def get_dex() -> Dex:
    global _DEX
    if _DEX is None:
        _DEX = Dex()
    return _DEX


# --------------------------------------------------------------------------- GTS rules

class GtsRules:
    """The GTS mod config: minimum-price rules, tax and listing timer."""

    def __init__(self, path=GTS_CONFIG):
        cfg = json.loads(path.read_text())
        self.base_price = float(cfg.get("basePrice", 0))
        self.tax = float(cfg.get("tax", 0))
        self.timer_minutes = cfg.get("timer", {})
        self.rules = sorted(cfg.get("globalModifiers", []), key=lambda r: -r["priority"])
        for r in self.rules:
            if r["type"] != "ADD":
                raise SystemExit(f"Unsupported GTS modifier type {r['type']!r}")
        if cfg.get("speciesModifiers"):
            raise SystemExit("speciesModifiers are set in the GTS config; floor_price() doesn't handle them yet")

    @lru_cache(maxsize=None)
    def floor(self, perfect_ivs: int, shiny: bool, hidden_ability: bool, legendary: bool, ultra_beast: bool) -> float:
        """The highest-priority rule the Pokémon qualifies for sets the minimum price."""
        for r in self.rules:
            s = r["spec"]
            if perfect_ivs < s["numberOfPerfectIvs"]:
                continue
            if s["isShinyRequired"] and not shiny:
                continue
            if s["isHiddenAbilityRequired"] and not hidden_ability:
                continue
            if s["isLegendaryRequired"] and not legendary:
                continue
            if s["isUltraBeastRequired"] and not ultra_beast:
                continue
            return self.base_price + float(r["value"])
        return self.base_price


_RULES: GtsRules | None = None


def get_rules() -> GtsRules:
    global _RULES
    if _RULES is None:
        _RULES = GtsRules()
    return _RULES
