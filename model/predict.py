"""Price a Pokémon for the GTS and estimate its chance of selling.

    python model/predict.py --species gible --ivs 31,31,31,0,31,31 --nature jolly
    python model/predict.py --species rayquaza --ivs 31,31,31,20,31,31 --nature adamant --level 100 --price 250000

IVs and EVs are in the order HP, Atk, Def, SpA, SpD, Spe.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from config import ARTIFACT_DIR
from features import FEATURE_GROUPS, FeatureContext, build_features, floors_for
from lookup import NATURES, species_info, to_id
from modelling import predict_price, sell_features, widen

MARKET_COLS = FEATURE_GROUPS["market"] + FEATURE_GROUPS["market_long"]

IV_ORDER = ["iv_hp", "iv_attack", "iv_defence", "iv_special_attack", "iv_special_defence", "iv_speed"]
EV_ORDER = [c.replace("iv_", "ev_") for c in IV_ORDER]
NICE = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5]


def nice_prices(lo: float, hi: float) -> list[float]:
    out = []
    for e in range(2, 10):
        for m in NICE:
            v = m * 10 ** e
            if lo <= v <= hi:
                out.append(v)
    return out


def money(x: float) -> str:
    return f"{x:,.0f}"


class Predictor:
    def __init__(self):
        meta_path = ARTIFACT_DIR / "meta.json"
        if not meta_path.exists():
            raise SystemExit("No trained model found: run `python model/train.py` first")
        self.meta = json.loads(meta_path.read_text())
        self.ctx = FeatureContext.from_dict(self.meta["context"])
        self.price_models = []
        for f in self.meta["price_models"]:
            m = CatBoostRegressor()
            m.load_model(str(ARTIFACT_DIR / f))
            self.price_models.append(m)
        self.sell_model = CatBoostClassifier()
        self.sell_model.load_model(str(ARTIFACT_DIR / "sell.cbm"))
        self.target = self.meta["config"]["target"]

    def make_row(self, species, ivs, nature, level=1, evs=None, form="normal", shiny=False, ability=None,
                 hidden_ability=False, gender=None, aspects="", held_item="", ball=None, breedable=None,
                 when=None) -> pd.DataFrame:
        species = to_id(species)
        d = self.meta["species_defaults"].get(species, self.meta["global_defaults"])
        row = {
            "event_time": when or pd.Timestamp.now(tz="UTC"), "species": species, "form": form,
            "shiny": shiny, "level": level, "nature": nature.lower(), "ability": ability or d["ability"],
            "hidden_ability": hidden_ability, "gender": gender or d["gender"],
            "breedable": d["breedable"] if breedable is None else breedable, "held_item": held_item,
            "caught_ball": ball or d["ball"], "aspects": aspects,
        }
        row.update(dict(zip(IV_ORDER, ivs)))
        row.update(dict(zip(EV_ORDER, evs or [0] * 6)))
        return pd.DataFrame([row])

    def _features(self, row):
        snap = self.meta["market_snapshot"]
        line = species_info(row["species"].iloc[0], row["form"].iloc[0]).line
        vals = {**snap["market"], **snap["species"].get(row["species"].iloc[0], {}), **snap["line"].get(line, {})}
        market = pd.DataFrame([{c: (np.nan if vals.get(c) is None else vals[c]) for c in MARKET_COLS}])
        for c in market.columns:
            if c.endswith("_n"):
                market[c] = market[c].fillna(0)
        return build_features(row, self.ctx, market)[self.meta["price_features"]]

    def _raw_price(self, X) -> np.ndarray:
        return np.exp(np.mean([np.log(predict_price(m, X, self.target)) for m in self.price_models], axis=0))

    def price(self, row) -> tuple[float, np.ndarray]:
        X = self._features(row)
        floors = floors_for(X)
        q = widen(self._raw_price(X), self.meta.get("interval_scale", 1.0), floors)
        return float(floors[0]), q[0]

    def sell_chance(self, row, prices) -> np.ndarray:
        X = self._features(row)
        p50 = self._raw_price(X)[0, 1]
        prices = np.asarray(prices, float)
        Xr = pd.concat([X] * len(prices), ignore_index=True)
        Xb = sell_features(Xr, prices, np.full(len(prices), p50))[self.meta["sell_features"]]
        p = self.sell_model.predict_proba(Xb)[:, 1]
        iso = self.meta.get("isotonic")
        if iso:
            p = np.interp(p, iso["x"], iso["y"])
        return p


def parse_six(text: str, name: str) -> list[int]:
    vals = [int(v) for v in text.replace("/", ",").split(",")]
    if len(vals) != 6:
        raise SystemExit(f"--{name} needs six numbers (HP, Atk, Def, SpA, SpD, Spe)")
    return vals


def describe_role(species: str, form: str, nature: str) -> str:
    info = species_info(to_id(species), form)
    if not info.roles:
        return "not in the Pokédex (custom species), so no role could be worked out"
    if info.breeding:
        return "breeding Pokémon: every IV counts, and any nature that suits some role is valued"
    best = max(info.roles, key=lambda r: r.nature_fit(nature))
    kind = "wall" if best.wall else f"{best.attack} attacker"
    others = [r.name for r in info.roles if r is not best]
    extra = f" (could also evolve into {', '.join(others)})" if others else ""
    return f"{best.name}: {best.speed} {kind}{extra}; {nature.title()} scores {best.nature_fit(nature):+.1f} for it"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--ivs", required=True, help="HP,Atk,Def,SpA,SpD,Spe e.g. 31,31,31,0,31,31")
    ap.add_argument("--nature", required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--evs", help="HP,Atk,Def,SpA,SpD,Spe (default all 0)")
    ap.add_argument("--form", default="normal")
    ap.add_argument("--shiny", action="store_true")
    ap.add_argument("--ability")
    ap.add_argument("--hidden-ability", action="store_true")
    ap.add_argument("--gender")
    ap.add_argument("--aspects", default="", help="comma-separated, e.g. galaxy")
    ap.add_argument("--held-item", default="")
    ap.add_argument("--ball")
    ap.add_argument("--not-breedable", action="store_true")
    ap.add_argument("--price", type=float, help="an asking price to check")
    ap.add_argument("--target", type=float, default=0.6, help="chance of selling to aim for (default 0.6)")
    a = ap.parse_args(argv)

    nature = a.nature.lower()
    if nature not in NATURES:
        raise SystemExit(f"Unknown nature {a.nature!r}")
    ivs = parse_six(a.ivs, "ivs")
    evs = parse_six(a.evs, "evs") if a.evs else None
    if any(not 0 <= v <= 31 for v in ivs):
        raise SystemExit("IVs must be between 0 and 31")
    if not 1 <= a.level <= 100:
        raise SystemExit("--level must be between 1 and 100")

    pr = Predictor()
    species = to_id(a.species)
    info = species_info(species, a.form)
    n_seen = pr.meta["species_counts"].get(species, 0)
    if not info.found and n_seen == 0:
        raise SystemExit(f"{a.species!r} is neither in the Pokédex nor in the GTS data")

    d = pr.meta["species_defaults"].get(species, pr.meta["global_defaults"])
    assumed = []
    if a.evs is None:
        assumed.append("no EVs")
    if a.ability is None and d["ability"]:
        assumed.append(f"ability {d['ability']}")
    if a.gender is None and d["gender"]:
        assumed.append(f"gender {d['gender']}")
    if a.ball is None and d["ball"]:
        assumed.append(f"ball {d['ball']}")

    row = pr.make_row(species, ivs, nature, a.level, evs, a.form, a.shiny, a.ability, a.hidden_ability,
                      a.gender, a.aspects, a.held_item, a.ball, False if a.not_breedable else None)
    floor, q = pr.price(row)
    tax = pr.meta["tax"]
    horizon = pr.meta["horizon_hours"]

    print(f"\n{species.title()} ({a.form}{', shiny' if a.shiny else ''}), level {a.level}, {nature.title()}, "
          f"IVs {'/'.join(map(str, ivs))}")
    if assumed:
        print(f"Assumed: {', '.join(assumed)}")
    if n_seen < 10:
        print(f"Warning: only {n_seen} past sales of this species, so the range is less reliable.")
    print(f"Role:    {describe_role(species, a.form, nature)}")
    top = pr.meta["line_top_natures"].get(info.line)
    if top:
        print("Best-selling natures for this line: " +
              ", ".join(f"{n.title()} (median {money(m)}, {int(c)} sales)" for n, m, c in top))
    print(f"\nGTS minimum price:     {money(floor)}")
    print(f"Estimated sale price:  {money(q[1])}  (likely range {money(q[0])} – {money(q[2])})")

    grid = nice_prices(max(floor, q[0] * 0.5), max(q[2] * 2, floor * 1.5))
    if floor not in grid:
        grid = [floor] + grid
    chances = pr.sell_chance(row, grid)
    print(f"\n{'Asking price':>14} {'Sells in ' + str(horizon) + ' h':>14} {'You receive':>12}")
    for p, c in zip(grid, chances):
        print(f"{money(p):>14} {c:>13.0%} {money(p * (1 - tax)):>12}")
    ok = [p for p, c in zip(grid, chances) if c >= a.target]
    if ok:
        print(f"\nSuggested price: {money(max(ok))} (highest round price with ≥{a.target:.0%} chance of selling)")
    else:
        print(f"\nNo price above the GTS minimum reaches a {a.target:.0%} chance; list at {money(grid[0])}.")
    if a.price is not None:
        if a.price < floor:
            print(f"\n{money(a.price)} is below the GTS minimum of {money(floor)}; the GTS won't accept it.")
        else:
            c = pr.sell_chance(row, [a.price])[0]
            print(f"\nAt {money(a.price)}: {c:.0%} chance of selling within {horizon} h; "
                  f"you'd receive {money(a.price * (1 - tax))} after the {tax:.0%} tax.")


if __name__ == "__main__":
    main(sys.argv[1:])
