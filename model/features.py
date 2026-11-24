"""Feature engineering shared by train.py, experiments.py and predict.py."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from lookup import BREEDING_IV_WEIGHTS, NATURES, STATS, get_rules, species_info

START = pd.Timestamp("2026-06-02", tz="UTC")
DETAILS_FROM = pd.Timestamp("2026-07-02", tz="UTC")   # nature/ability/aspects/ball logged from here

IV = {"hp": "iv_hp", "atk": "iv_attack", "def": "iv_defence",
      "spa": "iv_special_attack", "spd": "iv_special_defence", "spe": "iv_speed"}
EV = {s: c.replace("iv_", "ev_") for s, c in IV.items()}

ASPECT_IGNORE = {"male", "female", "genderless", "shiny"}
ASPECT_MIN_COUNT = 30

FEATURE_GROUPS: dict[str, list[str]] = {
    "identity": ["species", "form", "shiny", "line", "final_species"],
    "details": ["has_details", "ability", "hidden_ability", "gender", "breedable", "has_item", "ball"],
    "aspects": [],   # filled from the vocabulary (asp_*) + n_aspects
    "level_ev": ["level", "is_lvl1", "is_lvl100", "ev_total", "fully_trained"],
    "floor": ["log_floor"],
    "ivs": ["iv_hp", "iv_atk", "iv_def", "iv_spa", "iv_spd", "iv_spe", "iv_sum", "n_perfect", "n_zero"],
    "dex": ["dex_found", "bs_hp", "bs_atk", "bs_def", "bs_spa", "bs_spd", "bs_spe", "bst", "type1", "type2",
            "egg_group", "is_legendary", "is_mythical", "is_restricted", "is_ub", "is_paradox", "fully_evolved"],
    "nature_raw": ["nature", "nature_up", "nature_down"],
    "role": ["role_attack", "role_speed", "role_wall", "final_atk_spa_ratio", "final_base_spe",
             "final_bulk_ratio", "nature_fit", "iv_useful_score", "n_useful_perfect", "atk_dump_useful",
             "spe_dump_useful", "ev_useful_share"],
    "nature_market": ["line_nature", "species_nature", "nature_popularity"],
    "time": ["days"],
    "market": ["species_recent_logp", "species_recent_n", "line_recent_logp", "line_recent_n",
               "market_recent_ratio", "market_sell_rate"],
    "market_long": ["species_all_logp", "species_all_n", "line_all_logp", "line_all_n"],
}

SPECIES_WINDOW_DAYS = 30   # species/line price memory
MARKET_WINDOW_DAYS = 14    # market-wide price level

CAT_FEATURES = {"species", "form", "line", "final_species", "ability", "hidden_ability", "gender", "ball",
                "type1", "type2", "egg_group", "nature", "nature_up", "nature_down", "role_attack",
                "role_speed", "line_nature", "species_nature"}


def aspect_tokens(aspects: str) -> list[str]:
    return [t for t in (aspects or "").split(",") if t and t not in ASPECT_IGNORE]


def aspect_col(token: str) -> str:
    return "asp_" + re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")


class FeatureContext:
    """Everything learned from the training rows that features depend on."""

    def __init__(self, aspect_vocab: list[str], nature_popularity: dict[str, float]):
        self.aspect_vocab = aspect_vocab
        self.nature_popularity = nature_popularity

    @classmethod
    def fit(cls, sales: pd.DataFrame) -> "FeatureContext":
        counts = Counter(t for a in sales["aspects"] for t in aspect_tokens(a))
        vocab = sorted(t for t, n in counts.items() if n >= ASPECT_MIN_COUNT)
        known = sales.loc[sales["nature"].isin(NATURES.keys()), "nature"]
        pop = (known.value_counts(normalize=True)).to_dict()
        return cls(vocab, {k: float(v) for k, v in pop.items()})

    def to_dict(self) -> dict:
        return {"aspect_vocab": self.aspect_vocab, "nature_popularity": self.nature_popularity}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureContext":
        return cls(d["aspect_vocab"], d["nature_popularity"])

    def feature_names(self, drop_groups: tuple[str, ...] = ()) -> list[str]:
        names = []
        for g, cols in FEATURE_GROUPS.items():
            if g in drop_groups:
                continue
            if g == "aspects":
                cols = [aspect_col(t) for t in self.aspect_vocab] + ["n_aspects"]
            names.extend(cols)
        return names


def _best_role(info, nature, ivs):
    def iv_score(role):
        w = BREEDING_IV_WEIGHTS if info.breeding else role.weights
        return sum(w[s] * ivs[s] for s in STATS) / (31 * sum(w.values()))

    if nature is not None:
        return max(info.roles, key=lambda r: (r.nature_fit(nature), iv_score(r)))
    return max(info.roles, key=iv_score)


def build_features(df: pd.DataFrame, ctx: FeatureContext, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw GTS rows (columns as in data.COLUMNS) -> model features, one row per input row.

    `market` holds the backward-looking market columns (MarketHistory.features); NaN if omitted.
    """
    rules = get_rules()
    rows = []
    for r in df.itertuples(index=False):
        has_details = r.event_time >= DETAILS_FROM
        nature = r.nature if (has_details and r.nature in NATURES) else None
        ivs = {s: int(getattr(r, IV[s])) for s in STATS}
        evs = {s: int(getattr(r, EV[s])) for s in STATS}
        info = species_info(r.species, r.form)
        n_perfect = sum(v == 31 for v in ivs.values())
        ha = bool(r.hidden_ability) if has_details else False
        floor = rules.floor(n_perfect, bool(r.shiny), ha, info.legendary, info.ultra_beast)
        ev_total = sum(evs.values())

        f = {
            "species": r.species, "form": r.form or "normal", "shiny": int(bool(r.shiny)),
            "line": info.line,
            "has_details": int(has_details),
            "ability": (r.ability or "unknown") if has_details else "unknown",
            "hidden_ability": ("yes" if r.hidden_ability else "no") if has_details else "unknown",
            "gender": (r.gender or "unknown") if has_details else "unknown",
            "breedable": int(bool(r.breedable)),
            "has_item": float(bool(r.held_item)) if has_details else np.nan,
            "ball": (r.caught_ball or "unknown") if has_details else "unknown",
            "level": int(r.level), "is_lvl1": int(r.level == 1), "is_lvl100": int(r.level == 100),
            "ev_total": ev_total, "fully_trained": int(ev_total >= 508),
            "log_floor": math.log(floor),
            "iv_hp": ivs["hp"], "iv_atk": ivs["atk"], "iv_def": ivs["def"],
            "iv_spa": ivs["spa"], "iv_spd": ivs["spd"], "iv_spe": ivs["spe"],
            "iv_sum": sum(ivs.values()), "n_perfect": n_perfect, "n_zero": sum(v == 0 for v in ivs.values()),
            "dex_found": int(info.found),
            "type1": info.types[0], "type2": info.types[1], "egg_group": info.egg_group,
            "is_legendary": int(info.legendary), "is_mythical": int(info.mythical),
            "is_restricted": int(info.restricted), "is_ub": int(info.ultra_beast),
            "is_paradox": int(info.paradox), "fully_evolved": int(info.fully_evolved),
            "nature": nature or "unknown",
            "nature_up": NATURES[nature][0] if nature and NATURES[nature] else ("neutral" if nature else "unknown"),
            "nature_down": NATURES[nature][1] if nature and NATURES[nature] else ("neutral" if nature else "unknown"),
            "line_nature": f"{info.line}|{nature or 'unknown'}",
            "species_nature": f"{r.species}|{nature or 'unknown'}",
            "nature_popularity": ctx.nature_popularity.get(nature, np.nan) if nature else np.nan,
            "days": (r.event_time - START).total_seconds() / 86400,
        }
        bs = info.base_stats or {}
        for s in STATS:
            f[f"bs_{s}"] = bs.get(s, np.nan)
        f["bst"] = sum(bs.values()) if bs else np.nan

        if info.roles:
            role = _best_role(info, nature, ivs)
            w = BREEDING_IV_WEIGHTS if info.breeding else role.weights
            f.update({
                "final_species": "breeding" if info.breeding else role.name,
                "role_attack": "breeding" if info.breeding else role.attack,
                "role_speed": "breeding" if info.breeding else role.speed,
                "role_wall": np.nan if info.breeding else int(role.wall),
                "final_atk_spa_ratio": role.atk_spa_ratio, "final_base_spe": role.base_spe,
                "final_bulk_ratio": role.bulk_ratio,
                "nature_fit": role.nature_fit(nature) if nature else np.nan,
                "iv_useful_score": sum(w[s] * ivs[s] for s in STATS) / (31 * sum(w.values())),
                "n_useful_perfect": sum(ivs[s] == 31 for s in STATS if w[s] >= 0.5),
                "atk_dump_useful": int(not info.breeding and role.weights["atk"] <= 0.1 and ivs["atk"] <= 1),
                "spe_dump_useful": int(not info.breeding and role.speed == "slow" and ivs["spe"] <= 1),
                "ev_useful_share": (sum(w[s] * evs[s] for s in STATS) / ev_total) if ev_total else np.nan,
            })
        else:
            f.update({"final_species": "unknown", "role_attack": "unknown", "role_speed": "unknown",
                      "role_wall": np.nan, "final_atk_spa_ratio": np.nan, "final_base_spe": np.nan,
                      "final_bulk_ratio": np.nan, "nature_fit": np.nan, "iv_useful_score": np.nan,
                      "n_useful_perfect": np.nan, "atk_dump_useful": 0, "spe_dump_useful": 0,
                      "ev_useful_share": np.nan})

        toks = set(aspect_tokens(r.aspects))
        for t in ctx.aspect_vocab:
            f[aspect_col(t)] = float(t in toks) if has_details else np.nan
        f["n_aspects"] = float(len(toks)) if has_details else np.nan
        rows.append(f)

    X = pd.DataFrame(rows, index=df.index)
    for c in FEATURE_GROUPS["market"] + FEATURE_GROUPS["market_long"]:
        X[c] = market[c].to_numpy() if market is not None else np.nan
    X = X[ctx.feature_names()]
    for c in X.columns:
        if c in CAT_FEATURES:
            X[c] = X[c].astype(str)
    return X


def floors_for(X: pd.DataFrame) -> np.ndarray:
    return np.round(np.exp(X["log_floor"].to_numpy()), 2)   # undo float error from exp(log(x))


def cat_features(columns) -> list[str]:
    return [c for c in columns if c in CAT_FEATURES]


def _np_times(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(x), utc=True).dt.tz_convert(None).to_numpy("datetime64[ns]")


class MarketHistory:
    """Backward-looking market signals. For a row at time t only events before t are used.

    - species_recent_logp / line_recent_logp: mean log sale price of that species / evolution line
      over the previous SPECIES_WINDOW_DAYS (and how many sales that was)
    - market_recent_ratio: mean log(price / GTS floor) over all sales in the previous MARKET_WINDOW_DAYS
    - market_sell_rate: share of listings made 8 to 1 days earlier that sold within 24 h
      (listings from the last day don't have a known outcome yet)
    """

    def __init__(self, sales: pd.DataFrame, sales_log_floor: np.ndarray, listings: pd.DataFrame | None):
        t = _np_times(sales["event_time"])
        lp = np.log(sales["price"].to_numpy().astype(float))
        lines = np.array([species_info(s, f).line for s, f in zip(sales["species"], sales["form"])])
        self._by = {"species": self._index(sales["species"].to_numpy(), t, lp),
                    "line": self._index(lines, t, lp)}
        order = np.argsort(t)
        self._mt = t[order]
        self._mcum = np.concatenate([[0.0], np.cumsum((lp - sales_log_floor)[order])])
        if listings is not None and len(listings):
            lt = _np_times(listings["event_time"])
            o = np.argsort(lt)
            self._lt = lt[o]
            self._lcum = np.concatenate([[0.0], np.cumsum(listings["sold_within"].to_numpy()[o])])
        else:
            self._lt, self._lcum = np.array([], dtype=t.dtype), np.array([0.0])

    @staticmethod
    def _index(keys, t, lp):
        out = {}
        df = pd.DataFrame({"k": keys, "t": t, "lp": lp}).sort_values("t")
        for k, g in df.groupby("k", sort=False):
            out[k] = (g["t"].to_numpy(), np.concatenate([[0.0], np.cumsum(g["lp"].to_numpy())]))
        return out

    @staticmethod
    def _window(times, cum, lo, hi):
        a, b = np.searchsorted(times, lo, "left"), np.searchsorted(times, hi, "left")
        n = b - a
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, (cum[b] - cum[a]) / np.maximum(n, 1), np.nan)
        return mean, n

    def features(self, times: pd.Series, species, lines) -> pd.DataFrame:
        t = _np_times(times)
        out = pd.DataFrame(index=range(len(t)))
        sw = np.timedelta64(SPECIES_WINDOW_DAYS, "D")
        for name, keys in (("species", np.asarray(species)), ("line", np.asarray(lines))):
            mean, n = np.full(len(t), np.nan), np.zeros(len(t))
            amean, an = np.full(len(t), np.nan), np.zeros(len(t))
            idx = self._by[name]
            for k in np.unique(keys):
                if k not in idx:
                    continue
                m = keys == k
                mean[m], n[m] = self._window(*idx[k], t[m] - sw, t[m])
                amean[m], an[m] = self._window(*idx[k], np.full(m.sum(), np.datetime64("2000-01-01", "ns")), t[m])
            out[f"{name}_recent_logp"], out[f"{name}_recent_n"] = mean, n
            out[f"{name}_all_logp"], out[f"{name}_all_n"] = amean, an
        out["market_recent_ratio"], _ = self._window(self._mt, self._mcum, t - np.timedelta64(MARKET_WINDOW_DAYS, "D"), t)
        out["market_sell_rate"], _ = self._window(self._lt, self._lcum, t - np.timedelta64(8, "D"),
                                                  t - np.timedelta64(1, "D"))
        return out


def lines_for(df: pd.DataFrame) -> list[str]:
    return [species_info(s, f).line for s, f in zip(df["species"], df["form"])]
