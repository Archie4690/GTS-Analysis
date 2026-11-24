"""Tuning harness: try named ideas, score each on the VALIDATION set, log every result.

    python model/tune.py               # runs every round below
    python model/tune.py market hparams

Results are appended to model/artifacts/tuning_log.json, one entry per run, so every idea we tried
(including the ones that didn't help) stays on record. docs/tuning.md summarises the log.
The test set is never touched here.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from config import ARTIFACT_DIR
from features import FeatureContext, MarketHistory, build_features, lines_for
from modelling import fit_sell, predict_price, price_metrics, sell_features, sell_metrics
from train import LISTING_WEIGHTS, fit_config, jsonable, log, oof_p50, prepare

LOG = ARTIFACT_DIR / "tuning_log.json"
BASE_CFG = {"target": "log_ratio", "half_life": 120, "drop_instant": False}


class Bench:
    """Data and features built once, shared by every candidate."""

    def __init__(self):
        sales, listings, cuts = prepare()
        self.sales, self.listings = sales, listings
        self.tr, self.va = [(sales["split"] == s).to_numpy() for s in ("train", "val")]
        self.ltr, self.lva = [(listings["split"] == s).to_numpy() for s in ("train", "val")]
        self.price = sales["price"].to_numpy().astype(float)
        self.ctx = FeatureContext.fit(sales[self.tr])
        X0 = build_features(sales, self.ctx)
        hist = MarketHistory(sales, X0["log_floor"].to_numpy(), listings)
        m_s = hist.features(sales["event_time"], sales["species"], lines_for(sales))
        m_l = hist.features(listings["event_time"], listings["species"], lines_for(listings))
        self.X = build_features(sales, self.ctx, m_s)
        self.XL = build_features(listings, self.ctx, m_l)
        self._p50 = {}

    def cols(self, drop=()):
        return self.ctx.feature_names(tuple(drop))

    def score_a(self, cfg, drop=()):
        c = self.cols(drop)
        model = fit_config(cfg, self.X[c], self.sales, self.tr, self.X.loc[self.va, c], self.price[self.va])
        q = predict_price(model, self.X.loc[self.va, c], cfg["target"])
        return model, {"best_iteration": model.get_best_iteration(), **price_metrics(self.price[self.va], q)}

    def score_b(self, cfg, drop_a=(), drop_b=(), weighting="none", depth=6):
        """Model B on top of Model A (out-of-fold for training listings), scored on validation listings."""
        c = self.cols(drop_a)
        key = (json.dumps(cfg, sort_keys=True), tuple(drop_a))
        if key not in self._p50:      # Model A + out-of-fold estimates are shared across Model B candidates
            model_a, _ = self.score_a(cfg, drop_a)
            iters = model_a.get_best_iteration() + 1
            p50 = np.full(len(self.listings), np.nan)
            p50[self.ltr] = oof_p50(cfg, self.X[c], self.sales, self.tr, self.XL[c], self.ltr, self.listings,
                                    iters)[self.ltr]
            p50[self.lva] = predict_price(model_a, self.XL.loc[self.lva, c], cfg["target"])[:, 1]
            self._p50[key] = p50
        Xb = sell_features(self.XL[c], self.listings["price"].to_numpy().astype(float), self._p50[key])
        Xb = Xb[[x for x in Xb.columns if x not in drop_b]]
        y = self.listings["sold_within"].to_numpy()
        w = LISTING_WEIGHTS[weighting](self.listings["n_listings"].to_numpy()[self.ltr])
        mb = fit_sell(Xb[self.ltr], y[self.ltr], w, Xb[self.lva], y[self.lva], depth=depth)
        return {"best_iteration": mb.get_best_iteration(),
                **sell_metrics(y[self.lva], mb.predict_proba(Xb[self.lva])[:, 1])}


def record(round_name, candidate, hypothesis, model, settings, result):
    entries = json.loads(LOG.read_text()) if LOG.exists() else []
    entries.append(jsonable({"time": pd.Timestamp.now(tz="UTC"), "round": round_name, "candidate": candidate,
                             "hypothesis": hypothesis, "model": model, "settings": settings, "val": result}))
    LOG.write_text(json.dumps(entries, indent=1))
    key = "mean_abs_log_error" if model == "A" else "log_loss"
    log(f"  [{round_name}] {candidate}: val {key} {result[key]:.4f}")


def round_market(b: Bench):
    h = "Prices drift; recent prices of the same species/line and the market level should track that better than the date."
    for name, drop in [("baseline (no market features)", ("market",)),
                       ("+ market features", ()),
                       ("+ market features, no date", ("time",))]:
        _, r = b.score_a(BASE_CFG, drop)
        record("market", name, h, "A", {"cfg": BASE_CFG, "dropped": list(drop)}, r)


def round_hparams(b: Bench, drop=()):
    h = "Tree depth, learning rate and regularisation were left at defaults; check nearby values."
    for name, extra in [("depth 4", {"depth": 4}), ("depth 8", {"depth": 8}), ("l2 1", {"l2_leaf_reg": 1}),
                        ("l2 10", {"l2_leaf_reg": 10}), ("lr 0.03", {"learning_rate": 0.03})]:
        cfg = {**BASE_CFG, **extra}
        _, r = b.score_a(cfg, drop)
        record("hparams", name, h, "A", {"cfg": cfg, "dropped": list(drop)}, r)


def round_sell(b: Bench, drop_a=()):
    h = ("The share of listings that sell swings by month (39% Jun, 28% Sep, 40% Nov). A recent sell-rate "
         "feature should follow that; the raw date locks the model into old months.")
    for name, drop_b in [("with date, no market sell rate", ("market_sell_rate",)),
                         ("no date, no market sell rate", ("days", "market_sell_rate")),
                         ("with date + market sell rate", ()),
                         ("no date + market sell rate", ("days",))]:
        r = b.score_b(BASE_CFG, drop_a, drop_b)
        record("sell", name, h, "B", {"cfg": BASE_CFG, "dropped_a": list(drop_a), "dropped_b": list(drop_b)}, r)


def round_hparams2(b: Bench, drop=()):
    h = "Depth 4 and a slower learning rate each helped on their own; check neighbours and the combination."
    for name, extra in [("depth 3", {"depth": 3}), ("depth 5", {"depth": 5}),
                        ("depth 4 + lr 0.03", {"depth": 4, "learning_rate": 0.03}),
                        ("depth 4 + l2 1", {"depth": 4, "l2_leaf_reg": 1})]:
        cfg = {**BASE_CFG, **extra}
        _, r = b.score_a(cfg, drop)
        record("hparams2", name, h, "A", {"cfg": cfg, "dropped": list(drop)}, r)


BEST_CFG = {**BASE_CFG, "depth": 4}
BEST_DROP = ("market_long",)


def round_seeds(b: Bench):
    h = "Measure how much the validation score moves from the random seed alone, to know what counts as a real gain."
    for seed in range(5):
        cfg = {**BEST_CFG, "seed": seed}
        _, r = b.score_a(cfg, BEST_DROP)
        record("seeds", f"seed {seed}", h, "A", {"cfg": cfg, "dropped": list(BEST_DROP)}, r)


def round_long(b: Bench):
    h = "A species' whole price history (not just 30 days) may help rare species that sold long ago."
    for name, drop in [("30-day memory only", ("market_long",)), ("+ all-time memory", ())]:
        _, r = b.score_a(BEST_CFG, drop)
        record("long", name, h, "A", {"cfg": BEST_CFG, "dropped": list(drop)}, r)


def round_ensemble(b: Bench):
    h = "Averaging several models trained with different seeds usually smooths out each one's quirks."
    c = b.cols(BEST_DROP)
    logs = []
    for seed in range(5):
        cfg = {**BEST_CFG, "seed": seed}
        m, _ = b.score_a(cfg, BEST_DROP)
        logs.append(np.log(predict_price(m, b.X.loc[b.va, c], cfg["target"])))
    q = np.exp(np.mean(logs, axis=0))
    record("ensemble", "average of 5 seeds", h, "A", {"cfg": BEST_CFG, "dropped": list(BEST_DROP), "seeds": 5},
           price_metrics(b.price[b.va], q))


def round_regrid(b: Bench):
    h = "Target and recency weighting were chosen at depth 6; re-check them at depth 4."
    for target in ("log", "log_ratio"):
        for hl in (None, 60, 120, 240):
            cfg = {**BEST_CFG, "target": target, "half_life": hl}
            _, r = b.score_a(cfg, BEST_DROP)
            record("regrid", f"{target}, half-life {hl}", h, "A", {"cfg": cfg, "dropped": list(BEST_DROP)}, r)


def round_b_depth(b: Bench):
    h = "The sell-chance model also used depth 6 by default; check shallower and deeper trees."
    for depth in (4, 8):
        r = b.score_b(BEST_CFG, BEST_DROP, ("days",), depth=depth)
        record("b_depth", f"depth {depth}", h, "B", {"cfg": BEST_CFG, "depth": depth, "dropped_b": ["days"]}, r)
    r = b.score_b(BEST_CFG, BEST_DROP, ("days",), depth=6)
    record("b_depth", "depth 6 (with depth-4 price model)", h, "B", {"cfg": BEST_CFG, "depth": 6, "dropped_b": ["days"]}, r)


def round_confirm(b: Bench):
    h = ("Single runs move by up to 0.003 with the seed, which is as big as most gains so far. "
         "Re-score the main decisions averaged over 3 seeds.")
    cands = [("v1: depth 6, no market features", {**BASE_CFG}, ("market", "market_long")),
             ("depth 6 + market features", {**BASE_CFG}, ("market_long",)),
             ("depth 4 + market features", {**BASE_CFG, "depth": 4}, ("market_long",)),
             ("depth 4 + market, half-life 240", {**BASE_CFG, "depth": 4, "half_life": 240}, ("market_long",))]
    for name, cfg, drop in cands:
        runs = [b.score_a({**cfg, "seed": sd}, drop)[1] for sd in (0, 1, 2)]
        r = {k: float(np.mean([x[k] for x in runs])) for k in runs[0]}
        r["seed_spread"] = float(np.ptp([x["mean_abs_log_error"] for x in runs]))
        record("confirm", name + " (3-seed mean)", h, "A", {"cfg": cfg, "dropped": list(drop), "seeds": [0, 1, 2]}, r)


def round_catboost(b: Bench):
    h = ("CatBoost's structural options were left at defaults: how many categorical columns it combines "
         "(max_ctr_complexity), the tree growth policy, and when categories are one-hot encoded.")
    for name, extra in [("max_ctr_complexity 1", {"max_ctr_complexity": 1}),
                        ("max_ctr_complexity 2", {"max_ctr_complexity": 2}),
                        ("Lossguide trees, 16 leaves", {"grow_policy": "Lossguide", "max_leaves": 16}),
                        ("one_hot_max_size 10", {"one_hot_max_size": 10})]:
        cfg = {**BEST_CFG, "extra": extra}
        runs = [b.score_a({**cfg, "seed": sd}, BEST_DROP)[1] for sd in (0, 1, 2)]
        r = {k: float(np.mean([x[k] for x in runs])) for k in runs[0]}
        record("catboost", name + " (3-seed mean)", h, "A", {"cfg": cfg, "dropped": list(BEST_DROP)}, r)


ROUNDS = {"catboost": round_catboost, "confirm": round_confirm, "market": round_market, "hparams": round_hparams, "sell": round_sell, "hparams2": round_hparams2,
          "seeds": round_seeds, "long": round_long, "ensemble": round_ensemble, "regrid": round_regrid,
          "b_depth": round_b_depth}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(ROUNDS)
    bench = Bench()
    for r in wanted:
        log(f"Round: {r}")
        ROUNDS[r](bench)
