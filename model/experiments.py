"""Is more data always better? Learning curve, data-age windows and feature-group ablations.

    python model/experiments.py      (run after train.py; uses its chosen configuration)

All scores are Model A's error on the validation set (the 20%), so the test set stays untouched.
Writes model/artifacts/experiments.json and docs/figures/{learning_curve,data_age,feature_groups}.png.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import plots
from config import ARTIFACT_DIR, FIG_DIR, SEED
from features import FEATURE_GROUPS, FeatureContext
from modelling import predict_price, price_metrics
from train import fit_config, jsonable, log, prepare, with_market

FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
WINDOWS_DAYS = [30, 60, 90, None]   # None = all of the training period


def val_score(cfg, X, sales, mask, va):
    model = fit_config(cfg, X, sales, mask, X[va], sales["price"].to_numpy()[va])
    return price_metrics(sales["price"].to_numpy()[va], predict_price(model, X[va], cfg["target"]))


def main():
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    cfg = metrics["chosen_config"]
    dropped = tuple(metrics["dropped_groups"])
    out = {"config": cfg, "dropped_groups": list(dropped)}

    sales, listings, cuts = prepare()
    tr, va = (sales["split"] == "train").to_numpy(), (sales["split"] == "val").to_numpy()
    ctx = FeatureContext.fit(sales[tr])
    X_full, _ = with_market(sales, listings, ctx)
    X = X_full[ctx.feature_names(dropped)]
    rng = np.random.default_rng(SEED)

    # 1. Learning curve
    log("Learning curve")
    curve = []
    train_idx = np.flatnonzero(tr)
    for frac in FRACTIONS:
        scores = []
        for seed in (SEEDS if frac < 1 else [0]):
            pick = np.random.default_rng(seed).choice(train_idx, int(len(train_idx) * frac), replace=False)
            mask = np.zeros(len(sales), bool)
            mask[pick] = True
            scores.append(val_score(cfg, X, sales, mask, va))
        curve.append({"fraction": frac, "n_sales": int(len(train_idx) * frac),
                      "mean_abs_log_error": float(np.mean([s["mean_abs_log_error"] for s in scores])),
                      "median_abs_pct_error": float(np.mean([s["median_abs_pct_error"] for s in scores]))})
        log(f"  {frac:.0%}: {curve[-1]}")
    out["learning_curve"] = curve

    # 2. Data age: only the most recent N days of the training period
    log("Data-age windows")
    windows = []
    cut1 = cuts[0]
    for days in WINDOWS_DAYS:
        mask = tr.copy()
        if days:
            mask &= (sales["event_time"] >= cut1 - pd.Timedelta(days=days)).to_numpy()
        for weighting in ("chosen", "none"):
            c = dict(cfg) if weighting == "chosen" else {**cfg, "half_life": None}
            s = val_score(c, X, sales, mask, va)
            windows.append({"window_days": days or "all", "recency_weighting": c["half_life"],
                            "n_sales": int(mask.sum()), **s})
            log(f"  last {days or 'all'} days, half-life {c['half_life']}: {s['mean_abs_log_error']:.4f}")
    out["data_age"] = windows

    # 3. Feature groups: drop one at a time
    log("Feature-group ablation")
    base = val_score(cfg, X, sales, tr, va)
    groups = []
    for g in FEATURE_GROUPS:
        if g in dropped:
            continue
        keep = set(ctx.feature_names(dropped + (g,)))
        removed = [c for c in X.columns if c not in keep]
        # Keep the columns (prediction still needs log_floor for the GTS minimum) but hide them from the trees.
        c = {**cfg, "extra": {**(cfg.get("extra") or {}), "ignored_features": removed}}
        s = val_score(c, X, sales, tr, va)
        groups.append({"group": g, "n_features_removed": len(removed),
                       "mean_abs_log_error": s["mean_abs_log_error"],
                       "change_vs_all": s["mean_abs_log_error"] - base["mean_abs_log_error"]})
        log(f"  without {g}: {s['mean_abs_log_error']:.4f} ({groups[-1]['change_vs_all']:+.4f})")
    out["feature_groups"] = {"all_features": base, "drop_one": groups}

    # 4. Noisy rows: how much junk is there?
    big = listings["price"] >= 10_000_000
    out["noise"] = {
        "instant_sales_share": float((sales["secs_to_sell"] < 60).mean()),
        "troll_listings_10m_plus": int(big.sum()),
        "troll_listings_sold": int(listings.loc[big, "sold_within"].sum()),
        "pokemon_listed_10_plus": int((listings.groupby("pokemon_id").size() >= 10).sum()),
    }
    (ARTIFACT_DIR / "experiments.json").write_text(json.dumps(jsonable(out), indent=1))

    # Figures
    plots.line([c["n_sales"] for c in curve],
               {"Model A, validation set": ([np.exp(c["mean_abs_log_error"]) - 1 for c in curve], plots.BLUE)},
               FIG_DIR / "learning_curve.png", "Learning curve: does more training data help?",
               "Training sales used", "Typical error (× − 1)",
               xticks=[f"{c['n_sales']:,}\n({c['fraction']:.0%})" for c in curve])
    labels = [f"{w}d" if w else "All" for w in WINDOWS_DAYS]
    chosen = [np.exp(w["mean_abs_log_error"]) - 1 for w in windows if w["recency_weighting"] == cfg["half_life"]]
    unweighted = [np.exp(w["mean_abs_log_error"]) - 1 for w in windows if w["recency_weighting"] is None]
    series = {"No recency weighting": (unweighted, plots.ORANGE)}
    if cfg["half_life"]:
        series = {f"Recency weighting (half-life {cfg['half_life']} d)": (chosen, plots.BLUE), **series}
    plots.line(list(range(len(labels))), series, FIG_DIR / "data_age.png",
               "Training on only the most recent data", "Training window (before 9 Sep cut)",
               "Typical error (× − 1)", xticks=labels)
    order = sorted(groups, key=lambda g: -g["change_vs_all"])
    plots.bars([g["group"] for g in order], np.array([g["change_vs_all"] for g in order]),
               FIG_DIR / "feature_groups.png", "Extra validation error when a feature group is removed",
               "Change in mean absolute log error (higher = group matters more)")
    log("Done")


if __name__ == "__main__":
    main()
