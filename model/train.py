"""Train and evaluate both models on a chronological 70/20/10 split, then refit on everything.

    python model/train.py

Writes model/artifacts/{price.cbm, sell.cbm, meta.json, metrics.json} and docs/figures/*.png.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import plots
from config import (ARTIFACT_DIR, BASE_DROP_GROUPS, ENSEMBLE_SEEDS, FIG_DIR, INSTANT_SALE_SECONDS, PRICE_HPARAMS,
                    SEED, SELL_DEPTH, SELL_HORIZON_HOURS, SELL_USE_DATE)
from data import assign_split, build_tables, load_events, split_dates
from features import FeatureContext, MarketHistory, build_features, floors_for, lines_for
from lookup import NATURES, get_rules
from modelling import (calibration_table, fit_price, fit_sell, interval_scale, point_metrics, pokemon_fold,
                       predict_price, price_metrics, recency_weights, sell_features, sell_metrics, widen)

NATURE_ABLATION = [
    ("no nature or role features", ("nature_raw", "role", "nature_market")),
    ("+ raw nature", ("role", "nature_market")),
    ("+ role rule (nature_fit, role-weighted IVs/EVs)", ("nature_market",)),
    ("+ market preferences (line_nature, species_nature)", ()),
]
LISTING_WEIGHTS = {"none": lambda n: None, "1/sqrt(n)": lambda n: 1 / np.sqrt(n), "1/n": lambda n: 1 / n}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare():
    ev = load_events()
    sales, listings = build_tables(ev)
    cuts = split_dates(sales)
    sales["split"] = assign_split(sales, cuts)
    listings["split"] = assign_split(listings, cuts)
    return sales, listings, cuts


def fit_ensemble(cfg, X, sales, mask, X_val=None, price_val=None, iterations=None):
    return [fit_config({**cfg, "seed": sd}, X, sales, mask, X_val, price_val, iterations) for sd in ENSEMBLE_SEEDS]


def predict_ensemble(models, X, target):
    return np.exp(np.mean([np.log(predict_price(m, X, target)) for m in models], axis=0))


def train_mask(sales, base_mask, drop_instant):
    m = base_mask.copy()
    if drop_instant:
        m &= ~(sales["secs_to_sell"] < INSTANT_SALE_SECONDS).to_numpy()
    return m


def fit_config(cfg, X, sales, mask, X_val=None, price_val=None, iterations=None):
    m = train_mask(sales, mask, cfg["drop_instant"])
    w = recency_weights(sales.loc[m, "event_time"], cfg["half_life"])
    return fit_price(X[m], sales["price"].to_numpy()[m], cfg["target"], w, X_val, price_val, iterations,
                     learning_rate=cfg.get("learning_rate", 0.06), depth=cfg.get("depth", 6),
                     l2_leaf_reg=cfg.get("l2_leaf_reg", 3), seed=cfg.get("seed", SEED),
                     extra=cfg.get("extra"))


def species_buckets(species: pd.Series, train_counts: pd.Series) -> np.ndarray:
    n = species.map(train_counts).fillna(0).to_numpy()
    return np.select([n >= 30, n >= 10], ["30+ sales", "10-29 sales"], "under 10 sales")


def oof_p50(cfg, X_sales, sales, sale_mask, X_list, list_mask, listings, iterations):
    """Model A p50 for listings, each predicted by a model that never saw that Pokémon's sale."""
    out = np.full(len(listings), np.nan)
    s_fold, l_fold = pokemon_fold(sales["pokemon_id"]), pokemon_fold(listings["pokemon_id"])
    for k in range(5):
        model = fit_config(cfg, X_sales, sales, sale_mask & (s_fold != k), iterations=iterations)
        idx = list_mask & (l_fold == k)
        out[idx] = predict_price(model, X_list[idx], cfg["target"])[:, 1]
    return out


def with_market(sales, listings, ctx, return_history=False):
    """Features for sales and listings, including the backward-looking market columns."""
    base = build_features(sales, ctx)
    hist = MarketHistory(sales, base["log_floor"].to_numpy(), listings)
    X = build_features(sales, ctx, hist.features(sales["event_time"], sales["species"], lines_for(sales)))
    XL = build_features(listings, ctx,
                        hist.features(listings["event_time"], listings["species"], lines_for(listings)))
    return (X, XL, hist) if return_history else (X, XL)


def market_snapshot(hist, sales, at):
    """Market columns as of the end of the data, per species and per line, for predict.py."""
    at = at + pd.Timedelta(seconds=1)
    species = sorted(sales["species"].unique())
    lines = sorted(set(lines_for(sales)))
    sp = hist.features(pd.Series([at] * len(species)), species, ["?"] * len(species))
    ln = hist.features(pd.Series([at] * len(lines)), ["?"] * len(lines), lines)
    clean = lambda v: None if pd.isna(v) else float(v)
    sp_cols = [c for c in sp.columns if c.startswith("species_")]
    ln_cols = [c for c in ln.columns if c.startswith("line_")]
    return {
        "as_of": at,
        "species": {k: {c: clean(sp[c].iloc[i]) for c in sp_cols} for i, k in enumerate(species)},
        "line": {k: {c: clean(ln[c].iloc[i]) for c in ln_cols} for i, k in enumerate(lines)},
        "market": {c: clean(sp[c].iloc[0]) for c in sp.columns if c.startswith("market_")},
    }


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    log("Loading data")
    sales, listings, cuts = prepare()
    tr, va, te = [(sales["split"] == s).to_numpy() for s in ("train", "val", "test")]
    ltr, lva, lte = [(listings["split"] == s).to_numpy() for s in ("train", "val", "test")]
    price = sales["price"].to_numpy().astype(float)
    report["split"] = {
        "cut_dates": [cuts[0], cuts[1]],
        "sales": {s: int(m.sum()) for s, m in zip(("train", "val", "test"), (tr, va, te))},
        "listings": {s: int(m.sum()) for s, m in zip(("train", "val", "test"), (ltr, lva, lte))},
        "data_range": [sales["event_time"].min(), sales["event_time"].max()],
    }
    log(f"Split: {report['split']['sales']} sales, {report['split']['listings']} listings")

    ctx = FeatureContext.fit(sales[tr])
    X_full, XL_full = with_market(sales, listings, ctx)

    # ---- checks
    below_s = int((price < floors_for(X_full) - 1e-6).sum())
    below_l = int((listings["price"].to_numpy() < floors_for(XL_full) - 1e-6).sum())
    report["checks"] = {"sales_below_floor": below_s, "listings_below_floor": below_l,
                        "dex_match_rate_sales": float(X_full["dex_found"].mean()),
                        "instant_sale_share": float((sales["secs_to_sell"] < INSTANT_SALE_SECONDS).mean())}
    log(f"Floor check: {below_s} sales / {below_l} listings below floor; "
        f"dex match {report['checks']['dex_match_rate_sales']:.1%}")

    # ---- baselines
    glob = float(np.median(price[tr]))
    spec_med = sales[tr].groupby("species")["price"].median()
    base_species = sales["species"].map(spec_med).fillna(glob).to_numpy()
    floors = floors_for(X_full)
    report["baselines"] = {
        name: {
            "global_median": point_metrics(price[m], np.full(m.sum(), glob)),
            "species_median": point_metrics(price[m], np.maximum(base_species[m], floors[m])),
            "floor_only": point_metrics(price[m], floors[m]),
        } for name, m in (("val", va), ("test", te))
    }

    # ---- Model A tuning on validation
    log("Tuning Model A (12 configurations)")
    tuning = []
    for target, hl, drop in itertools.product(["log", "log_ratio"], [None, 60, 120], [False, True]):
        cfg = {"target": target, "half_life": hl, "drop_instant": drop, **PRICE_HPARAMS}
        Xg = X_full[ctx.feature_names(BASE_DROP_GROUPS)]
        model = fit_config(cfg, Xg, sales, tr, Xg[va], price[va])
        q = predict_price(model, Xg[va], target)
        tuning.append({**cfg, "best_iteration": model.get_best_iteration(), **price_metrics(price[va], q)})
        log(f"  {cfg} -> val log err {tuning[-1]['mean_abs_log_error']:.4f}")
    best_row = min(tuning, key=lambda r: r["mean_abs_log_error"])
    cfg = {k: best_row[k] for k in ("target", "half_life", "drop_instant", *PRICE_HPARAMS)}
    report["tuning"] = tuning
    report["chosen_config"] = cfg
    log(f"Chosen: {cfg}")

    # ---- nature / role ablation
    log("Nature/role ablation")
    ablation = []
    for label, drop in NATURE_ABLATION:
        drop = tuple(drop) + BASE_DROP_GROUPS
        cols = ctx.feature_names(drop)
        model = fit_config(cfg, X_full[cols], sales, tr, X_full.loc[va, cols], price[va])
        q = predict_price(model, X_full.loc[va, cols], cfg["target"])
        ablation.append({"features": label, "dropped_groups": list(drop), "n_features": len(cols),
                         "best_iteration": model.get_best_iteration(), **price_metrics(price[va], q)})
        log(f"  {label}: val log err {ablation[-1]['mean_abs_log_error']:.4f}")
    best_abl = min(ablation, key=lambda r: r["mean_abs_log_error"])
    drop_groups = tuple(best_abl["dropped_groups"])
    report["nature_ablation"] = ablation
    report["dropped_groups"] = list(drop_groups)
    cols = ctx.feature_names(drop_groups)
    X, XL = X_full[cols], XL_full[cols]

    # ---- Model A: test
    log("Model A on the test set")
    ens_a = fit_ensemble(cfg, X, sales, tr, X[va], price[va])
    model_a = ens_a[0]
    a_iters = int(np.mean([m.get_best_iteration() for m in ens_a])) + 1
    q_te_raw = predict_ensemble(ens_a, X[te], cfg["target"])
    # Quantile ranges come out too narrow when prices drift; stretch them so the validation set gets ~50%.
    scale = interval_scale(price[va], predict_ensemble(ens_a, X[va], cfg["target"]), floors[va])
    q_te = widen(q_te_raw, scale, floors[te])
    train_counts = sales[tr]["species"].value_counts()
    bucket = species_buckets(sales["species"], train_counts)
    report["model_a"] = {"best_iteration": a_iters, "interval_scale": scale, "ensemble_seeds": list(ENSEMBLE_SEEDS),
                         "val": price_metrics(price[va], widen(predict_ensemble(ens_a, X[va], cfg["target"]), scale, floors[va])),
                         "test_coverage_before_widening": price_metrics(price[te], q_te_raw)["p25_p75_coverage"],
                         "test": price_metrics(price[te], q_te), "test_by_frequency": {}}
    for b in ("30+ sales", "10-29 sales", "under 10 sales"):
        m = bucket[te] == b
        report["model_a"]["test_by_frequency"][b] = {
            "model": price_metrics(price[te][m], q_te[m]),
            "species_median": point_metrics(price[te][m], np.maximum(base_species[te][m], floors[te][m])),
        }
    report["model_a"]["test_below_floor"] = int((q_te[:, 1] < floors[te] - 1e-6).sum())
    log(f"  test: {report['model_a']['test']}")

    # Shuffled 70/20/10 for comparison only.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(sales))
    n = len(sales)
    r_tr = np.zeros(n, bool); r_tr[perm[: int(0.7 * n)]] = True
    r_va = np.zeros(n, bool); r_va[perm[int(0.7 * n): int(0.9 * n)]] = True
    r_te = ~(r_tr | r_va)
    m_rand = fit_config(cfg, X, sales, r_tr, X[r_va], price[r_va])
    report["model_a"]["shuffled_split_test"] = price_metrics(price[r_te], predict_price(m_rand, X[r_te], cfg["target"]))
    log(f"  shuffled-split test: {report['model_a']['shuffled_split_test']}")

    imp = model_a.get_feature_importance()
    report["model_a"]["feature_importance"] = dict(sorted(zip(cols, map(float, imp)), key=lambda kv: -kv[1]))

    # ---- Model B
    log("Model B: out-of-fold price estimates for training listings")
    p50_list = np.full(len(listings), np.nan)
    p50_list[ltr] = oof_p50(cfg, X, sales, tr, XL, ltr, listings, a_iters)[ltr]
    p50_list[lva | lte] = predict_ensemble(ens_a, XL[lva | lte], cfg["target"])[:, 1]
    lprice = listings["price"].to_numpy().astype(float)
    Xb = sell_features(XL, lprice, p50_list)
    y = listings["sold_within"].to_numpy()
    nlist = listings["n_listings"].to_numpy()

    log("Model B: choosing listing weights")
    b_rows, b_models = [], {}
    for (name, fn), use_days in itertools.product(LISTING_WEIGHTS.items(), (SELL_USE_DATE,)):
        bcols = [c for c in Xb.columns if use_days or c != "days"]
        mb = fit_sell(Xb.loc[ltr, bcols], y[ltr], fn(nlist[ltr]), Xb.loc[lva, bcols], y[lva], depth=SELL_DEPTH)
        b_models[(name, use_days)] = (mb, bcols)
        b_rows.append({"weighting": name, "uses_date": use_days, "best_iteration": mb.get_best_iteration(),
                       **sell_metrics(y[lva], mb.predict_proba(Xb.loc[lva, bcols])[:, 1])})
        log(f"  {name}, date={use_days}: val log loss {b_rows[-1]['log_loss']:.4f}, AUC {b_rows[-1]['auc']:.3f}")
    best_b = min(b_rows, key=lambda r: r["log_loss"])
    model_b, bcols = b_models[(best_b["weighting"], best_b["uses_date"])]
    Xb = Xb[bcols]
    b_iters = model_b.get_best_iteration() + 1

    p_val = model_b.predict_proba(Xb[lva])[:, 1]
    p_te = model_b.predict_proba(Xb[lte])[:, 1]
    val_ece = sell_metrics(y[lva], p_val)["calibration_error"]
    iso = None
    if val_ece > 0.03:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y[lva])
    p_te_final = iso.predict(p_te) if iso is not None else p_te

    gap = Xb["price_gap"].to_numpy()
    edges = np.quantile(gap[ltr], np.linspace(0, 1, 11)[1:-1])
    rates = pd.Series(y[ltr]).groupby(np.digitize(gap[ltr], edges)).mean()
    base_gap = pd.Series(np.digitize(gap[lte], edges)).map(rates).fillna(y[ltr].mean()).to_numpy()
    report["model_b"] = {
        "horizon_hours": SELL_HORIZON_HOURS, "weighting_choice": b_rows, "chosen_weighting": best_b["weighting"],
        "best_iteration": b_iters, "val_calibration_error_raw": val_ece, "isotonic_calibration": iso is not None,
        "chosen_uses_date": best_b["uses_date"],
        "test": sell_metrics(y[lte], p_te_final),
        "test_raw": sell_metrics(y[lte], p_te),
        "baselines_test": {
            "overall_rate": sell_metrics(y[lte], np.full(lte.sum(), y[ltr].mean())) | {"auc": 0.5},
            "price_gap_deciles": sell_metrics(y[lte], base_gap),
        },
        "calibration_table_test": calibration_table(y[lte], p_te_final).to_dict(orient="records"),
    }
    log(f"  test: {report['model_b']['test']}")

    # ---- figures from the evaluation models
    plots.predicted_vs_actual(price[te], q_te[:, 1], FIG_DIR / "predicted_vs_actual.png",
                              "Predicted vs actual price (test set)")
    labels = ["30+ sales", "10-29 sales", "under 10 sales"]
    tbf = report["model_a"]["test_by_frequency"]
    plots.error_by_bucket(["30+", "10–29", "Under 10"],
                          [tbf[l]["model"]["median_abs_pct_error"] for l in labels],
                          [tbf[l]["species_median"]["median_abs_pct_error"] for l in labels],
                          [tbf[l]["model"]["n"] for l in labels], FIG_DIR / "error_by_frequency.png",
                          "Error by how often a species sold in training")
    plots.calibration(calibration_table(y[lte], p_te_final), FIG_DIR / "calibration.png",
                      "Sell-chance calibration (test listings)")
    fi = report["model_a"]["feature_importance"]
    plots.importance(list(fi.keys()), list(fi.values()), FIG_DIR / "feature_importance.png",
                     "What drives the price model (top 20)")

    # ---- refit on all data
    log("Refitting on all data")
    ctx_all = FeatureContext.fit(sales)
    cols_all = ctx_all.feature_names(drop_groups)
    X_all, XL_all, hist_all = with_market(sales, listings, ctx_all, return_history=True)
    X_all, XL_all = X_all[cols_all], XL_all[cols_all]
    everything = np.ones(len(sales), bool)
    final_a_iters = int(a_iters * 1.1) + 1
    final_b_iters = int(b_iters * 1.1) + 1
    final_a = fit_ensemble(cfg, X_all, sales, everything, iterations=final_a_iters)
    p50_all = oof_p50(cfg, X_all, sales, everything, XL_all, np.ones(len(listings), bool), listings, final_a_iters)
    Xb_all = sell_features(XL_all, lprice, p50_all)[bcols]
    final_b = fit_sell(Xb_all, y, LISTING_WEIGHTS[best_b["weighting"]](nlist), iterations=final_b_iters,
                       depth=SELL_DEPTH)
    for old in ARTIFACT_DIR.glob("price*.cbm"):
        old.unlink()
    price_files = []
    for sd, m in zip(ENSEMBLE_SEEDS, final_a):
        price_files.append(f"price_{sd}.cbm")
        m.save_model(str(ARTIFACT_DIR / price_files[-1]))
    final_b.save_model(str(ARTIFACT_DIR / "sell.cbm"))

    # Lookups for predict.py
    detailed = sales[sales["event_time"] >= pd.Timestamp("2026-07-02", tz="UTC")]
    def mode(s):
        s = s[s != ""]
        return s.mode().iloc[0] if len(s) else ""
    defaults = detailed.groupby("species").agg(ability=("ability", mode), gender=("gender", mode),
                                                ball=("caught_ball", mode), breedable=("breedable", "mean"))
    defaults["breedable"] = defaults["breedable"] >= 0.5
    global_default = {"ability": "", "gender": mode(detailed["gender"]), "ball": mode(detailed["caught_ball"]),
                      "breedable": True}
    lines = pd.Series(X_all["line"].to_numpy(), index=sales.index) if "line" in X_all else sales["species"]
    nat = detailed.assign(line=lines.loc[detailed.index])
    nat = nat[nat["nature"].isin(NATURES)]
    top = (nat.groupby(["line", "nature"])["price"].agg(["size", "median"]).reset_index()
           .query("size >= 3").sort_values(["line", "median"], ascending=[True, False]))
    line_natures = {l: g.head(3)[["nature", "median", "size"]].values.tolist() for l, g in top.groupby("line")}

    meta = {
        "trained_at": pd.Timestamp.now(tz="UTC"), "data_end": sales["event_time"].max(),
        "config": cfg, "dropped_groups": list(drop_groups), "price_models": price_files, "price_features": cols_all, "interval_scale": scale,
        "sell_features": list(Xb_all.columns), "context": ctx_all.to_dict(),
        "horizon_hours": SELL_HORIZON_HOURS, "tax": get_rules().tax,
        "isotonic": None if iso is None else {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
        "iterations": {"price": final_a_iters, "sell": final_b_iters},
        "species_counts": sales["species"].value_counts().to_dict(),
        "species_defaults": defaults.to_dict(orient="index"), "global_defaults": global_default,
        "line_top_natures": line_natures,
        "market_snapshot": market_snapshot(hist_all, sales, sales["event_time"].max()),
    }
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(jsonable(meta), indent=1))
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(jsonable(report), indent=1))

    # Example demand curve from the final models
    from predict import Predictor
    pr = Predictor()
    row = pr.make_row("gible", [31, 31, 31, 12, 31, 31], "jolly")
    floor, q = pr.price(row)
    grid = np.geomspace(floor, max(q[2] * 4, floor * 10), 60)
    plots.demand_curve(grid, pr.sell_chance(row, grid), floor, q[1], FIG_DIR / "demand_curve.png",
                       "Example: 5-IV Jolly Gible (level 1)")
    log("Done. Artifacts in model/artifacts, figures in docs/figures")


if __name__ == "__main__":
    main()
