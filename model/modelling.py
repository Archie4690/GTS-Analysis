"""Model fitting, prediction and metrics shared by train.py and experiments.py."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import QUANTILES, SEED
from features import cat_features, floors_for

QUANTILE_LOSS = "MultiQuantile:alpha=" + ",".join(str(q) for q in QUANTILES)
MAX_ITERS = 3000


# --------------------------------------------------------------------------- weights / folds

def recency_weights(times: pd.Series, half_life_days: float | None) -> np.ndarray | None:
    if not half_life_days:
        return None
    age = (times.max() - times).dt.total_seconds().to_numpy() / 86400
    return 0.5 ** (age / half_life_days)


def pokemon_fold(pokemon_ids: pd.Series, k: int = 5) -> np.ndarray:
    """Stable fold per Pokémon, so a Pokémon's sale and listings always share a fold."""
    return np.array([int(hashlib.md5(p.encode()).hexdigest()[:8], 16) % k for p in pokemon_ids])


# --------------------------------------------------------------------------- Model A: price

def price_target(price: np.ndarray, X: pd.DataFrame, target: str) -> np.ndarray:
    y = np.log(price)
    return y - X["log_floor"].to_numpy() if target == "log_ratio" else y


def fit_price(X, price, target="log", weights=None, X_val=None, price_val=None, iterations=None,
              learning_rate=0.06, depth=6, l2_leaf_reg=3, seed=SEED, extra=None, verbose=False) -> CatBoostRegressor:
    cats = cat_features(X.columns)
    model = CatBoostRegressor(
        loss_function=QUANTILE_LOSS, iterations=iterations or MAX_ITERS, learning_rate=learning_rate,
        depth=depth, l2_leaf_reg=l2_leaf_reg, random_seed=seed, thread_count=-1, verbose=verbose,
        allow_writing_files=False, **(extra or {}),
    )
    train = Pool(X, price_target(price, X, target), cat_features=cats, weight=weights)
    if X_val is not None and iterations is None:
        val = Pool(X_val, price_target(price_val, X_val, target), cat_features=cats)
        model.fit(train, eval_set=val, early_stopping_rounds=200, use_best_model=True)
    else:
        model.fit(train)
    return model


def predict_price(model, X: pd.DataFrame, target="log") -> np.ndarray:
    """(n, 3) array of p25/p50/p75 prices, sorted per row and never below the GTS floor."""
    raw = np.asarray(model.predict(X)).reshape(len(X), -1)
    if target == "log_ratio":
        raw = raw + X["log_floor"].to_numpy()[:, None]
    q = np.sort(np.exp(raw), axis=1)
    return np.maximum(q, floors_for(X)[:, None])


def widen(q: np.ndarray, scale: float, floors: np.ndarray) -> np.ndarray:
    """Stretch p25/p75 away from p50 (in log space) by `scale`; keep everything above the floor."""
    lq = np.log(q)
    out = np.column_stack([lq[:, 1] - scale * (lq[:, 1] - lq[:, 0]), lq[:, 1],
                           lq[:, 1] + scale * (lq[:, 2] - lq[:, 1])])
    return np.maximum(np.exp(out), floors[:, None])


def interval_scale(price, q, floors, target=0.5) -> float:
    """Smallest stretch that makes the p25-p75 range catch `target` of actual prices."""
    for s in np.arange(1.0, 5.01, 0.05):
        w = widen(q, s, floors)
        if ((price >= w[:, 0]) & (price <= w[:, 2])).mean() >= target:
            return float(round(s, 2))
    return 5.0


def price_metrics(price, q) -> dict:
    price = np.asarray(price, float)
    p25, p50, p75 = q[:, 0], q[:, 1], q[:, 2]
    ape = np.abs(p50 - price) / price
    log_err = np.abs(np.log(p50) - np.log(price))
    return {
        "n": int(len(price)),
        "median_abs_pct_error": float(np.median(ape)),
        "mean_abs_log_error": float(log_err.mean()),
        "typical_factor": float(np.exp(log_err.mean())),
        "within_25pct": float((ape <= 0.25).mean()),
        "within_50pct": float((ape <= 0.50).mean()),
        "p25_p75_coverage": float(((price >= p25) & (price <= p75)).mean()),
    }


def point_metrics(price, p50) -> dict:
    """Metrics for a single-number baseline (no range)."""
    p50 = np.asarray(p50, float)
    m = price_metrics(price, np.column_stack([p50, p50, p50]))
    m.pop("p25_p75_coverage")
    return m


# --------------------------------------------------------------------------- Model B: sell chance

def sell_features(X: pd.DataFrame, price: np.ndarray, p50: np.ndarray) -> pd.DataFrame:
    Xb = X.copy()
    Xb["log_price"] = np.log(price)
    Xb["price_gap"] = Xb["log_price"] - np.log(p50)
    return Xb


def fit_sell(Xb, y, weights=None, Xb_val=None, y_val=None, iterations=None, depth=6, verbose=False) -> CatBoostClassifier:
    cats = cat_features(Xb.columns)
    model = CatBoostClassifier(
        loss_function="Logloss", iterations=iterations or MAX_ITERS, learning_rate=0.06, depth=depth,
        l2_leaf_reg=3, random_seed=SEED, thread_count=-1, verbose=verbose, allow_writing_files=False,
        monotone_constraints={"log_price": -1, "price_gap": -1},
    )
    train = Pool(Xb, y, cat_features=cats, weight=weights)
    if Xb_val is not None and iterations is None:
        model.fit(train, eval_set=Pool(Xb_val, y_val, cat_features=cats), early_stopping_rounds=200,
                  use_best_model=True)
    else:
        model.fit(train)
    return model


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    t = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t.reset_index(drop=True)


def sell_metrics(y, p) -> dict:
    if np.ptp(p) == 0:
        ece = float(abs(p[0] - np.mean(y)))
    else:
        t = calibration_table(y, p)
        ece = float((t["n"] * (t["predicted"] - t["actual"]).abs()).sum() / t["n"].sum())
    return {
        "n": int(len(y)), "sold_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "calibration_error": ece,
    }
