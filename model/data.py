"""Load GTS events from Postgres and shape them into sales and listing-outcome tables.

Only the pseudonymised table is read; seller_id is never selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg

from config import DSN, SELL_HORIZON_HOURS, SPLIT_FRACTIONS

COLUMNS = [
    "id", "event_time", "action", "pokemon_id", "species", "form", "shiny", "level",
    "iv_hp", "iv_attack", "iv_defence", "iv_special_attack", "iv_special_defence", "iv_speed",
    "ev_hp", "ev_attack", "ev_defence", "ev_special_attack", "ev_special_defence", "ev_speed",
    "nature", "ability", "hidden_ability", "gender", "breedable", "held_item", "caught_ball",
    "aspects", "price",
]


def load_events() -> pd.DataFrame:
    sql = f"SELECT {', '.join(COLUMNS)} FROM gts_events ORDER BY pokemon_id, event_time, id"
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=COLUMNS)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    for c in ("nature", "ability", "gender", "held_item", "caught_ball", "aspects"):
        df[c] = df[c].fillna("")
    return df


def _link_events(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["pokemon_id", "event_time", "id"]).reset_index(drop=True)
    g = ev.groupby("pokemon_id", sort=False)
    ev["prev_action"] = g["action"].shift(1)
    ev["prev_time"] = g["event_time"].shift(1)
    ev["next_action"] = g["action"].shift(-1)
    ev["next_time"] = g["event_time"].shift(-1)
    return ev


def build_tables(ev: pd.DataFrame, horizon_hours: float = SELL_HORIZON_HOURS):
    """Return (sales, listings).

    sales:    one row per 'sold' event, with secs_to_sell (from its listing) where known.
    listings: one row per 'selling' event with a known outcome: sold_within (0/1) and
              n_listings (how many times that Pokémon was listed in total).
    """
    ev = _link_events(ev)
    data_end = ev["event_time"].max()

    sales = ev[ev["action"] == "sold"].copy()
    listed = sales["prev_action"] == "selling"
    sales["secs_to_sell"] = np.where(
        listed, (sales["event_time"] - sales["prev_time"]).dt.total_seconds(), np.nan)

    listings = ev[ev["action"] == "selling"].copy()
    horizon = pd.Timedelta(hours=horizon_hours)
    listings["sold_within"] = (
        (listings["next_action"] == "sold") & (listings["next_time"] - listings["event_time"] <= horizon)
    ).astype(int)
    listings["n_listings"] = listings.groupby("pokemon_id")["id"].transform("count")
    # Outcome unknown for listings made in the last `horizon` of the data.
    listings = listings[listings["event_time"] <= data_end - horizon].copy()

    drop = ["prev_action", "prev_time", "next_action", "next_time"]
    return (sales.drop(columns=drop).sort_values("event_time").reset_index(drop=True),
            listings.drop(columns=drop).sort_values("event_time").reset_index(drop=True))


def split_dates(sales: pd.DataFrame, fractions=SPLIT_FRACTIONS):
    """Cut dates at the 70th and 90th percentile of sale time (chronological 70/20/10)."""
    t = sales["event_time"].sort_values().reset_index(drop=True)
    n = len(t)
    c1 = t.iloc[int(round(n * fractions[0]))]
    c2 = t.iloc[int(round(n * (fractions[0] + fractions[1])))]
    return c1, c2


def assign_split(df: pd.DataFrame, cuts) -> pd.Series:
    c1, c2 = cuts
    return pd.Series(np.select([df["event_time"] < c1, df["event_time"] < c2], ["train", "val"], "test"),
                     index=df.index)
