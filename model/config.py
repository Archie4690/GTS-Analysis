"""Shared settings: where things live, how the data is split, model knobs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOOKUP_DIR = ROOT / "lookup"
ARTIFACT_DIR = ROOT / "model" / "artifacts"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

GTS_CONFIG = LOOKUP_DIR / "gts_config.json"
POKEDEX = LOOKUP_DIR / "pokedex.json"
POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"

# Local Postgres from db/compose.yml (trust auth, bound to localhost).
DSN = "host=127.0.0.1 port=5434 dbname=gts user=gts"

# Chronological 70/20/10 split, cut at these quantiles of sale time.
SPLIT_FRACTIONS = (0.70, 0.20, 0.10)

# Model B: "sold within this many hours". 24 h is the GTS default listing length.
SELL_HORIZON_HOURS = 24

# Sales completed this quickly after listing look like pre-arranged trades.
INSTANT_SALE_SECONDS = 60

# Model A quantiles (p25 / p50 / p75).
QUANTILES = (0.25, 0.5, 0.75)

SEED = 42

# Chosen by model/tune.py on the validation set (see docs/tuning.md and artifacts/tuning_log.json).
PRICE_HPARAMS = {"depth": 4, "learning_rate": 0.06, "l2_leaf_reg": 3, "extra": {"max_ctr_complexity": 1}}
ENSEMBLE_SEEDS = (0, 1, 2, 3, 4)          # the price model is the average of one model per seed
BASE_DROP_GROUPS = ("market_long",)       # feature groups that tuning showed don't help
SELL_DEPTH = 4
SELL_USE_DATE = False                     # the recent market sell rate replaces the date
