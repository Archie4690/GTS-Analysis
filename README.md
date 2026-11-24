# GTS Analysis

Price prediction for the SmashMC Cobblemon GTS. Given a Pokémon about to be listed (species, IVs,
nature, level and so on), the model estimates:

1. **what it will sell for**: a likely price and a p25–p75 range, never below the GTS minimum, and
2. **the chance it sells within 24 hours** at any asking price. That's the GTS's default listing
   length, and the chance is used to suggest a price.

The write-ups explain how and why it works:

| Doc | What's in it |
|---|---|
| [docs/data.md](docs/data.md) | The dataset, the GTS rules (floors, timer, tax), what the data shows about nature and level, and whether it's rich enough |
| [docs/model.md](docs/model.md) | The features (including the role-based "ideal IV/nature" logic), the two models, the 70/20/10 time split, and whether more data is always better |
| [docs/tuning.md](docs/tuning.md) | How the model was tuned: every idea tried, its validation score, and what was kept |
| [docs/results.md](docs/results.md) | Test-set results, baselines, ablations, the learning curve and charts |

**Results in one line:** on the most recent month of sales (never seen in training), the median
price error is about 38%, against 58% for "what this species usually sells for", and the sell-chance
model ranks the listing that sold above the one that didn't 77% of the time. Most of the remaining
error is noise in player-set prices. See [docs/results.md](docs/results.md).

## Quick start (no database needed)

The trained models are in `model/artifacts/`, so you can price Pokémon straight after cloning:

```bash
python3 -m venv .venv
.venv/bin/pip install -r model/requirements.txt
.venv/bin/python model/fetch_lookup.py        # downloads Showdown's pokedex.json (once)
.venv/bin/python model/predict.py --species gible --ivs 31,31,31,0,31,31 --nature jolly
```

## Data and privacy

The training data is a GTS log export from the SmashMC server. It is **not included** in this repo.
- The raw export holds player and Pokémon UUIDs. It stays in `raw/`, which is git-ignored.
- `db/load.sql` loads it into Postgres with the UUIDs replaced by salted hashes. The salt is kept
  outside the repo.
- The model reads only that pseudonymised table, and never uses the seller as a feature.

What *is* included contains only aggregate statistics: the trained models, `meta.json` and the
metrics. That means you can run `predict.py`, but retraining (`train.py`, `tune.py`,
`experiments.py`) needs access to the database.

## Layout

```
db/            Postgres setup: compose.yml, load.sql (CSV -> pseudonymised table), export.sql
raw/           the raw export (git-ignored: contains raw UUIDs; the model never reads it)
lookup/        reference data: gts_config.json (GTS mod config), pokedex.json (Showdown)
model/         the Python code (see below); artifacts/ holds trained models and metrics
docs/          write-ups and figures
```

| File | Purpose |
|---|---|
| `model/config.py` | Database connection, paths, split fractions, 24 h horizon |
| `model/fetch_lookup.py` | Downloads Showdown's `pokedex.json` into `lookup/` |
| `model/lookup.py` | Pokédex lookups, evolution lines, roles, the nature table, GTS floor rules |
| `model/data.py` | Reads `gts_events` from Postgres; builds the sales and listing-outcome tables; time split |
| `model/features.py` | Feature engineering shared by training and prediction |
| `model/modelling.py` | CatBoost fitting, prediction and metrics helpers |
| `model/train.py` | Tuning, evaluation on the 70/20/10 split, final refit, figures |
| `model/experiments.py` | Learning curve, data-age windows, feature-group ablations |
| `model/tune.py` | Tuning harness: named ideas scored on validation, logged to `artifacts/tuning_log.json` |
| `model/predict.py` | Command-line pricing tool |
| `model/plots.py` | Chart styling for `docs/figures/` |

## Setup

1. **Database.** Start Postgres and load the export, as described in `db/load.sql`:
   ```bash
   docker compose -f db/compose.yml up -d
   docker exec gts-postgres psql -U gts -d gts -v salt="$(cat ~/.config/gts-salt)" -f /db/load.sql
   ```
   The model reads only the pseudonymised `gts_events` table on `127.0.0.1:5434`, never
   `raw/gts_events.csv`.

2. **Python environment** (Python 3.13):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r model/requirements.txt
   ```

3. **Reference data.** `lookup/gts_config.json` is the GTS mod config and is already in the repo.
   Download the Pokédex once:
   ```bash
   .venv/bin/python model/fetch_lookup.py
   ```

## Train

```bash
.venv/bin/python model/train.py         # about 5 minutes on 4 cores
.venv/bin/python model/experiments.py   # about 2 minutes; needs train.py's metrics.json
```

`train.py` writes `model/artifacts/{price_0..4.cbm, sell.cbm, meta.json, metrics.json}` and the charts
in `docs/figures/`. The price model is an average of five models trained with different seeds.

**Retrain regularly, ideally weekly.** The model uses the last 30 days of prices for each species
and the recent market sell rate. `predict.py` takes those from a snapshot made when the model was
trained, so the longer you leave it, the less current the snapshot is. Retraining whenever new GTS
data is loaded keeps it fresh, and so does retraining after the GTS config changes.

To try a new idea, add a round to `model/tune.py` (see [docs/tuning.md](docs/tuning.md)), run
`.venv/bin/python model/tune.py <round>`, and copy any winner into `model/config.py`.

## Predict

```bash
.venv/bin/python model/predict.py --species gible --ivs 31,31,31,0,31,31 --nature jolly
.venv/bin/python model/predict.py --species rayquaza --ivs 31,31,31,20,31,31 --nature adamant \
    --level 100 --evs 0,252,4,0,0,252 --price 250000
```

IVs and EVs are given in the order **HP, Atk, Def, SpA, SpD, Spe**. Only `--species`, `--ivs` and
`--nature` are required. The other options are:

| Option | Meaning |
|---|---|
| `--level` (default 1), `--evs` (default all 0) | Level and EVs |
| `--form` | e.g. `hisui`, `galarian`, `alola`, `therian` |
| `--shiny`, `--hidden-ability`, `--not-breedable` | Flags |
| `--ability`, `--gender`, `--ball` | Default to the most common value for that species |
| `--aspects` | Cosmetic aspects, e.g. `galaxy` |
| `--held-item` | Held item |
| `--price` | Check one asking price |
| `--target` | The chance of selling the suggested price should reach (default 0.6) |

The tool prints:
- the GTS minimum price
- the Pokémon's role (worked out from its final evolution) and how well the nature suits it
- the best-selling natures for its evolution line
- the estimated sale price and range
- a table of asking prices, each with its chance of selling within 24 h and what you'd receive after the 2% tax
- a suggested price

## Licence

[MIT](LICENSE). The Pokédex data downloaded by `fetch_lookup.py` comes from
[Pokémon Showdown](https://github.com/smogon/pokemon-showdown) and isn't redistributed here.
Pokémon is © Nintendo / Game Freak / The Pokémon Company. This is an unofficial fan project.
