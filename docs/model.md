# How the model works

There are two models, both gradient-boosted decision trees (CatBoost):

| | Question | Trained on | Output |
|---|---|---|---|
| **Model A** (price) | What will this Pokémon sell for? | 12,777 sales | p25 / p50 / p75 price, never below the GTS floor |
| **Model B** (sell chance) | If I list it at price X, will it sell within 24 h? | 34,326 listings with a known outcome | A probability |

`predict.py` uses A to estimate the price, then asks B about a ladder of round prices to suggest
the highest one likely to sell.

```
Pokémon ──► features ──► Model A ──► p50 price (+ range) ──┐
                │                                         ▼
                └──────────────► Model B ◄── asking price, price ÷ p50
                                    │
                                    ▼
                    chance of selling within 24 h at each price
```

CatBoost suits this data for three reasons:
- **Text categories with many values.** It handles species (996), ability and so on directly, using
  "ordered target statistics": each row's category is encoded using only the rows before it, so a
  Pokémon's own price never leaks into its encoding.
- **Rare categories.** It shrinks them towards the overall average, which matters when 580 species
  have fewer than 10 sales.
- **Noisy data.** It's robust on small, noisy tabular data.

## Features

All features are built by `model/features.py`, the same code for training and prediction, so they
can't drift apart.

| Group | Features | Why |
|---|---|---|
| Identity | species, form, shiny, evolution line, final evolution | What it is |
| Details | ability, hidden ability, gender, breedable, held item, ball | Logged from 2 Jul. Before that, *unknown* |
| Aspects | an on/off flag for each cosmetic aspect seen at least 30 times (galaxy, anime, bronze, …) | Cosmetic variants sell at a premium |
| Level & EVs | level, is level 1 (bred), is level 100 (ready to battle), EV total, fully trained | See [data.md](data.md) for what level means |
| Floor | log of the GTS minimum price from `lookup/gts_config.json` | Encodes the server's own view of rarity |
| IVs | the six IVs, their sum, number of 31s, number of 0s | |
| Pokédex | base stats and total, types, egg group, legendary/mythical/restricted/Ultra Beast/paradox, fully evolved | Lets rare species borrow from similar ones |
| Nature | the nature, the stat it raises, the stat it lowers | |
| **Role** | role on three axes, `nature_fit`, role-weighted IV and EV scores | The "ideal IVs and nature" logic, explained next |
| Market memory | the line's best-selling natures (`line_nature`), nature popularity | Market favourites, like Timid Charmander |
| Recent market | the species' and line's mean log price over the last 30 days, the market price level over the last 14 days, and the share of listings that sold over the last week | Follows the market as it drifts |
| Time | days since 2 Jun | The price model only. Trees can't extrapolate, so for new dates it acts like "the most recent period". |

## "Ideal IVs and nature": the role logic

A nature raises one stat by 10% and lowers another. Whether that helps depends on what the Pokémon
is for, so the model works out each Pokémon's role from its **fully evolved** base stats
(`lookup.py`). The market prices a Gible as a future Garchomp, not as a Gible.

**1. Find the final evolution.**
- Follow Showdown's evolution data to the end of the line: Gible → Gabite → Garchomp.
- **More than one final form** (Ralts → Gardevoir or Gallade; Eevee → 8 forms): each one is scored,
  and the one that suits the Pokémon's nature best is used. A Modest Ralts is a good Gardevoir, and
  a Jolly Ralts is a good Gallade.
- **Regional "bias" forms** (Goomy `hisui-bias`) follow the regional evolution: Hisuian Goodra.
- **Ditto is a breeding Pokémon.** It's scored against six stand-in roles (fast or slow physical,
  fast or slow special, physical or special wall), so any nature that suits *some* role is good,
  and all six IVs count.

**2. Place it on three axes**, using base stats:

| Axis | Rule |
|---|---|
| Attack | **physical** if Atk ≥ 1.2 × SpA, **special** if SpA ≥ 1.2 × Atk, otherwise **mixed** |
| Speed | **fast** if Spe ≥ 80, **slow** if Spe ≤ 50 (a Trick Room candidate), otherwise **medium** |
| Bulk | a **wall** if mean(HP, Def, SpD) ≥ 1.1 × its best attacking stat |

The raw ratios behind these rules are also features, so the trees can learn their own cut-offs.

**3. Weight each stat by how much the role uses it:**

| Stat | Attacker | Wall |
|---|---|---|
| The attacking stat it uses | 1 | 0.4 (0.1 each if mixed) |
| The other attacking stat | 0 (0.5 if mixed) | 0 |
| Def, SpD | 0.3 | 1 |
| Spe | fast 1 / medium 0.6 / slow 0 | fast 0.6 / medium 0.3 / slow 0 |
| HP (IVs and EVs only) | 0.7 | 1 |

**4. Score the nature.** `nature_fit` = weight of the raised stat − weight of the lowered stat. It
runs from −1 to 1, and the five neutral natures score 0.

| Pokémon | Role | Good | Bad |
|---|---|---|---|
| Garchomp (Gible) | fast physical | Jolly +1, Adamant +1 | Modest −1, Quiet −1 |
| Charizard (Charmander) | fast special | Timid +1, Modest +1 | Adamant −1, Brave −1 |
| Toxapex (Mareanie) | slow wall, mixed attacks | Bold, Impish, Calm, Careful +0.9 | Mild, Lonely −0.9 |
| Skarmory | physical wall | Impish +1 | Mild −1 |
| Conkeldurr | slow physical | Adamant +1, **Brave +1** (Trick Room, not penalised) | Timid −1 |
| Ralts | Gardevoir *or* Gallade | Modest, Timid, Jolly, Adamant +1 | |

**5. Score the IVs and EVs with the same weights:**
- a weighted IV score
- the number of 31s in the stats that matter (weight ≥ 0.5)
- **useful zeros:**
  - a 0–1 Atk IV on a special attacker or wall (less damage from Foul Play and confusion)
  - a 0–1 Spe IV on a slow Pokémon (Trick Room)
- `ev_useful_share`: the share of its EVs that went into stats that matter. 252 Atk / 252 Spe on
  Garchomp scores about 1; random EVs from wild battles score low.

**6. Add what the market has learned.** The rule can't know that buyers pay far more for Timid
Charizard than for Modest, even though both suit it. `line_nature` (e.g. `charmander|timid`) lets
CatBoost learn each line's favourites from sales. Rare combinations are shrunk towards zero, so the
rule takes over for species with few sales.

**Did it help?** The ablation in [results.md](results.md) answers that with numbers. In the final model:
- the raw nature label on its own adds almost nothing
- the role rule and the learned favourites each cut validation error by about 0.01 (removing either
  one costs about as much as removing all the raw IVs)
- together they're the reason nature counts at all

## Model A: price

- **Target: log(price).** Working in log space means a 2× miss costs the same at 10k as at 1M.
  `train.py` re-checks this choice against log(price / floor) on validation every run. With the
  final settings, log(price) won.
- **Training rows: sales completed within 60 seconds of listing are left out.** Those are mostly
  sniped under-priced listings and arranged trades, and they drag estimates down. They still count
  when the model is scored. This was also chosen on validation (see [tuning.md](tuning.md), round 7).
- **Output: three quantiles (p25, p50, p75)** from CatBoost's `MultiQuantile` loss. p50 is the
  estimate, and p25–p75 is the "likely range".
- **The range is widened by a factor chosen on the validation set** so that it catches about half of
  actual prices. Without this, the range was too narrow, because prices drift between training and
  later periods.
- **Recency weighting:** older sales count for half as much every 120 days.
- **Floor:** predictions are clipped at the GTS minimum, because the GTS won't accept a lower price.
- **Settings:** depth-4 trees, learning rate 0.06, no automatic category combinations, early stopping
  on validation. The final model is an **average of 5 models trained with different seeds**. See
  [tuning.md](tuning.md) for how each setting was chosen.

## Model B: sell chance

- **One row per listing.** The label is 1 if the next event for that Pokémon is a sale within 24 h,
  and 0 if it sold later, was relisted, or has no later event.
- **Features:** everything Model A uses, minus the date, plus:
  - **log(asking price)**
  - **price_gap** = log(asking price) − log(Model A's p50): how far above or below fair value it's listed
  - **the recent market sell rate**: the share of listings that sold over the last week, so the model
    follows busy and quiet periods
- **No leakage through price_gap.** A training listing whose Pokémon later sold must not have its
  price_gap computed by a Model A that saw that sale. So, for training listings, Model A is refit 5
  times, each time holding out a fifth of Pokémon (grouped by `pokemon_id`), and every listing gets
  its estimate from the model that didn't see it.
- **Monotone constraint:** the chance of selling can only fall as the asking price rises.
- **Listing weights:** Pokémon relisted many times would dominate. The weighting (none, 1/√n or 1/n)
  is chosen on validation.
- If validation shows the probabilities are off, an isotonic recalibration is applied.

## Splitting the data: 70 / 20 / 10 by time

Sales are sorted by time and cut at the 70th and 90th percentiles:

| Split | Sales | Period | Used for |
|---|---|---|---|
| Train | 8,944 | 2 Jun – 9 Sep | Fitting |
| Validation | 2,555 | 9 Sep – 21 Oct | Every tuning decision, early stopping, range width |
| Test | 1,278 | 21 Oct – 18 Nov | The final score only |

Listings use the same two dates.

**Why not shuffle?** Prices drift. A shuffled split lets the model learn from October prices and
then be tested on September, which flatters it. It's also not how the model gets used: it always
predicts the future from the past. `train.py` reports the shuffled-split score too, so the gap is visible.

After testing, both models are **refit on all the data** before saving, so `predict.py` uses
everything up to 18 Nov.

## Is more data always better?

More rows of **accurate, relevant** data almost never make a model like this worse; each extra row
just helps less than the last. What hurts is the wrong kind of data. Each case was checked:

1. **Amount.** `experiments.py` retrains on 10–100% of the training sales. The learning curve in
   [results.md](results.md) shows whether more months would help.
2. **Age.** Old prices can drag predictions towards an old market: June legendaries sold for about
   115k, against about 70k in November. `experiments.py` compares training on only the last 30, 60
   or 90 days with training on everything. Recency weighting and the 30-day market features are the
   gentler fix: they keep old data but trust recent data more.
3. **Noisy rows.**
   - 34% of sales happen within 60 seconds (snipes or arranged trades), and 10% are dumps at the floor.
   - Dropping the fast sales from *training* didn't help at first. Once the model was simpler (no
     automatic category combinations), it clearly did, so the final model trains without them.
     That's a case where less data really is better.
   - 144 "troll" listings at 10M or more never sell. They're kept as genuine unsold listings.
4. **Too many columns.** This is the main way "more data" hurts here. Every weak feature gives the
   model another chance to fit coincidences in 12.8k noisy sales. Each feature group has to lower
   the validation error to stay:
   - all-time price memory was dropped
   - moves and shard were never added
   - Smogon usage was left out on purpose. It reflects the Showdown competitive meta, not what
     players on this server pay. It may be worth a test later as one more feature group.
5. **Leakage.** Nothing that isn't known at listing time is used: not time-to-sell, not the buyer,
   not later prices. The market features look strictly backwards, and Model B's training rows use
   out-of-fold price estimates.

## Limitations

- **Sellers choose the price.** A Pokémon dumped at the floor or sold to a friend can't be predicted
  from its stats. That's a ceiling on accuracy, not a bug.
- **New or rare forms and cosmetics** with little history are priced poorly until more sales build
  up. Eternal Floette had one training sale, then sold for 150–200k in the validation period, where
  the model predicted under 10k.
- **The listing timer isn't logged.** A listing set to expire after 5 minutes counts as "didn't sell
  within 24 h".
- **The market snapshot goes stale.** `predict.py` uses the recent-market values from the end of the
  training data, so retrain when new data is exported.
