# Tuning log

This document records how the model was tuned: what was tried, why, what it scored, and what was
kept. The raw numbers for every attempt are in `model/artifacts/tuning_log.json`, which is appended
by `model/tune.py`.

## The rules

1. **Decisions use the validation set only.** That's the 20% of sales from 9 Sep to 21 Oct 2026.
   The model trains on the earlier 70% and is scored on this 20%.
2. **The test set (the last 10%, 21 Oct to 18 Nov) is a final exam, not a tuning tool.** Each time
   you look at it and change something because of what you saw, it becomes less honest.
   - It was looked at once after the first version (v1), and that led to two changes, described below.
   - An intermediate retrain (v2) was stopped before its test score was read, because tuning was going to continue.
   - It was looked at once more for the final version.
   - The v1 and final results are both reported in [results.md](results.md).
3. **One idea per candidate, written down as a hypothesis before it runs.** That includes the ideas
   that didn't work, so nobody tries them again by accident.
4. **The main score** is the mean absolute log error of the p50 price. Lower is better.
   - `exp(error)` is the typical factor the prediction is off by: 0.49 means about ×1.63.
   - A log scale treats a 10k Pokémon priced at 20k the same as a 100k one priced at 200k, which
     matches how wrong it feels.
   - The sell-chance model is scored on validation log loss, with AUC and calibration error shown alongside.
5. **Small differences are noise, and this was measured (round 4).**
   - The same model trained with five different random seeds scored between 0.4854 and 0.4881 on
     validation, a spread of 0.0027.
   - A single run can't separate changes smaller than about 0.003. From round 4 on, close calls are
     settled by averaging over 3 seeds.
   - A tie goes to the simpler option.

## How to repeat it

```bash
.venv/bin/python model/tune.py                 # every round
.venv/bin/python model/tune.py hparams sell    # selected rounds
```

Each round is a function in `model/tune.py` (`round_market`, `round_hparams` and so on). Adding an
idea means adding a round with its hypothesis, then running it. Once a winner is clear, copy it
into `model/config.py` (`PRICE_HPARAMS`) or `model/features.py`, and retrain with `train.py`.

## Round 0: first version (v1)

`train.py` searched 12 combinations of three choices:
- the target: log(price) or log(price / GTS floor)
- recency weighting: none, or a half-life of 60 or 120 days
- whether to drop sales completed within 60 s of listing

It also ran the nature/role ablation.

| Choice | Best on validation | Notes |
|---|---|---|
| Target | log(price / floor) | The model predicts how far above the floor a Pokémon sells. |
| Recency half-life | 120 days | Older sales count for half as much every 120 days. |
| Drop sales within 60 s | No | Dropping them didn't help, even though 34% of sales happen that fast. |
| Nature features | Raw nature + role rule | The learned `line_nature` premiums added nothing on top: 0.4898 vs 0.4894, a tie. |

**Validation error: 0.4898.**

**What looking at the test set showed (once):**
- **The likely range was too narrow.** Only 32% of test prices fell inside p25–p75, against a target of 50%.
- **The sell-chance percentages were too low.** The share of listings selling within 24 h swings by
  month: 39% in Jun, 37% Jul, 30% Aug, 28% Sep, 38% Oct, 40% Nov. The model had learned the slow
  Aug/Sep market through the date feature.

**Changes made because of it:**
- The p25/p75 range is now stretched by a factor chosen on the validation set (`interval_scale`), so
  it catches 50% of validation prices.
- The sell-chance model gets a choice, decided on validation, of whether to use the date.

## Error analysis (validation set, v1)

Before adding more features, I checked where the error comes from:

| Group | Sales | Mean abs log error | Bias (+ = predicted too high) |
|---|---|---|---|
| Sold at exactly the GTS minimum | 375 (15%) | 0.645 | +0.645 |
| Everything else | 2,180 | 0.461 | −0.094 |
| Sold within 60 s of listing | 986 (39%) | 0.513 | +0.178 |
| Sold later | 1,569 | 0.473 | −0.089 |
| Legendary | 752 | 0.381 | +0.039 |
| Not legendary | 1,803 | 0.533 | +0.004 |

- **13% of sales account for 40% of the error.** These are the sales that miss by more than ×2.7, and they're mostly:
  - **Floor dumps.** Corvizard, Electivire, Dragonite and Gyarados sold at the 500/750 minimum.
    The seller was clearing space, which nothing about the Pokémon can reveal.
  - **Rare forms with almost no history.** Eternal Floette sold for 150–200k from September onwards,
    but the training period had just one sale of it (at 850k). That's too little for the model to trust.
  - **One-off prices.** A Ditto at 1.1M, and a 300k egg.
- **Same-day sales of the same species already vary by about ±29%** (median log-price standard
  deviation 0.29). That's a rough floor on how accurate any model can be with this data.
- **Instant sales tend to be cheap.** They're under-priced listings that got sniped, or trades
  agreed in advance. That's why removing them didn't help: they're real sales, just at a discount.

**What this means:** most of the remaining error is in how players set prices, not in what the
model knows about the Pokémon. Further gains will be small.

## Round 1: recent-market features

**Hypothesis:** prices drift. The recent average price of the same species and evolution line,
plus a market-wide price level, should follow that drift better than the raw date.

The features (`MarketHistory` in `features.py`) only look backwards. For a sale at time t, they use
sales strictly before t:
- **`species_recent_logp`, `line_recent_logp`:** the mean log price over the previous 30 days, with counts
- **`market_recent_ratio`:** the mean log(price / floor) across all sales in the previous 14 days
- **`market_sell_rate`:** the share of listings made 8 to 1 days earlier that sold within 24 h
  (listings from the last day don't have a known outcome yet)

| Candidate | Validation error | Median % error |
|---|---|---|
| v1 (no market features) | 0.4898 | 33.2% |
| **+ market features** | **0.4882** | 33.2% |
| + market features, without the date | 0.4908 | 33.0% |

**Decision:** keep the market features, and keep the date for the price model. The gain is small
but consistent. 85% of validation sales had a recent same-species price available, and 92% had a
recent same-line price.

## Round 2: hyperparameters

**Hypothesis:** tree depth, learning rate and regularisation were left at their defaults (depth 6,
learning rate 0.06, L2 3). Nearby values might fit better.

| Candidate | Validation error | Median % error |
|---|---|---|
| Default (depth 6) | 0.4882 | 33.2% |
| **Depth 4** | **0.4835** | **32.1%** |
| Depth 8 | 0.4886 | 33.7% |
| L2 1 | 0.4872 | 33.0% |
| L2 10 | 0.4915 | 33.7% |
| Learning rate 0.03 | 0.4856 | 32.7% |

Round 2b checked the neighbours of the two winners:

| Candidate | Validation error |
|---|---|
| Depth 3 | 0.4867 |
| **Depth 4** | **0.4835** |
| Depth 5 | 0.4871 |
| Depth 4 + learning rate 0.03 | 0.4852 |
| Depth 4 + L2 1 | 0.4896 |

**Decision:** depth 4, keeping the default learning rate and L2. Shallower trees generalise better
here, which makes sense with 9k noisy sales spread over 996 species. Deeper trees memorise one-off prices.

## Round 3: sell-chance model

**Hypothesis:** the sell rate swings by month. A recent sell-rate feature should follow it, while the
date ties the model to whichever months it was trained on.

| Candidate | Val log loss | AUC | Brier | Calibration error |
|---|---|---|---|---|
| With date, no market sell rate | 0.5171 | 0.785 | 0.1714 | 2.3 pts |
| No date, no market sell rate | 0.5162 | 0.784 | 0.1709 | 1.7 pts |
| With date + market sell rate | 0.5155 | 0.787 | 0.1708 | 2.4 pts |
| **No date + market sell rate** | **0.5121** | **0.789** | **0.1695** | 2.2 pts |

**Decision:** drop the date from the sell-chance model and use the recent market sell rate.
`train.py` re-checks this on validation every time it runs.

## Round 4: how big is the noise? (seeds)

**Hypothesis:** the gains so far are 0.002–0.006. Before believing them, measure how much the score
moves from the random seed alone.

| Seed | 0 | 1 | 2 | 3 | 4 | (42, the default) |
|---|---|---|---|---|---|---|
| Validation error (depth 4 + market) | 0.4856 | 0.4881 | 0.4855 | 0.4854 | 0.4875 | 0.4835 |

**Finding:** the spread is about 0.003. The 0.4835 from rounds 1–2 was partly a lucky seed. Single
runs can't separate the earlier candidates, so the main decisions were re-scored as 3-seed averages:

| Candidate (mean of seeds 0, 1, 2) | Validation error | Seed spread |
|---|---|---|
| v1: depth 6, no market features | 0.4922 | |
| depth 6 + market features | 0.4894 | |
| **depth 4 + market features** | **0.4864** | |
| depth 4 + market, half-life 240 | 0.4881 | |

**Decision:** both kept changes hold up: market features gain about 0.003 and depth 4 another
0.003, each above the noise. Half-life 240 isn't better than 120.

## Round 5: more ideas

| Candidate | Validation error | Decision |
|---|---|---|
| **All-time price memory** (whole sales history of the species/line, not just 30 days) | 0.4864, vs 0.4835 on the same seed | **Dropped:** worse. The ordered category statistics already capture long-run species prices. |
| **Ensemble:** average of 5 models with seeds 0–4 | **0.4845**, vs a mean of 0.4864 for the single seeds | **Kept:** averaging removes much of the seed luck and gives a steadier model. |
| Target/half-life re-check at depth 4 (8 runs) | Best: log(price/floor) with half-life 120 (0.4835) or 240 (0.4828) | **No change:** 240 wasn't better on the 3-seed average (round 4). |
| Sell-chance model depth 4 / 6 / 8 | Log loss 0.5121 / 0.5137 / 0.5132 | **Depth 4** |

## Round 6: CatBoost structure (3-seed means)

**Hypothesis:** CatBoost combines categorical columns automatically (up to 4 at a time), grows
symmetric trees, and one-hot encodes categories with 2 or fewer values. Other settings might suit
small, noisy data better.

| Candidate | Validation error |
|---|---|
| Baseline (depth 4 + market) | 0.4864 |
| **max_ctr_complexity 1** (no automatic category combinations) | **0.4848** |
| max_ctr_complexity 2 | 0.4861 |
| Lossguide trees, 16 leaves | 0.4860 |
| one_hot_max_size 10 | 0.4858 |

**Decision:** max_ctr_complexity 1. The gain is at the edge of the noise, but it's also the simpler
and faster model, which settles a tie. Everything else is a tie.

## Round 7: re-checking the basics with the final settings (in `train.py`)

`train.py` re-runs the 12-way search over target × recency × fast-sale handling every time,
using the current settings. With depth 4, max_ctr_complexity 1 and the market features, the answer
changed:

| Target | Half-life | Drop sales < 60 s | Validation error |
|---|---|---|---|
| **log(price)** | **120** | **yes** | **0.4788** |
| log(price) | none | yes | 0.4823 |
| log(price) | 60 | yes | 0.4839 |
| log(price/floor) | 60 | no | 0.4854 |
| log(price/floor) | 120 | yes | 0.4870 |
| … | | | |
| log(price/floor) | 120 | no (the earlier choice) | 0.4927 |

**Finding:** dropping fast sales from training now helps consistently. All six "drop" configurations
are in the top seven, and the gap to the earlier choice (0.014) is about five times the seed noise.
They're mostly sniped under-priced listings and arranged trades. With automatic category combinations
switched off, the model is more easily pulled down by them, so it does better learning from normal
sales only.

**Decision:** take what `train.py` picked: log(price), half-life 120, fast sales dropped from
training. They're still included when scoring. This was decided on validation before the final
test-set look.

The nature ablation was also re-run with the final settings. This time the full feature set,
including the learned `line_nature` favourites, came out best (0.4788). In v1 the role rule alone
was best. Both differences are small.

## When to stop

Rounds 5 and 6 found nothing that beat the current model by more than the noise, apart from the
ensemble, which mostly removes noise. On top of that, the error analysis shows most of the remaining
error comes from floor dumps, sniped listings and one-off prices. So tuning has converged: more
changes to this model on this data won't make much difference.

## Overall (validation set)

| Version | Validation error | Typical factor | Notes |
|---|---|---|---|
| v1 (depth 6, no market features) | 0.4922 | ×1.64 | 3-seed mean |
| + market features | 0.4894 | ×1.63 | 3-seed mean |
| + depth 4 | 0.4864 | ×1.63 | 3-seed mean |
| + max_ctr_complexity 1 | 0.4848 | ×1.62 | 3-seed mean |
| + log(price) target, fast sales dropped from training (round 7, single seed) | 0.4788 | ×1.61 | |
| **Final: 5-seed ensemble (seeds 0–4)** | **0.484** | ×1.62 | The 0.4788 above was one run with the default seed 42. Like round 4's 0.4835, it was partly luck; the ensemble is the honest figure. |

The final settings live in `model/config.py`:
- `PRICE_HPARAMS`, `ENSEMBLE_SEEDS`, `BASE_DROP_GROUPS`, `SELL_DEPTH`, `SELL_USE_DATE`
- the target (log(price)), recency weighting (120 days) and dropping fast sales from training,
  which `train.py` re-confirms on validation every run

The final test-set results are in [results.md](results.md).

## Ideas not tried yet

These are listed so they're picked up next time:
- **Predicting floor dumps separately.** About 10% of all sales (15% in the validation period) are dumps at the minimum. A small
  model for "will this be dumped?" could stop it pulling down the price for normal sales. It would
  only help with the sell-chance side, though, since sellers choose to dump.
- **Moves.** Egg moves and Hidden Power types may matter for some species. They were left out of
  v1 so as not to add hundreds of sparse columns (see [model.md](model.md), "Is more data always better?").
- **Retraining regularly.** The market features use the last 30 days. Retraining weekly keeps the
  snapshot used by `predict.py` fresh, and that's probably worth more than any further tuning.
