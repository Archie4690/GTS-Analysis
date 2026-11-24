# The data

## Where it comes from

The SmashMC Cobblemon server logs every GTS event to `pokemon_logger_events`. `db/export.sql`
exported the GTS rows on 18 Nov 2026, leaving out names, nicknames, original trainers and
descriptions. `db/load.sql` loads them into Postgres as `gts_events`, replacing player and Pokémon
UUIDs with salted hashes. The model only reads `gts_events`, never selects `seller_id`, and never
uses the raw CSV.

| | |
|---|---|
| Period | 2 Jun – 18 Nov 2026 |
| Listings (`selling`) | 34,396 |
| Sales (`sold`) | 12,777 |
| Distinct Pokémon | 18,889 |
| Species | 996, plus `egg_species` and 9 server-custom species |
| Shards | Celebi, CobblemonGold, HoOh, Lugia. The GTS is shared across shards, so shard isn't a feature. |

Each row records:
- what the Pokémon is: species, form, shiny, level, gender, ability, hidden ability, aspects, ball, held item, moves
- its six IVs and six EVs
- its nature
- the price

**Fields missing before 2 Jul 2026.** Nature, ability, gender, hidden ability, ball, held item,
aspects and moves were only logged from 2 Jul onwards. That's 8,929 rows, or 19%. The model treats
these fields as *unknown* for those rows, not as "none".

## How a listing ends

The GTS is fixed-price: a sale always goes through at the listed price (12,758 of 12,759 sales
match the previous listing exactly). Each listing is followed by one of three things:

- **a sale** at that price,
- **a relisting** of the same Pokémon, usually after the old listing expired or was withdrawn, and
  usually cheaper (7,695 of 13,155 relistings), or
- **nothing** in the data, meaning it expired, was withdrawn, or is still listed.

Listing length is chosen by the seller and isn't logged. The config allows 5 minutes to 72 hours,
with a default of 24 hours.
- 92% of sales happen within 24 h, and none took longer than 72 h.
- The sell-chance model predicts "sold within 24 h". Listings from the last 24 h of the data are
  left out because their outcome isn't known yet.

**Very fast sales.** 34% of sales complete within 60 seconds of listing, and a quarter within 38
seconds. They tend to be cheaper than the model expects, so they're probably under-priced listings
getting sniped, or trades agreed in advance (the seller lists it and the buyer is ready). They're
real sales, so they stay in the data. Dropping them was tested and didn't help (see [tuning.md](tuning.md)).

**Relisting is lopsided.** 12,275 Pokémon were listed once, while 185 were listed 10 or more times,
making 2,440 rows. Overall, the 10% of Pokémon listed 4 or more times account for 33% of listing rows.

## The GTS rules (`lookup/gts_config.json`)

**Minimum prices.** Each rule in the config's `globalModifiers` has:
- a minimum number of perfect (31) IVs
- optional requirements: shiny, hidden ability, legendary or Ultra Beast
- a value and a priority

Of the rules a Pokémon qualifies for, the one with the **highest priority sets its minimum price**.
The fees don't stack, and `basePrice` is 0. Some of the floors:

| Perfect IVs | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Normal | 500 | 750 | 1,000 | 2,500 | 5,000 | 25,000 | 50,000 |
| Shiny | 5,000 | 5,000 | 5,000 | 7,500 | 10,000 | 30,000 | 60,000 |
| Hidden ability | 2,500 | 3,000 | 3,500 | 5,000 | 15,000 | 50,000 | 100,000 |
| Legendary | 30,000 | 30,000 | 30,000 | 30,000 | 37,500 | 50,000 | 100,000 |
| Ultra Beast | 20,000 | 20,000 | 20,000 | 20,000 | 25,000 | 30,000 | 50,000 |
| Shiny + hidden-ability legendary | 120,000 | | | | 127,500 | 150,000 | 200,000 |

**How it was checked.** With "legendary" meaning Showdown's Sub-Legendary, Restricted Legendary or
Mythical tags, **none of the 12,777 sales or 34,326 listings with a known outcome is below its
computed floor.** Every non-shiny, non-HA group with 0–6 perfect IVs bottoms out exactly at its floor.

The server-custom species (Corvizard, Lugaidon, Arcernatus and others) sell for 250k–850k, but
Corvizard has sold for 500. So the server doesn't treat them as legendary, and they get the ordinary floors.

**Tax.** The GTS takes 2%, so the seller receives 98% of the price. `predict.py` shows this amount.

## What moves the price

**Server rules and rarity.** The GTS floor is the single most important feature. It encodes
legendary status, shiny, hidden ability and perfect IVs all at once. Legendaries come with 3 perfect
IVs, so "3+ perfect IVs" partly just means "legendary".

**Nature matters a lot, and it's judged on the evolved form.** Median prices of non-shiny sales:

| Line | Best natures | Worst |
|---|---|---|
| Charmander (→ Charizard, fast special attacker) | Timid 80k (69 sales) | Adamant 5k |
| Dratini (→ Dragonite, physical) | Adamant 120k (39) | Modest 8k |
| Froakie (→ Greninja, fast mixed attacker) | Timid 170k (27) | Lonely/Relaxed 5k |
| Gible (→ Garchomp, fast physical) | Jolly 50k (101) | Quiet 2k |
| Toxapex / Mareanie (a wall) | **Bold** 120k / 100k | Mild 4k |
| Ralts (→ Gardevoir *or* Gallade) | Modest 75k, Jolly 60k | Timid 7k |
| Ditto (breeding; passes its nature on with an Everstone) | Modest/Bold/Timid/Jolly 35–40k | Mild/Lonely/Rash 10k |

Part of each gap is because breeders put their best IVs on the nature they're breeding for. The
model has the IVs as separate features, so it can tell the two apart.

**Level only matters in combination with other things.** For breedable, non-shiny sales:

| Level | Sales | Median | Avg perfect IVs | Share with EVs |
|---|---|---|---|---|
| 1 | 2,312 | 40k | 2.8 | 0% |
| 2–20 | 1,075 | 10k | 0.7 | 7% |
| 21–50 | 1,570 | 10k | 1.0 | 31% |
| 51–99 | 2,753 | 30k | 2.0 | 35% |
| 100 | 1,217 | 10k | 1.0 | 43% |

- **Level 1 usually means bred**, and bred Pokémon have better IVs and natures.
- **Level 100 only adds value on a good Pokémon.** Charizard, Dragonite and Greninja at level 100
  sell for about 30k, against 10–15k at lower levels. Rayquaza sells for 275k at level 100 and 135k
  at level 1. A random wild-caught level 100 isn't worth more.

**Cosmetic aspects.** `galaxy` Pokémon (1,630 rows) sell at about twice the median, and `heavenly`,
`hero`, `mega` and `crowned` are expensive too. Aspects seen at least 30 times in training become
on/off features.

## Prices and demand change over time

- **Prices drift.** The median legendary sold for about 115k in Jun/Jul and about 70k in Oct/Nov.
- **Demand swings too.** The share of listings that sold within 24 h by month:

| Jun | Jul | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|
| 39% | 37% | 30% | 28% | 38% | 40% |

That's why the data is split by time and not shuffled, why recent sales are weighted more heavily,
and why the model uses recent-market features (see [model.md](model.md)).

## Is the data rich enough?

**Yes for a useful model, but how accurate it is depends on the species.**

| Sales per species | Species | Sales |
|---|---|---|
| 100+ | 12 | 2,133 |
| 30–99 | 86 | 4,071 |
| 10–29 | 253 | 4,223 |
| 3–9 | 392 | 2,077 |
| 1–2 | 188 | 273 |

- **Common species have plenty of history,** about half of all sales.
- **580 species have fewer than 10 sales.** For these the model leans on what it can work out from
  the Pokédex: base stats, legendary tags, role, and the evolution line.
- **Same-day sales of the same species vary by about ±29%** (median log-price standard deviation
  0.29). Prices are set by players, so some of that noise can't be predicted from the Pokémon at all.
- **About 10% of sales (1,305) are at exactly the GTS minimum,** usually sellers clearing space. In the validation period it was 15%.

The learning curve in [results.md](results.md) shows how much more data would help.
