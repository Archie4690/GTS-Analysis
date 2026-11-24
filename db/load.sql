-- Loads the GTS export into gts_events, swapping player and Pokémon UUIDs for
-- salted hashes. Raw UUIDs only ever sit in a temporary table, so nothing in
-- the database links back to a player. Re-running rebuilds the table.
--   docker exec gts-postgres psql -U gts -d gts -v salt="$(cat ~/.config/gts-salt)" -f /db/load.sql
\set ON_ERROR_STOP on

\if :{?salt}
\else
    \echo 'Missing salt: pass -v salt="$(cat ~/.config/gts-salt)"'
    \quit
\endif

BEGIN;

CREATE TEMP TABLE raw_events (
    event_time          TIMESTAMPTZ      NOT NULL,
    action              TEXT             NOT NULL,
    player_uuid         UUID             NOT NULL,
    pokemon_uuid        UUID             NOT NULL,
    species             TEXT             NOT NULL,
    form                TEXT             NOT NULL,
    shiny               BOOLEAN          NOT NULL,
    level               INTEGER          NOT NULL,
    iv_hp               INTEGER          NOT NULL,
    iv_attack           INTEGER          NOT NULL,
    iv_defence          INTEGER          NOT NULL,
    iv_special_attack   INTEGER          NOT NULL,
    iv_special_defence  INTEGER          NOT NULL,
    iv_speed            INTEGER          NOT NULL,
    ev_hp               INTEGER          NOT NULL,
    ev_attack           INTEGER          NOT NULL,
    ev_defence          INTEGER          NOT NULL,
    ev_special_attack   INTEGER          NOT NULL,
    ev_special_defence  INTEGER          NOT NULL,
    ev_speed            INTEGER          NOT NULL,
    nature              TEXT,            -- NULL before 2 Jul 2026: the logger didn't record it yet
    ability             TEXT,
    hidden_ability      BOOLEAN          NOT NULL,
    gender              TEXT,
    breedable           BOOLEAN          NOT NULL,
    held_item           TEXT,
    caught_ball         TEXT,
    size                DOUBLE PRECISION NOT NULL,
    aspects             TEXT,
    forced_aspects      TEXT,
    form_aspects        TEXT,
    moves               TEXT,            -- newline-separated
    shard_name          TEXT             NOT NULL,
    price               BIGINT           NOT NULL
) ON COMMIT DROP;

COPY raw_events FROM '/data/raw/gts_events.csv' (FORMAT csv, HEADER);

DROP TABLE IF EXISTS gts_events;
CREATE TABLE gts_events AS
SELECT event_time,
       action,
       MD5(:'salt' || player_uuid::TEXT)  AS seller_id,
       MD5(:'salt' || pokemon_uuid::TEXT) AS pokemon_id,
       species, form, shiny, level,
       iv_hp, iv_attack, iv_defence, iv_special_attack, iv_special_defence, iv_speed,
       ev_hp, ev_attack, ev_defence, ev_special_attack, ev_special_defence, ev_speed,
       nature, ability, hidden_ability, gender, breedable, held_item, caught_ball, size,
       aspects, forced_aspects, form_aspects, moves, shard_name, price
FROM raw_events
ORDER BY event_time;

ALTER TABLE gts_events ADD COLUMN id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY;
CREATE INDEX ON gts_events (pokemon_id, event_time);
CREATE INDEX ON gts_events (action, event_time);
CREATE INDEX ON gts_events (species);

COMMIT;

ANALYZE gts_events;

SELECT action, COUNT(*) AS events, MIN(event_time)::DATE AS first, MAX(event_time)::DATE AS last
FROM gts_events
GROUP BY action;
