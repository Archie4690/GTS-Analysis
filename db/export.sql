-- The export run on the live SmashMC database (18 Nov 2026) to make ../raw/gts_events.csv.
-- Read-only. Run at the psql prompt on the Cobblemon box as one line (\copy can't span lines),
-- then scp the file to data/raw/ and delete the server copy.
-- Leaves out names, nicknames, original trainer, description (contains usernames) and the
-- raw Pokémon blob. UUIDs are hashed on the local load (load.sql).
\copy (SELECT event_time, action, player_uuid, pokemon_uuid, species, form, shiny, level, iv_hp, iv_attack, iv_defence, iv_special_attack, iv_special_defence, iv_speed, ev_hp, ev_attack, ev_defence, ev_special_attack, ev_special_defence, ev_speed, nature, ability, hidden_ability, gender, breedable, held_item, caught_ball, size, aspects, forced_aspects, form_aspects, moves, shard_name, REPLACE(SUBSTRING(description FROM ' for ([0-9,]+)$'), ',', '')::BIGINT AS price FROM pokemon_logger_events WHERE source = 'gts' AND action IN ('selling', 'sold')) TO '/home/minecraft/CobblemonSpawn/smash-properties/gts_events.csv' CSV HEADER
