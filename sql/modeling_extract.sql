/*
1. Purpose: This script will be used to create a reproducible slice of data using deterministic documented
SQL filters which will be exported and used for modeling purposes.
2. Scope: This script utilizes filters to omit variables which encode outcomes or post-contact information.
3. Non-goal: This script does not handle train/test splits nor does it perform feature engineering or transformations.
Also, it does not validate completeness or data quality.
4. Usage: This script should be used after the QA script and prior to modeling. It is safe to rerun if necessary.

A. Population: The population will consist of all MLB regular season batted ball event observations within
the batted_ball_events table.
B. Feature Inclusion Rules: Included variables detail pitch location, batter/pitcher handedness, batted ball metrics,
and bat tracking metrics.
C. Leakage Defense: Excludes variables which encode outcome and variables which occur post-contact.
D. Grain Consistency: Will ensure one row = one observation via batted_ball_id as the row identifier.
E. Output Intent: The output should be a filtered dataset ready for export and modeling.

Missingness Policy: This script will omit observations with null values for time-of-contact metrics. Null values in other
columns will be allowed.
- launch_speed
- launch_angle

 Regular Season Filter: Omits dates outside domestic MLB regular season games. A small number of international
 opener regular season observations are omitted to preserve population integrity as preseason games
 were played concurrently and game-type identifiers were not retained in the curated schema.
- 2021 MLB regular season: 2021-04-01 to 2021-10-03
- 2022 MLB regular season: 2022-04-07 to 2022-10-05
- 2023 MLB regular season: 2023-03-30 to 2023-10-01
- 2024 MLB regular season: 2024-03-28 to 2024-09-30
- 2025 MLB regular season: 2025-03-27 to 2025-09-28

Target Leakage Filter: Will completely filter out events variable as it coincides with is_hit resulting in target leakage.
- events

Post-Contact Filter: Will filter out hit location information which occurs post-contact.
- hc_x
- hc_y
- hit_distance_sc

Derived Variable Filter: Will remove less informative variable(s) which can be derived from other continuous variables.
- zone

 */
create or replace view public.modeling_extract as
select
	batted_ball_id,
	game_date,
	is_hit,
	bb_type,
	plate_x,
	plate_z,
	stand,
	p_throws,
	launch_speed,
	launch_angle,
	bat_speed,
	swing_length,
	attack_direction,
	attack_angle,
	swing_path_tilt,
	intercept_ball_minus_batter_pos_x_inches,
	intercept_ball_minus_batter_pos_y_inches
from public.batted_ball_events
where (game_date between '2021-04-01' and '2021-10-03'
	or game_date between '2022-04-07' and '2022-10-05'
	or game_date between '2023-03-30' and '2023-10-01'
	or game_date between '2024-03-28' and '2024-09-30'
	or game_date between '2025-03-27' and '2025-09-28'
	)
and launch_speed is not null
and launch_angle is not null
;

--scratch work ensuring view counts match original filtered dataset

--select min(game_date), max(game_date), count(*)
--from public.batted_ball_events
--where (game_date between '2021-04-01' and '2021-10-03'
--	or game_date between '2022-04-07' and '2022-10-05'
--	or game_date between '2023-03-30' and '2023-10-01'
--	or game_date between '2024-03-28' and '2024-09-30'
--	or game_date between '2025-03-27' and '2025-09-28'
--	)
--and launch_speed is not null
--and launch_angle is not null
--;

--select count(*) as view_obs from public.modeling_extract;
--select min(game_date), max(game_date) from public.modeling_extract;
