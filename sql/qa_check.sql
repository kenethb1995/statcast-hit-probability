/*
1. Purpose: The script aims to confirm data is ingested properly and as expected from Statcast after ETL loads.
2. Scope: The script verifies observations align with expected batted ball data. It ensures proper loading of
data via observation count checks based on day, month, and year. Finally, it verifies label consistency,
data ranges, and data values take on expected values.
3. Non-goal: The script does not enforce modeling requirements or model row counts and does not filter season based
on pre, regular or postseason.
4. Usage: Use this script as a quality assurance check prior to modeling and after full ETL loads.
 */

/*
 * Section 1: Coverage Check
 */

--total observations in table
select COUNT(*)
from public.batted_ball_events;

--date range of observations in table
select
	min(game_date) as date_of_first_record,
	max(game_date) as date_of_last_record
from public.batted_ball_events
;

--observations by day
select
	game_date,
	count(*) as observations
from public.batted_ball_events
group by game_date
order by game_date;

--total observations by month
select
	date_trunc('month', game_date)::date as year_month,
	count(*) as observations
from public.batted_ball_events
group by year_month
order by year_month
;

--total observations by year
select
	extract(year from game_date)::text as season,
	count(*) as observations
from public.batted_ball_events
group by season
order by season
;

--Total unique days with batted ball events
select
	count(distinct(game_date)) as total_number_of_days_w_bbe
from public.batted_ball_events
;

--Total unique days with batted ball events by year
select
	extract(year from game_date)::text as season,
	count(distinct(game_date)) as number_of_days_w_bbe
from public.batted_ball_events
group by season
order by season
;

/*
 * Section 2: Target Check
 */

--daily batted ball event hit rate
select
	game_date,
	sum(is_hit::int) as total_hits,
	count(*) as total_bbe,
	avg(is_hit::int)::DECIMAL(5,4) as bbe_hit_rate
from public.batted_ball_events
group by game_date
order by game_date;

--monthly batted ball event hit rate
select
	date_trunc('month', game_date)::date as year_month,
	sum(is_hit::int) as total_hits,
	count(*) as total_bbe,
	avg(is_hit::int)::decimal(5,4) as bbe_hit_rate
from public.batted_ball_events
group by year_month
order by year_month
;

--yearly batted ball event hit rate
select
	extract(year from game_date)::text as season,
	sum(is_hit::int) as total_hits,
	count(*) as total_bbe,
	avg(is_hit::int)::decimal(5,4) as bbe_hit_rate
from public.batted_ball_events
group by season
order by season
;

/*
 * Section 3: Missingness Check
 */

--missing value percentages for variables which were available throughout table date range
select
	extract(year from game_date)::text as season,
	round(avg(case when events is null then 1 else 0 end) * 100, 2) as percent_null_events,
	round(avg(case when zone is null then 1 else 0 end) * 100, 2) as percent_null_zone,
	round(avg(case when plate_x is null then 1 else 0 end) * 100, 2) as percent_null_plate_x,
	round(avg(case when plate_z is null then 1 else 0 end) * 100, 2) as percent_null_plate_z,
	round(avg(case when stand is null then 1 else 0 end) * 100, 2) as percent_null_stand,
	round(avg(case when p_throws is null then 1 else 0 end) * 100, 2) as percent_null_p_throws,
	round(avg(case when launch_speed is null then 1 else 0 end) * 100, 2) as percent_null_launch_speed,
	round(avg(case when launch_angle is null then 1 else 0 end) * 100, 2) as percent_null_launch_angle,
	round(avg(case when hc_x is null then 1 else 0 end) * 100, 2) as percent_null_hc_x,
	round(avg(case when hc_y is null then 1 else 0 end) * 100, 2) as percent_null_hc_y,
	round(avg(case when hit_distance_sc is null then 1 else 0 end) * 100, 2) as percent_null_hit_distance_sc
from public.batted_ball_events
group by season
order by season
;

--missing value percentages for variables which were not implemented until roughly mid-way through 2023 MLB season
select
	extract(year from game_date)::text as season,
	round(avg(case when bat_speed is null then 1 else 0 end) * 100, 2) as percent_null_bat_speed,
	round(avg(case when swing_length is null then 1 else 0 end) * 100, 2) as percent_null_swing_length,
	round(avg(case when swing_path_tilt is null then 1 else 0 end) * 100, 2) as percent_null_swing_path_tilt,
	round(avg(case when attack_direction is null then 1 else 0 end) * 100, 2) as percent_null_attack_direction,
	round(avg(case when intercept_ball_minus_batter_pos_x_inches is null then 1 else 0 end) * 100, 2) as percent_null_intercept_ball_minus_batter_pos_x_inches,
	round(avg(case when intercept_ball_minus_batter_pos_y_inches is null then 1 else 0 end) * 100, 2) as percent_null_intercept_ball_minus_batter_pos_y_inches
from public.batted_ball_events
group by season
order by season
;

/*
 * Section 4: Ranges and Outlier Check
 */

--checking ranges align with expected values for numeric columns
--thresholds represent ranges which are anticipated to be rare for batter contact(not necessarily invalid)
select
	count(*) as total_rows,
	min(launch_speed) as min_launch_speed,
	max(launch_speed) as max_launch_speed,
	100 * avg(case when launch_speed < 20 then 1 else 0 end) as percent_launch_speed_below_20mph,
	min(launch_angle) as min_launch_angle,
	max(launch_angle) as max_launch_angle,
	100 * avg(case when launch_angle > 80 or launch_angle < -80 then 1 else 0 end) as percent_launch_angle_plus_minus_80,
	min(plate_x) as min_plate_x,
	max(plate_x) as max_plate_x,
	100 * avg(case when plate_x > 4 or plate_x < -4 then 1 else 0 end) as percent_plate_x_outside_typical_contact_window,
	min(plate_z) as min_plate_z,
	max(plate_z) as max_plate_z,
	100 * avg(case when plate_z > 5 or plate_z < 0.5 then 1 else 0 end) as percent_plate_z_outside_typical_contact_window
from public.batted_ball_events
;

/*
 * Section 5: Label Consistency
 */

--verifies that only events with value single, double, triple or home_run have an is_hit value of true
select count(*) as mislabeled_hits
from public.batted_ball_events
where is_hit = true
	and events not in ('single', 'double', 'triple', 'home_run')
;
--verifies that no events with value single, double, triple or home_run have an is_hit value of false
select count(*) as missed_hits
from public.batted_ball_events
where is_hit = false
	and events in ('single', 'double', 'triple', 'home_run')
;



