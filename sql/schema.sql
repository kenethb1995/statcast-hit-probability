/*
1. Purpose: The script creates the batted_ball_events table stored in PostgreSQL. Each row
represents a unique batted-ball event.
2. Scope: The created table will contain a curated subset of variables related to batted-ball events. Variable
categories include identifiers, outcomes, contact physics, swing mechanics, plate location, hit location,
batter/pitcher handedness and bat tracking (will contain nulls as metrics tracked later).
3. Non-goal: The table will deliberately exclude irrelevant variables and Statcast expected metrics such as
xBA(expected batting average) and xwOBA(expected weighted on-base average) to avoid target leakage.
4. Usage: Run this script prior to the ETL process script.
*/

CREATE TABLE batted_ball_events (
    -- identifiers
    batted_ball_id BIGSERIAL PRIMARY KEY,
    game_date DATE NOT NULL,

    -- outcome
    events TEXT,
    is_hit BOOLEAN NOT NULL,

    -- contact classification
    bb_type TEXT NOT NULL,

    -- context
    zone TEXT,
    plate_x DOUBLE PRECISION,
    plate_z DOUBLE PRECISION,
    stand TEXT,
    p_throws TEXT,

    -- contact physics
    launch_speed DOUBLE PRECISION,
    launch_angle DOUBLE PRECISION,

    -- swing mechanics
    bat_speed DOUBLE PRECISION,
    swing_length DOUBLE PRECISION,
    attack_direction DOUBLE PRECISION,
    attack_angle DOUBLE PRECISION,
    swing_path_tilt DOUBLE PRECISION,
    intercept_ball_minus_batter_pos_x_inches DOUBLE PRECISION,
    intercept_ball_minus_batter_pos_y_inches DOUBLE PRECISION,

    -- hit location details
    hc_x DOUBLE PRECISION,
    hc_y DOUBLE PRECISION,
    hit_distance_sc DOUBLE PRECISION
);