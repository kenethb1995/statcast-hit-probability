from pybaseball import statcast
import pandas as pd

EXPECTED_COLUMNS = [
    'game_date',
    'events',
    'is_hit',
    'bb_type',
    'zone',
    'plate_x',
    'plate_z',
    'stand',
    'p_throws',
    'launch_speed',
    'launch_angle',
    'bat_speed',
    'swing_length',
    'attack_direction',
    'attack_angle',
    'swing_path_tilt',
    'intercept_ball_minus_batter_pos_x_inches',
    'intercept_ball_minus_batter_pos_y_inches',
    'hc_x',
    'hc_y',
    'hit_distance_sc'
]

def pull_statcast_batted_balls(start_date, end_date):
    """
    Pulls Statcast data for a date range and returns one row per batted-ball event.

    - Filters to batted-ball events using bb_type
    - Creates binary response variable is_hit from events
    - Returns a trimmed DataFrame aligned with the PostgreSQL schema

    Note:
    Some swing metrics are only available in recent seasons and may be null.
    """
    #creates df with pitch by pitch statcast data for selected dates
    df = statcast(start_dt=start_date, end_dt=end_date)
    #if statcast returns no data or unexpected schema for the day, skip
    if df is None or df.empty or "bb_type" not in df.columns:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    #filters for non-na values within bb_type column as na suggests a non batted ball event occured
    df = df[df['bb_type'].notna()].copy()

    #is_hit response variable created. Uses hit dictionary to label hit types as hit or not hit using events column
    hit = {'single', 'double', 'triple', 'home_run'}
    df['is_hit'] = df['events'].isin(hit)

    df_final = df[EXPECTED_COLUMNS].copy()

    #data validation -- ensures data is loaded to database properly
    if list(df_final.columns) != EXPECTED_COLUMNS:
        raise ValueError("Column order or names do not match schema")

    return df_final


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df_final = pull_statcast_batted_balls('2025-03-18', '2025-03-19')

    print('Shape:', df_final.shape)
    print('Columns', df_final.columns.tolist())
    print('Unique Event Types:', df_final['events'].unique())
    print('Unique Batted Ball Types:', df_final['bb_type'].unique())
    print('Missing Event Percentage:', df_final['events'].isna().sum()/df_final.shape[0])
    print('Missing Batted Ball Percentage:', df_final['bb_type'].isna().sum()/df_final.shape[0])
    print('Tail:', df_final.tail(3))

"""
Variables selected with description and rough data type.
game_date(date/time): data of when game took place.
events(Text): Used to derive is_hit response variable
is_hit(boolean): response variable describing whether batted ball was a hit or not
zone(text): Zone location of the ball when it crosses the plate from the catcher's perspective.
stand(text): Side of the plate batter is standing.
p_throws(text): Hand pitcher throws.
bb_type(text): Batted ball type, ground_ball, line_drive, fly_ball, popup
plate_x(float): Horizontal position of the ball when it crosses home plate from the catcher's perspective.
plate_z(float): Vertical position of the ball when it crosses home plate from the catcher's perspective.
hc_x(float): Hit coordinate X of batted ball.
hc_y(float): Hit coordinate Y of batted ball.
hit_distance_sc(float): Projected hit distance of the batted ball.
launch_speed(float): Exit velocity of the batted ball as tracked by Statcast.
launch_angle(float): Launch angle of the batted ball as tracked by Statcast.
bat_speed(float): Speed of bat measured at the sweet spot of bat. Roughly 6 inches from end of bat.
swing_length(float): Calculated in terms of feet the bat traveled during the swing,
attack_direction(float): Measures the horizontal direction that the sweet spot of the bat is moving at the point of contact with the baseball.
attack_angle(float): Reported with attack direction. Vertical angle the sweet spot of the bat is traveling at point of contact with the baseball.
swing_path_tilt(float): Maeasures the angular orientation of the "plane" of the swing, as compared to the ground, defined by the path of the bat in the 40 ms prior to contact, which approximates a slice of a flat disc in shape. A higher angle is a "steeper" swing (further from horizontal) and a lower angle is a "flatter" swing (closer to horizontal).
intercept_ball_minus_batter_pos_x_inches(float): Measures the horizontal location of the ball at the point of contact relative to the batter's position in the batter's box
intercept_ball_minus_batter_pos_y_inches(float): Measurement of the difference in the Y-axis position (depth in the batter's box) between where the ball is intercepted (or reaches its closest point to the bat) and the batter's position.
"""

"""
Omitted variables due to Target Leakage:
estimated_ba_using_speedangle
estimated_woba_using_speedangle

These are benchmarks, not features
Wise to omit variables that are a function of other features and the target
Want model to learn these relationships -- not borrow them

How These Metrics Should Be Used Instead
--Use them as model metrics!!!
After training your model, compare: 
- your predicted probabilities
- MLB's expected BA
This answers:
    "How close am I to Statcast's own model"
"""
