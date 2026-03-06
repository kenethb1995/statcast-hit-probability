"""
1.Purpose: This script retrieves and enforces modeling dataset requirements.
2.Scope: Reads database view created by modeling_extract.sql into pandas DataFrame and ensures
target variable integrity, presence of required columns, and schema parity.
3.Non-goal: Does not perform transformations, test/train splits, or feature engineering.
4.Usage: This script should be run after database view is created and prior to split.py.
"""

import pandas as pd
from src.etl.db import get_engine


def load_modeling_extract():
    engine = get_engine()
    sql_query = "SELECT * FROM public.modeling_extract ORDER BY game_date"
    df = pd.read_sql(sql_query, engine)
    df['game_date'] = pd.to_datetime(df['game_date'])

    assert df.shape[0] > 0, 'modeling_extract returned empty dataframe'

    columns_set = set(df.columns)
    required_columns = {'game_date', 'is_hit', 'bb_type',
                        'plate_x', 'plate_z', 'stand',
                        'p_throws', 'launch_speed', 'launch_angle',
                        'bat_speed', 'swing_length', 'attack_direction',
                        'attack_angle', 'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches',
                        'intercept_ball_minus_batter_pos_y_inches'}
    assert required_columns.issubset(columns_set), "Missing required columns"

    assert df['is_hit'].isnull().sum() == 0, "Target Variable is_hit contains null values"

    unique_values = df['is_hit'].unique()
    assert set(unique_values).issubset({0, 1, False, True}), 'is_hit is not binary'

    return df


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df = load_modeling_extract()
    print(df.head(3))
    print(df.tail(3))

    print(df.shape)
    print("Data Type For is_hit: ", df['is_hit'].dtype)
    print("Value Counts: ", df['is_hit'].value_counts().head())
    print("Min Date: ", min(df.game_date), "\nMax Date:", max(df.game_date))
    print("Unique Years: ", df.game_date.dt.year.unique())
