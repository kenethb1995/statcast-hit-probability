"""
1.Purpose: This script performs time-based splitting on the modeling dataset to create train/test DataFrames.
2.Scope: Training DataFrame composed of observations between train_start date and train_end date (inclusive). Testing
DataFrame composed of observations collected on dates on or after test_start. Default baseline model will use 2021-2024
for training and 2025 for testing.
3.Non-goal: Does not utilize randomization to create train/test splits. Nor does the script perform transformations
or feature engineering.
4.Usage: This script should be run after the retrieval and validation of the predictive modeling dataset.
"""
import pandas as pd
from src.modeling.extract import load_modeling_extract


def time_split(df, train_start, train_end, test_start):
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='raise').dt.normalize()
    train_start = pd.to_datetime(train_start).normalize()
    train_end = pd.to_datetime(train_end).normalize()
    test_start = pd.to_datetime(test_start).normalize()
    assert train_end >= train_start, "train_end date cannot be before train_start date"
    assert test_start > train_end, "test_start date must be greater than train_end date"

    train_df = df[(df['game_date'] >= train_start) & (df['game_date'] <= train_end)]
    test_df = df[df['game_date'] >= test_start]

    assert len(train_df) > 0, "train_df is empty"
    assert len(test_df) > 0, "test_df is empty"

    return train_df, test_df


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    df = load_modeling_extract()
    train_df, test_df = time_split(df, '2021-01-01', '2024-12-31', '2025-01-01')
    print('Training DataFrame Shape: ', train_df.shape)
    print(
        f"Training DataFrame Date Range: {train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}")
    print('Testing DataFrame Shape: ', test_df.shape)
    print(f"Testing DataFrame Date Range: {test_df['game_date'].min().date()} to {test_df['game_date'].max().date()}")

    print(f"Hit Rate For Batted Ball Events: Training DataFrame\n{train_df['is_hit'].mean().round(3)}")
    print(f"Hit Rate For Batted Ball Events: Testing DataFrame\n{test_df['is_hit'].mean().round(3)}")