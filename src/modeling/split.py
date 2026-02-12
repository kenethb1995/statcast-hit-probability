"""
1.Purpose: This script performs time-based splitting on the modeling dataset to create train/test DataFrames.
2.Scope: Training DataFrame includes observations on or before December 31, 2024 and the testing DataFrame
includes observations on or after January 1, 2025.
3.Non-goal: Does not utilize randomization to create train/test splits. Nor does the script perform transformations
or feature engineering.
4.Usage: This script should be run after the retrieval and validation of the predictive modeling dataset.
"""
import pandas as pd
from extract import load_modeling_extract

def time_split(df):

    training = '2024-12-31'
    testing = '2025-01-01'
    train_df = df[df['game_date'] <= training]
    test_df = df[df['game_date'] >= testing]

    assert len(train_df) > 0, "train_df is empty"
    assert len(test_df) > 0, "test_df is empty"
    assert train_df['game_date'].max() < test_df['game_date'].min(), \
        f"train_df game_date {train_df['game_date'].max()} overlaps test_df game_date {test_df['game_date'].min()}"

    return train_df, test_df

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    df = load_modeling_extract()
    train_df, test_df = time_split(df)
    print('Training DataFrame Shape: ',train_df.shape)
    print(f"Training DataFrame Date Range: {train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}")
    print('Testing DataFrame Shape: ',test_df.shape)
    print(f"Testing DataFrame Date Range: {test_df['game_date'].min().date()} to {test_df['game_date'].max().date()}")

    print(f"Hit Rate For Batted Ball Events: Training DataFrame\n{train_df['is_hit'].mean().round(3)}")
    print(f"Hit Rate For Batted Ball Events: Testing DataFrame\n{test_df['is_hit'].mean().round(3)}")