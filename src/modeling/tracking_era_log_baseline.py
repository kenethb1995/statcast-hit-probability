"""
This script should be quite similar to baseline_logistic_regression.py just includes tracking era variable.
Tracking-Era Date Range: Will be first date that tracking variables are available
Training Dates: , all of 2024
Testing Dates: , all of 2025
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from src.modeling.extract import load_modeling_extract
from src.modeling.split import time_split

# explicit column naming
BASE_NUMERIC_VARIABLES = ['plate_x',
                          'plate_z',
                          'launch_speed',
                          'launch_angle']

CATEGORICAL_VARIABLES = ['bb_type',
                         'stand',
                         'p_throws',
                         ]
TRACKING_NUMERIC_VARIABLES = ['bat_speed',
                              'swing_length',
                              'attack_direction',
                              'attack_angle',
                              'swing_path_tilt',
                              'intercept_ball_minus_batter_pos_x_inches',
                              'intercept_ball_minus_batter_pos_y_inches']
CATEGORICAL_REFERENCE = ['ground_ball',
                         'R',
                         'R']

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    # Section 1: Creating train/test splits
    df = load_modeling_extract()

    print(df[df['game_date'] >= '2023-07-14'].isna().mean())
    print(df[df['game_date'].dt.year == 2024].isna().mean())
    print(df[df['game_date'].dt.year == 2025].isna().mean())


