"""
1.Purpose: This script serves as a controlled A/B feature evaluation comparing tracking-era bat metrics versus
baseline contact metrics under identical preprocessing and splits.

2.Scope: Ensures experimental validity by utilizing time-based splits and shared pipeline configuration. Also
ensures both models receive identical rows by omitting observations with missing values in all variables used.

3.Non-Goal: This script does not perform hyperparameter tuning, threshold optimization, feature engineering,
or model selection.

4.Usage: This script should be run after baseline_logistic_regression.py and bat_tracking_logistic_regression.py have
been implemented and validated.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from src.modeling.extract import load_modeling_extract
from src.modeling.split import time_split

# Model Variables
BASE_NUMERIC_VARIABLES = ['plate_x',
                          'plate_z',
                          'launch_speed',
                          'launch_angle']

BASE_CATEGORICAL_VARIABLES = ['bb_type',
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
MODEL_A_VARIABLES = BASE_NUMERIC_VARIABLES + BASE_CATEGORICAL_VARIABLES
MODEL_B_VARIABLES = BASE_NUMERIC_VARIABLES + TRACKING_NUMERIC_VARIABLES + BASE_CATEGORICAL_VARIABLES
ALL_VARIABLES = MODEL_B_VARIABLES

CATEGORICAL_REFERENCE = {"bb_type": 'ground_ball',
                         "stand": 'R',
                         "p_throws": 'R'}

# Train/Test Split Dates
TRAIN_START = '2023-07-14'
TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'

# Threshold
THRESHOLD = 0.5


def build_pipeline(numeric_variables, categorical_variables, categorical_reference, random_state=100):
    assert set(categorical_variables).issubset(categorical_reference.keys()), \
        "Not Every Categorical Variable Has A Set Reference Category"
    drop_list = [categorical_reference[col] for col in categorical_variables]
    numeric_transformer = Pipeline(steps=[("median_impute", SimpleImputer(strategy='median')),
                                          ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[('mode_impute', SimpleImputer(strategy='most_frequent')),
                                              ("encoder", OneHotEncoder(
                                                  handle_unknown='ignore',
                                                  drop=drop_list,
                                                  sparse_output=False
                                              ))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_variables),
            ('categorical', categorical_transformer, categorical_variables)],
        remainder='drop'
    )

    model = Pipeline(steps=[('preprocessing', preprocessor),
                            ('model', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=random_state))])

    return model


def evaluation(y_true, y_proba, threshold):
    y_predicted = (y_proba >= threshold).astype(int)
    observed_hit_rate = y_true.mean()
    expected_hit_rate = y_proba.mean()
    pred_hit_at_thresh = y_predicted.mean()

    model_accuracy = accuracy_score(y_true, y_predicted)
    model_precision = precision_score(y_true, y_predicted, zero_division=0)
    model_recall = recall_score(y_true, y_predicted, zero_division=0)
    model_specificity = recall_score(y_true, y_predicted, pos_label=0, zero_division=0)
    model_f1 = f1_score(y_true, y_predicted,zero_division=0)
    model_roc_auc = roc_auc_score(y_true, y_proba)
    model_ap_score = average_precision_score(y_true, y_proba)

    model_metrics = {'Observed Hit Rate': observed_hit_rate,
                     'Expected Hit Rate': expected_hit_rate,
                     'Predicted Hits At Threshold': pred_hit_at_thresh,
                     'Accuracy': model_accuracy,
                     'Precision': model_precision,
                     'Recall': model_recall,
                     'Specificity': model_specificity,
                     'F1': model_f1,
                     'ROC AUC': model_roc_auc,
                     'Average Precision': model_ap_score}

    return model_metrics


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df = load_modeling_extract()
    REQUIRED_COLUMNS = set(MODEL_B_VARIABLES + ['is_hit', 'game_date'])
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing the following required columns: {missing}\n"
    print(df.shape)

    # Apply time split for dataset
    train_df, test_df = time_split(df,
                                   train_start=TRAIN_START,
                                   train_end=TRAIN_END,
                                   test_start=TEST_START)
    assert train_df.shape[0] > 0, "Empty Train DataFrame"
    assert test_df.shape[0] > 0, "Empty Test DataFrame"

    # Filtering out observations with missing data in columns defined in ALL_VARIABLES
    keep_obs_train = train_df[ALL_VARIABLES].notna().all(axis=1)
    keep_obs_test = test_df[ALL_VARIABLES].notna().all(axis=1)

    train_df_filtered = train_df.loc[keep_obs_train].copy()
    test_df_filtered = test_df.loc[keep_obs_test].copy()

    assert train_df_filtered.shape[0] > 0, "Filtered Training DataFrame Is Empty"
    assert test_df_filtered.shape[0] > 0, "Filtered Testing DataFrame Is Empty"

    pre_filter_train_obs = train_df.shape[0]
    post_filter_train_obs = train_df_filtered.shape[0]
    diff_obs_train = pre_filter_train_obs - post_filter_train_obs

    pre_filter_test_obs = test_df.shape[0]
    post_filter_test_obs = test_df_filtered.shape[0]
    diff_obs_test = pre_filter_test_obs - post_filter_test_obs

    print(f"Pre-filtered Training DataFrame Observations: {pre_filter_train_obs:,d}\n"
          f"Post-filtered Training DataFrame Observations: {post_filter_train_obs:,d}\n"
          f"Observations Dropped After Filtering: {diff_obs_train:,d}\n"
          f"Percent Of Train Observations Dropped: {diff_obs_train / pre_filter_train_obs:.2%}\n")

    print(f"Pre-filtered Testing DataFrame Observations: {pre_filter_test_obs:,d}\n"
          f"Post-filtered Testing DataFrame Observations: {post_filter_test_obs:,d}\n"
          f"Observations Dropped After Filtering: {diff_obs_test:,d}\n"
          f"Percent Of Test observations Dropped: {diff_obs_test / pre_filter_test_obs:.2%}\n")

    # safely overwriting train_df/test_df prior to modeling
    train_df = train_df_filtered
    test_df = test_df_filtered

    assert train_df[ALL_VARIABLES].isna().sum().sum() == 0, "Training DataFrame Contains NAs"
    assert test_df[ALL_VARIABLES].isna().sum().sum() == 0, "Testing DataFrame Contains NAs"

    # Model A and Model B DataFrame prep
    X_train_A = train_df[MODEL_A_VARIABLES].copy()
    X_test_A = test_df[MODEL_A_VARIABLES].copy()

    X_train_B = train_df[MODEL_B_VARIABLES].copy()
    X_test_B = test_df[MODEL_B_VARIABLES].copy()

    y_train = train_df['is_hit']
    y_test = test_df['is_hit']

    print(f"Training DataFrame BBE Hit Rate: {y_train.mean():.3f}")
    print(f"Testing DataFrame BBE Hit Rate: {y_test.mean():.3f}")

    assert X_train_A.shape[0] == y_train.shape[0], "Unequal Number of Observations Between X_train_A & y_train"
    assert X_train_B.shape[0] == y_train.shape[0], "Unequal Number of Observations Between X_train_B & y_train"

    assert X_test_A.shape[0] == y_test.shape[0], "Unequal Number of Observations Between X_test_A & y_test"
    assert X_test_B.shape[0] == y_test.shape[0], "Unequal Number of Observations Between X_test_B & y_test"

    assert X_train_A.index.equals(X_train_B.index), "Indexes Between Training DataFrames Do Not Match"
    assert X_test_A.index.equals(X_test_B.index), "Indexes Between Testing DataFrames Do Not Match"

    # Fitting Model A and Model B
    model_A = build_pipeline(BASE_NUMERIC_VARIABLES,
                             BASE_CATEGORICAL_VARIABLES,
                             CATEGORICAL_REFERENCE)
    model_A.fit(X_train_A, y_train)

    model_A_prob = model_A.predict_proba(X_test_A)[:, 1]

    model_B = build_pipeline(BASE_NUMERIC_VARIABLES + TRACKING_NUMERIC_VARIABLES,
                             BASE_CATEGORICAL_VARIABLES,
                             CATEGORICAL_REFERENCE)
    model_B.fit(X_train_B, y_train)

    model_B_prob = model_B.predict_proba(X_test_B)[:, 1]

    # Evaluation
    model_A_metrics = evaluation(y_test, model_A_prob, THRESHOLD)
    model_B_metrics = evaluation(y_test, model_B_prob, THRESHOLD)

    results_df = pd.DataFrame([model_A_metrics, model_B_metrics],
                       index=['Model A', 'Model B'])
    results_df.loc['Metric Difference: B - A'] = results_df.loc['Model B'] - results_df.loc['Model A']
    print(results_df)


