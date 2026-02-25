"""
This script should be quite similar to baseline_logistic_regression.py just includes tracking era variable.
Tracking-Era Date Range: Will be first date that tracking variables are available
Training Dates: 2023-07-14 - 2024-12-31
Testing Dates: 2025-01-01 - 2025-12-31
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
ALL_NUMERIC_VARIABLES = BASE_NUMERIC_VARIABLES + TRACKING_NUMERIC_VARIABLES
CATEGORICAL_REFERENCE = ['ground_ball',
                         'R',
                         'R']

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    # Section 1: Creating train/test splits
    df = load_modeling_extract()

    train_df_track, test_df_track = time_split(df,
                                               train_start='2023-07-14',
                                               train_end='2024-12-31',
                                               test_start='2025-01-01')
    #print(train_df_track.shape)
    #print(test_df_track.shape)

    X_train_track = train_df_track[CATEGORICAL_VARIABLES + ALL_NUMERIC_VARIABLES]
    y_train_track = train_df_track['is_hit']

    X_test_track = test_df_track[CATEGORICAL_VARIABLES + ALL_NUMERIC_VARIABLES]
    y_test_track = test_df_track['is_hit']

    #print(X_train_track.shape, X_test_track.shape)
    #print(y_train_track.shape, y_test_track.shape)

    #Section 2: Baseline Logistic Regression Model with tracking era variables
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="median")),
                                          ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="most_frequent")),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore',
                                                                       drop=CATEGORICAL_REFERENCE,
                                                                       sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, ALL_NUMERIC_VARIABLES),
            ('categorical', categorical_transformer, CATEGORICAL_VARIABLES)
        ]
    )

    model_track = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', LogisticRegression(max_iter=1000, solver="lbfgs", random_state=100))])

    model_track.fit(X_train_track, y_train_track)
    pred_proba_track = model_track.predict_proba(X_test_track)[:, 1]
    predictions_track = model_track.predict(X_test_track)

    # Actual Hit Rate, Model's Expected Hit Rate, and Rate of BBE Events Classified As Hits @.50 Threshold
    print(f"Actual Hit Rate: {y_test_track.mean().round(3)}")
    print(f"Expected Hit Rate: {pred_proba_track.mean().round(3)}")
    print(f"Rate of BBE Events Classified As Hits @.50 Threshold : {(pred_proba_track >= .50).mean().round(3)}")

    #confusion matrix
    cm = confusion_matrix(y_true=y_test_track, y_pred=predictions_track)
    class_labels = ['Not Hit', 'Hit']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    #plt.show()

    # relevant model performance metrics
    accuracy_track = accuracy_score(y_test_track, predictions_track)
    precision_track = precision_score(y_test_track, predictions_track, zero_division=0)
    recall_track = recall_score(y_test_track, predictions_track, zero_division=0)
    specificity_track = recall_score(y_test_track, predictions_track, pos_label=0, zero_division=0)
    f1_track = f1_score(y_test_track, predictions_track, zero_division=0)
    roc_auc_track = roc_auc_score(y_test_track, pred_proba_track)
    ap_score_track = average_precision_score(y_test_track, pred_proba_track)

    print(f"Accuracy: {round(accuracy_track, 3)}\nPrecision: {round(precision_track, 3)}\nRecall: {round(recall_track, 3)}\n"
          f"Specificity: {round(specificity_track, 3)}\nF1: {round(f1_track, 3)}\nROC AUC: {round(roc_auc_track, 3)}\n"
          f"Average Precision: {round(ap_score_track, 3)}")


