"""
1.Purpose: This script produces a baseline logistic regression model with baseline contact variables and serves
as the reference model. Also, outputs evaluation metrics and model coefficients for interpretation along with
relevant visualizations.

2.Scope: Utilizes time-based split for creation of training/testing DataFrames. Training DataFrame consists of
observations recorded between January 1, 2021, and December 31, 2024, and testing observations recorded on or
after January 1, 2025. Applies preprocessing pipelines including imputation, scaling, and encoding prior to
logistic regression training.

3.Non-Goal: Does not aim to provide the best model possible. No threshold optimization or hyperparameter tuning.

4.Usage: Run after proper implementation of extract.py and split.py.
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from src.modeling.extract import load_modeling_extract
from src.modeling.split import time_split

# explicit column naming
NUMERIC_VARIABLES = ['plate_x',
                     'plate_z',
                     'launch_speed',
                     'launch_angle']

CATEGORICAL_VARIABLES = ['bb_type',
                         'stand',
                         'p_throws',
                         ]
CATEGORICAL_REFERENCE = ['ground_ball',
                         'R',
                         'R']

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    # Section 1: Creating train/test splits
    df = load_modeling_extract()
    train_df, test_df = time_split(df,
                                   train_start='2021-01-01',
                                   train_end='2024-12-31',
                                   test_start='2025-01-01')

    # train/test splits
    X_train = train_df[NUMERIC_VARIABLES + CATEGORICAL_VARIABLES]
    y_train = train_df['is_hit']

    X_test = test_df[NUMERIC_VARIABLES + CATEGORICAL_VARIABLES]
    y_test = test_df['is_hit']

    # sanity checks
    print(f"Training DataFrame Shape: {X_train.shape}")
    print(f"Testing DataFrame Shape: {X_test.shape}")

    assert "is_hit" not in X_train.columns and "game_date" not in X_train.columns and \
           "batted_ball_id" not in X_train.columns, "Training DataFrame includes unexpected columns"
    assert X_train.columns.equals(
        X_test.columns), "Columns in training DataFrame do not match columns in testing DataFrame"
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train do not have the same number of rows"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test do not have the same number of rows"
    assert all(pd.api.types.is_numeric_dtype(X_train[col]) for col in NUMERIC_VARIABLES), "Non-numeric dtypes present"
    assert all(not pd.api.types.is_numeric_dtype(X_train[col]) for col in
               CATEGORICAL_VARIABLES), "Numeric dtype present in categorical variables"

    # Section 2: baseline logistic model
    # preprocessing and modeling pipeline
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore',
                                                                       drop=CATEGORICAL_REFERENCE,
                                                                       sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, NUMERIC_VARIABLES),
            ('categorical', categorical_transformer, CATEGORICAL_VARIABLES)
        ]
    )

    model = Pipeline(steps=[('preprocessing', preprocessor),
                            ('model', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=100))])

    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    # predicted hit rate vs. actual hit rate
    print(f"Actual Hit Rate: {y_test.mean().round(3)}")
    print(f"Expected Hit Rate: {pred_proba.mean().round(3)}")
    print(f"Predicted Hit Rate @.50 Threshold: {(pred_proba >= .50).mean().round(3)}")

    # relevant model performance metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    specificity = recall_score(y_test, predictions, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, pred_proba)
    ap_score = average_precision_score(y_test, pred_proba)

    print(f"Accuracy: {round(accuracy, 3)}\nPrecision: {round(precision, 3)}\nRecall: {round(recall, 3)}\n"
          f"Specificity: {round(specificity, 3)}\nF1: {round(f1, 3)}\nROC AUC: {round(roc_auc, 3)}\n"
          f"Average Precision: {round(ap_score, 3)}")

    # Outputting Model Coefficients
    feature_names = model[:-1].get_feature_names_out()
    model_coefs = pd.DataFrame(
        model[-1].coef_.ravel(),
        columns=["coefficients"],
        index=feature_names
    )
    model_coefs = model_coefs.sort_values("coefficients", ascending=False)
    model_coefs['odds ratio'] = np.exp(model_coefs['coefficients'])
    print(model_coefs)

    # Ensuring visuals can save to proper folder
    project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    fig_dir = os.path.join(project_root, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Hit Probability (%) by Batted Ball Type
    plt.rcParams["figure.figsize"] = (8, 6)
    ax = sns.barplot(df,
                     x="bb_type",
                     y="is_hit", estimator="mean",
                     errorbar=None,
                     order=["line_drive", "fly_ball", "ground_ball", "popup"],
                     color="steelblue")
    ax.set_xlabel("Batted Ball Type")
    ax.set_ylabel("Hit Probability (%)")
    ax.bar_label(ax.containers[0], labels=[f"{v.get_height()*100:.1f}%" for v in ax.containers[0]], fontsize=10)
    plt.title("Hit Probability by Batted Ball Type",
              loc="center",
              pad=20,
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Hit_Probability_by_Batted_Ball_Type.png"), dpi=300)
    plt.show()

    # Receiver Operating Characteristic
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='steelblue')
    plt.plot([0, 1], [0, 1], '--', label='Random (AUC = 0.5)', color='black')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Baseline Logistic Regression',
              loc="center",
              pad=20,
              fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ROC_Curve_Baseline_Logistic_Regression.png"), dpi=300)
    plt.show()

    # Coefficients Barplot
    model_coefs_renamed = model_coefs.rename(index={
        'categorical__bb_type_line_drive': 'Line Drive',
        'numeric__launch_speed': 'Launch Speed',
        'categorical__bb_type_fly_ball': 'Fly Ball',
        'categorical__p_throws_L': 'LH Pitcher',
        'numeric__plate_z': 'Plate Z',
        'numeric__plate_x': 'Plate X',
        'categorical__stand_L': 'LH Batter',
        'numeric__launch_angle': 'Launch Angle',
        'categorical__bb_type_popup': 'Popup'
    })
    model_coefs_renamed = model_coefs_renamed.sort_values("coefficients")
    model_coefs_renamed['impact'] = model_coefs_renamed['coefficients'].apply(lambda x: 'positive' if x > 0 else 'negative')
    palette = {'positive': 'steelblue',
               'negative': '#ab4541'}
    sns.barplot(model_coefs_renamed, x="coefficients", y=model_coefs_renamed.index, hue='impact', palette=palette)
    plt.axvline(x=0, linestyle="--", color="grey")
    plt.xlabel("Coefficients")
    plt.ylabel("Predictors")
    plt.title("Coefficient Impact On Hit Probability",
              loc="center",
              pad=20,
              fontweight="bold")
    plt.gca().get_legend().remove()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Coefficient_Impact_on_Hit_Probability.png"), dpi=300)
    plt.show()

    # Batted ball outcomes by launch speed and angle (1000 random samples)
    df = df.copy().sample(n=1000, random_state=100)
    df['is_hit'] = df['is_hit'].map({True: "Hit", False: "Non-Hit"})
    palette = {'Hit': 'steelblue',
               'Non-Hit': '#ab4541'}
    sns.scatterplot(data=df,
                    x='launch_angle',
                    y='launch_speed',
                    hue='is_hit',
                    palette=palette,
                    alpha=.70)
    plt.xlabel("Launch Angle (Degrees)")
    plt.ylabel("Launch Speed (MPH)")
    plt.legend(loc='lower right',
               reverse=True)
    plt.title(label="Batted Ball Outcomes by Launch Speed and Angle",
              loc="center",
              pad=20,
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Batted_Ball_Outcomes_by_Launch_Speed_and_Angle.png"), dpi=300)
    plt.show()