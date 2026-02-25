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
    # print(f"Training DataFrame Shape: {X_train.shape}")
    # print(f"Testing DataFrame Shape: {X_test.shape}")

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

    # confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=predictions)
    class_labels = ['Not Hit', 'Hit']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Greens, values_format='d')
    #plt.show()

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