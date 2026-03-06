Model will answer one precise question:
At the moment of contact, based only on information available then,
what is the probability this batted ball becomes a hit?

Schema (Table Creation)
==
The schema is implemented in `sql/schema.sql`
- Creates batted_ball_events table, a curated subset of variables related to batted ball events, within PostgreSQL. 
- Each row represents a unique batted ball event
- Not a full mirror of the Statcast schema
- Omits Statcast expected metrics to avoid target leakage

Run this script prior to the ETL script.

Data Quality Assurance (QA)
==
The QA layer is implemented in `sql/qa_check.sql`
- Confirms Statcast batted-ball data is properly ingested and loaded into PostgreSQL.
- Confirms load is complete for 2021-2025 Statcast window by verifying counts by day, month, and year.
- Ensures numeric variables take on plausible ranges and values.
- Verifies labels (is_hit vs. events) are consistent.
- Does not filter season by preseason, regular or postseason.
- Does not restrict observations to model row counts.
- Does not apply modeling feature requirements.

Run this script after full ETL loads and prior to modeling.

Modeling Extract 
==
The extract is incorporated in `sql/modeling_extract.sql`
- Creates a reproducible slice of data for use in predictive modeling where one row represents one batted-ball event.
- Creates the view public.modeling_extract, which serves as the modeling blueprint.
- Filters observations to ensure only domestic regular season games between 2021-2025 are present.
- Omits the variables hc_x, hc_y, and hit_distance_sc as they are captured post-contact.
- Omits the variable zone as it is captured by continuous variables plate_x and plate_z.
- Omits the variable events as it encodes the outcome of each batted ball event.
- Removes entire row if either launch_speed or launch_angle are null.
- Retains observations with null values in all other columns.
- Does not create train/test splits
- Does not perform feature engineering or transformations.

Run this script after data quality assurance and prior to modeling.

Predictive Model Dataset Retrieval
==
The retrieval of the modeling dataset is handled by `extract.py`
- Retrieves the frozen modeling view from PostgreSQL and enforces modeling dataset requirements.
- Ensures schema parity, required column presence, and target variable validity.
- Does not handle transformations, train/test splits, or feature engineering.

Run after the modeling view has been created and prior to time-based train/test split.

Training and Testing DataFrames
==
Creation of training and testing DataFrames is handled by `split.py`
- Observations on or before December 31, 2024, used to create Training DataFrame.
- Observations on or after January 1, 2025, used to create Testing DataFrame.
- Does not use random sampling to create train/test DataFrames in order to mirror real-world deployment, where models are trained on past seasons and evaluated on future seasons.
- Does not handle transformation or feature engineering.

Run this script after retrieval and validation of the predictive modeling dataset.

Baseline Logistic Regression Model 
==
Creation of the baseline logistic regression model is handled by `baseline_logistic_regression.py`
- Trains logistic regression model using baseline Statcast features plate_x, plate_z, launch_speed, launch_angle, bb_type, stand, and p_throws.
- Uses time-based split where training observations occur between January 1, 2021, and December 31, 2024, and testing observations occur on or after January 1, 2025.
- Applies preprocessing pipelines including imputation, scaling of numeric features, and one-hot encoding for categorical features.
- Uses a fixed threshold of 0.50 for predicted hits.
- Serves as the baseline model for subsequent bat-tracking feature comparison experiments.

Run this script after successful predictive model dataset retrieval and train/test split.

Bat-Tracking Logistic Regression Model
==
Creation of the bat-tracking logistic regression model is handled by `bat_tracking_logistic_regression.py`
- Trains logistic regression model using the same baseline Statcast features with inclusion of bat-tracking features bat_speed, swing_length, attack_direction, attack_angle, swing_path_tilt, intercept_ball_minus_batter_pos_x_inches, and intercept_ball_minus_batter_pos_y_inches.
- Uses time-based split where training observations occur between July 14, 2023, and December 31, 2024, and testing observations occur on or after January 1, 2025.
- Applies preprocessing pipelines including imputation, scaling of numeric features, and one-hot encoding for categorical features.
- Uses a fixed threshold of 0.50 for predicted hits.
- Serves as the bat-tracking model for feature comparison experiments.

Run this script after implementation of the baseline logistic regression model.

A/B Feature Experimental Comparison
==
Feature comparison handled by `baseline_vs_bat_tracking_comparison.py`
- Uses time-based splits to create training and testing DataFrames
- Observations between July 14, 2023, and December 31, 2024 (inclusive) are used to create training DataFrame.
- Observations on or after January 1, 2025, are used to create testing DataFrame.
- Both models receive identical rows by omitting observations with missing values across any features used in the comparison.
- Predicted hits are determined using a fixed threshold of 0.50.
- Does not handle feature engineering, threshold optimization, hyperparameter tuning, or model selection.
- Produces summary table directly comparing models based on predicted hits at fixed threshold, Accuracy, Precision, Recall, Specificity, F1, ROC AUC, and Average Precision.

Run this script after implementing and validating baseline models.

Feature Experiment Summary
==
**Experiment Goal:**<br> 
Determine whether the inclusion of bat-tracking features provides meaningful improvement in estimating hit probability compared to baseline Statcast features when using a logistic regression model with identical preprocessing and modeling pipelines.

**Ranking Performance:**<br> 
Model B produced an ROC AUC value of .766399 compared to model A’s ROC AUC of .766662. Therefore, when it comes to assigning probabilities, if we randomly select a hit and a non-hit, both models will assign a higher probability to the hit than the non-hit roughly 77% of the time. As a result, both models are near identical and discriminate between classes (hit versus non-hit) well. 
Of note, model B produced a higher average precision value of .594423 compared to model A’s .591399. Ultimately, this difference is negligible but conveys model B may have ranked a few hits higher than non-hits compared to model A.

**Threshold Behavior:**<br> 
Model B and model A both shared a fixed threshold of 0.50 with batted ball events being classified as hits whenever predicted probabilities were greater than or equal to 0.50 and classified as non-hits whenever predicted probabilities fell below 0.50. Ultimately, both models were quite conservative with model B producing a predicted hit rate of .207187 compared to model A’s predicted hit rate of .211176. In other words, model B predicted 20.7% of all batted ball events would result in a hit while model A predicted 21.1%, suggesting both models are quite conservative at the 0.50 threshold. Interestingly, model B had a higher precision of .621211 compared to model A’s .618198, yet model A had a recall of .412276 compared to model B’s .406459 showcasing the precision recall tradeoff. Because model B was a tad more selective when predicting hits, it could predict less false positives (non-hit predicted as hit). However, because it is more selective it also doesn’t predict hits as often leading to more false negatives (hit predicted as non-hit) leading to a lower recall rate. 
Furthermore, both models were trained and tested on unbalanced data with respective hit rates of .319 and .317. This imbalance could lead to model accuracy being misleading. To account for this imbalance F1 scores were calculated and again model A edged out model B (.494662 to .491397) confirming the inclusion of bat-tracking metrics led to no meaningful improvement in precision or recall.

**Conclusion:**<br> 
All core evaluation metrics confirm the inclusion of bat-tracking features did not lead to any meaningful improvements in batted ball event hit prediction capabilities. In fact, because the evaluation metrics between the two models were near identical this suggests that the baseline Statcast contact metrics up to the point of contact captured a lot of the same variation captured by bat-tracking metrics. This is sensible because features such as launch_speed and launch_angle are contact dependent. In the causal timeline of events, a hit is preceded by contact and contact is preceded by a swing. Therefore, features which capture swing mechanics will be captured by contact physics. Although predictive capabilities were not improved, it would be wrong to suggest bat-tracking features are useless. Instead, the results suggest this project is not the right instance to utilize bat-tracking metrics but if the future work aimed to predict launch_speed, then I believe bat-tracking features used would shine and be of great use. Ultimately, it can be concluded that bat-tracking features provide no meaningful improvement in estimating hit probabilities.
