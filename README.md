Model will answer one precise question:
At the moment of contact, based only on information available then,
what is the probabilty this batted ball becomes a hit?

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
- Observations on or before December 31, 2024 used to create training DataFrame.
- Observations on or after January 1, 2025 used to create testing DataFrame.
- Does not use random sampling to create train/test DataFrames in order to mirror real-world deployment, where models are trained on past seasons and evaluated on future seasons.
- Does not handle transformation or feature engineering.

Run this script after retrieval and validation of the predictive modeling dataset.
