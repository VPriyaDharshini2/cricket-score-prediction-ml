"""
Experiment 4: Temporal Feature Ablation

Purpose:
    Determine which groups of temporal/momentum features
    contribute to cricket final-score prediction.

Controlled experiment:
    - Same dataset
    - Same match-level train/test split
    - Same training rows
    - Same testing rows
    - Same XGBoost configuration
    - Only the feature groups are changed

Experiments:

A. State Only
B. State + Recent Scoring
C. State + Ball Quality
D. State + Wicket Momentum
E. State + All Temporal Features
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_score_dataset.csv"
)

RESULTS_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ablation_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 100_000


# ============================================================
# BASE FEATURES
# ============================================================

BASE_NUMERIC_FEATURES = [
    "innings",
    "over",
    "ball",
    "total_score",
    "wickets",
    "last_5_overs_score",
]


STATE_FEATURES = [
    "current_run_rate",
    "overs_remaining",
]


# ============================================================
# TEMPORAL FEATURE GROUPS
# ============================================================

RECENT_SCORING_FEATURES = [
    "last_3_overs_score",
    "last_10_overs_score",
]


BALL_QUALITY_FEATURES = [
    "boundary_rate",
    "dot_ball_rate",
]


WICKET_MOMENTUM_FEATURES = [
    "recent_5_over_wickets",
]


ALL_TEMPORAL_FEATURES = (
    RECENT_SCORING_FEATURES
    + BALL_QUALITY_FEATURES
    + WICKET_MOMENTUM_FEATURES
)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "season",
    "venue",
    "batting_team",
    "bowling_team",
    "batter",
    "non_striker",
    "bowler",
]


TARGET = "final_score"

GROUP_COLUMN = "match_id"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING TEMPORAL DATASET")
    print("=" * 70)

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Matches: "
        f"{df[GROUP_COLUMN].nunique():,}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    all_numeric = (
        BASE_NUMERIC_FEATURES
        + STATE_FEATURES
        + ALL_TEMPORAL_FEATURES
        + [TARGET]
    )

    required_columns = (
        all_numeric
        + CATEGORICAL_FEATURES
        + [GROUP_COLUMN]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required_columns].copy()

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    for column in all_numeric:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Handle categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # Remove rows where required numerical information
    # is unavailable.
    #
    # Every experiment will therefore use EXACTLY the
    # same population of rows.
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=(
            BASE_NUMERIC_FEATURES
            + STATE_FEATURES
            + ALL_TEMPORAL_FEATURES
            + [TARGET]
        )
    )

    removed = before - len(df)

    print(
        f"Rows removed due to missing values: "
        f"{removed:,}"
    )

    print(
        f"Rows available: {len(df):,}"
    )

    return df


# ============================================================
# MATCH-LEVEL SPLIT
# ============================================================

def split_by_match(df):

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    matches = df[
        GROUP_COLUMN
    ].unique()

    rng.shuffle(matches)

    split_index = int(
        len(matches) * (1 - TEST_SIZE)
    )

    train_matches = matches[
        :split_index
    ]

    test_matches = matches[
        split_index:
    ]

    train_df = df[
        df[GROUP_COLUMN].isin(train_matches)
    ].copy()

    test_df = df[
        df[GROUP_COLUMN].isin(test_matches)
    ].copy()

    overlap = (
        set(train_matches)
        &
        set(test_matches)
    )

    if overlap:

        raise RuntimeError(
            "Match leakage detected."
        )

    print("\nMATCH-LEVEL SPLIT")
    print("-" * 70)

    print(
        f"Training matches: "
        f"{len(train_matches):,}"
    )

    print(
        f"Testing matches: "
        f"{len(test_matches):,}"
    )

    print(
        f"Training rows before sampling: "
        f"{len(train_df):,}"
    )

    print(
        f"Testing rows before sampling: "
        f"{len(test_df):,}"
    )

    print(
        f"Overlapping matches: "
        f"{len(overlap)}"
    )

    return train_df, test_df


# ============================================================
# SAME ROW SAMPLE FOR EVERY EXPERIMENT
# ============================================================

def limit_rows(train_df, test_df):

    if len(train_df) > MAX_TRAIN_ROWS:

        train_df = train_df.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
        )

    if len(test_df) > MAX_TEST_ROWS:

        test_df = test_df.sample(
            n=MAX_TEST_ROWS,
            random_state=RANDOM_STATE,
        )

    print("\nROWS USED FOR ALL ABLATION EXPERIMENTS")
    print("-" * 70)

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Testing rows: "
        f"{len(test_df):,}"
    )

    return train_df, test_df


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(numeric_features):

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ============================================================
# XGBOOST
# ============================================================

def build_xgboost():

    return XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_and_evaluate(
    experiment_name,
    temporal_features,
    train_df,
    test_df,
):

    print("\n")
    print("=" * 70)

    print(
        f"TRAINING: {experiment_name}"
    )

    print("=" * 70)

    numeric_features = (
        BASE_NUMERIC_FEATURES
        + STATE_FEATURES
        + temporal_features
    )

    feature_columns = (
        numeric_features
        + CATEGORICAL_FEATURES
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET
    ]

    print(
        f"Numeric features: "
        f"{numeric_features}"
    )

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        numeric_features
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                build_xgboost(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    train_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    start = time.time()

    predictions = model.predict(
        X_test
    )

    prediction_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"MSE  : {mse:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    print(
        f"Train time: "
        f"{train_time:.2f}s"
    )

    return {
        "Experiment": experiment_name,
        "Temporal_Features": ", ".join(
            temporal_features
        )
        if temporal_features
        else "None",
        "Train_Rows": len(train_df),
        "Test_Rows": len(test_df),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Train_Time_sec": train_time,
        "Prediction_Time_sec": prediction_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    df = prepare_data(df)

    # --------------------------------------------------------
    # MATCH-LEVEL SPLIT
    # --------------------------------------------------------

    train_df, test_df = split_by_match(
        df
    )

    # --------------------------------------------------------
    # SAME ROWS FOR EVERY EXPERIMENT
    # --------------------------------------------------------

    train_df, test_df = limit_rows(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # EXPERIMENT DEFINITIONS
    # --------------------------------------------------------

    experiments = [
        (
            "A - State Only",
            [],
        ),

        (
            "B - State + Recent Scoring",
            RECENT_SCORING_FEATURES,
        ),

        (
            "C - State + Ball Quality",
            BALL_QUALITY_FEATURES,
        ),

        (
            "D - State + Wicket Momentum",
            WICKET_MOMENTUM_FEATURES,
        ),

        (
            "E - State + All Temporal",
            ALL_TEMPORAL_FEATURES,
        ),
    ]

    results = []

    # --------------------------------------------------------
    # RUN ALL EXPERIMENTS
    # --------------------------------------------------------

    for experiment_name, features in experiments:

        result = train_and_evaluate(
            experiment_name=experiment_name,
            temporal_features=features,
            train_df=train_df,
            test_df=test_df,
        )

        results.append(result)

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("ABLATION EXPERIMENT RESULTS")
    print("=" * 70)

    display_columns = [
        "Experiment",
        "Train_Rows",
        "Test_Rows",
        "MAE",
        "RMSE",
        "R2",
        "Train_Time_sec",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BASELINE FOR COMPARISON
    # --------------------------------------------------------

    state_result = results_df.iloc[0]

    state_rmse = state_result["RMSE"]
    state_mae = state_result["MAE"]
    state_r2 = state_result["R2"]

    # --------------------------------------------------------
    # IMPROVEMENT CALCULATIONS
    # --------------------------------------------------------

    results_df["RMSE_Improvement"] = (
        state_rmse
        - results_df["RMSE"]
    )

    results_df["MAE_Improvement"] = (
        state_mae
        - results_df["MAE"]
    )

    results_df["R2_Improvement"] = (
        results_df["R2"]
        - state_r2
    )

    # --------------------------------------------------------
    # PERCENTAGE IMPROVEMENT
    # --------------------------------------------------------

    results_df["RMSE_Improvement_Pct"] = (
        (
            state_rmse
            - results_df["RMSE"]
        )
        / state_rmse
        * 100
    )

    results_df["MAE_Improvement_Pct"] = (
        (
            state_mae
            - results_df["MAE"]
        )
        / state_mae
        * 100
    )

    # --------------------------------------------------------
    # IMPROVEMENT TABLE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ABLATION IMPROVEMENT OVER STATE-ONLY MODEL")
    print("=" * 70)

    improvement_columns = [
        "Experiment",
        "RMSE_Improvement",
        "RMSE_Improvement_Pct",
        "MAE_Improvement",
        "MAE_Improvement_Pct",
        "R2_Improvement",
    ]

    print(
        results_df[
            improvement_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("ABLATION EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to:\n{RESULTS_CSV}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()