"""
Experiment 2: Match-state features for cricket score prediction.

Purpose:
    Compare the original baseline feature set against the same feature
    set extended with:

        - current_run_rate
        - overs_remaining

Experimental control:
    - Same dataset
    - Same eligible rows
    - Same match-level train/test split
    - Same sampled rows
    - Same XGBoost configuration

This makes the comparison an apples-to-apples feature experiment.
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
    / "score_prediction_dataset.csv"
)

RESULTS_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "state_feature_experiment_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 100_000


# ============================================================
# FEATURE GROUPS
# ============================================================

BASE_NUMERIC_FEATURES = [
    "innings",
    "over",
    "ball",
    "total_score",
    "wickets",
    "last_5_overs_score",
]

STATE_NUMERIC_FEATURES = [
    "current_run_rate",
    "overs_remaining",
]

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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Matches: {df['match_id'].nunique():,}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required = (
        BASE_NUMERIC_FEATURES
        + STATE_NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET, "match_id"]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[required].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = (
        BASE_NUMERIC_FEATURES
        + STATE_NUMERIC_FEATURES
        + [TARGET]
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Categorical conversion
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # Remove rows where CRR is undefined.
    #
    # This creates the SAME eligible population for both
    # baseline and modified models.
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=(
            BASE_NUMERIC_FEATURES
            + STATE_NUMERIC_FEATURES
            + [TARGET]
        )
    )

    removed = before - len(df)

    print(
        f"\nRows removed because current_run_rate "
        f"is unavailable: {removed:,}"
    )

    print(
        f"Rows available for controlled experiment: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# MATCH-LEVEL SPLIT
# ============================================================

def split_by_match(df):

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    matches = df["match_id"].unique()

    rng.shuffle(matches)

    split_index = int(
        len(matches) * (1 - TEST_SIZE)
    )

    train_matches = matches[:split_index]
    test_matches = matches[split_index:]

    train_df = df[
        df["match_id"].isin(train_matches)
    ].copy()

    test_df = df[
        df["match_id"].isin(test_matches)
    ].copy()

    overlap = (
        set(train_matches)
        & set(test_matches)
    )

    if overlap:
        raise RuntimeError(
            "Match leakage detected."
        )

    print("\nMATCH-LEVEL SPLIT")
    print("-" * 70)

    print(
        f"Training matches: {len(train_matches):,}"
    )

    print(
        f"Testing matches:  {len(test_matches):,}"
    )

    print(
        f"Training rows:    {len(train_df):,}"
    )

    print(
        f"Testing rows:     {len(test_df):,}"
    )

    print(
        f"Overlapping matches: {len(overlap)}"
    )

    return train_df, test_df


# ============================================================
# LIMIT ROWS
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

    print("\nROWS USED")
    print("-" * 70)

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Testing rows:  {len(test_df):,}"
    )

    return train_df, test_df


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    numeric_features
):

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
    name,
    numeric_features,
    train_df,
    test_df,
):

    print("\n")
    print("=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

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

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    train_time = (
        time.time() - start
    )

    start = time.time()

    predictions = model.predict(
        X_test
    )

    prediction_time = (
        time.time() - start
    )

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
        f"Train time: {train_time:.2f}s"
    )

    return {
        "Experiment": name,
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
    # LOAD + PREPARE
    # --------------------------------------------------------

    df = load_data()

    df = prepare_data(df)

    # --------------------------------------------------------
    # MATCH-LEVEL SPLIT
    # --------------------------------------------------------

    train_df, test_df = split_by_match(
        df
    )

    # --------------------------------------------------------
    # SAME SAMPLE FOR BOTH EXPERIMENTS
    # --------------------------------------------------------

    train_df, test_df = limit_rows(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # BASELINE EXPERIMENT
    # --------------------------------------------------------

    baseline_result = train_and_evaluate(
        name="Baseline XGBoost",
        numeric_features=BASE_NUMERIC_FEATURES,
        train_df=train_df,
        test_df=test_df,
    )

    # --------------------------------------------------------
    # MODIFIED EXPERIMENT
    # --------------------------------------------------------

    modified_numeric_features = (
        BASE_NUMERIC_FEATURES
        + STATE_NUMERIC_FEATURES
    )

    modified_result = train_and_evaluate(
        name="Baseline + State Features",
        numeric_features=modified_numeric_features,
        train_df=train_df,
        test_df=test_df,
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            baseline_result,
            modified_result,
        ]
    )

    print("\n")
    print("=" * 70)
    print("FEATURE EXPERIMENT COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    baseline_rmse = (
        baseline_result["RMSE"]
    )

    modified_rmse = (
        modified_result["RMSE"]
    )

    baseline_mae = (
        baseline_result["MAE"]
    )

    modified_mae = (
        modified_result["MAE"]
    )

    rmse_change = (
        baseline_rmse
        - modified_rmse
    )

    mae_change = (
        baseline_mae
        - modified_mae
    )

    r2_change = (
        modified_result["R2"]
        - baseline_result["R2"]
    )

    print("\n")
    print("=" * 70)
    print("FEATURE IMPACT")
    print("=" * 70)

    print(
        f"RMSE change: {rmse_change:+.4f}"
    )

    print(
        f"MAE change : {mae_change:+.4f}"
    )

    print(
        f"R2 change  : {r2_change:+.4f}"
    )

    results.to_csv(
        RESULTS_CSV,
        index=False,
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_CSV
    )


if __name__ == "__main__":
    main()