"""
Experiment 3: Temporal and momentum features for cricket
score prediction.

Controlled comparison:

Model A:
    Baseline + State Features

Model B:
    Baseline + State Features
    +
    Temporal/Momentum Features

Temporal features:
    - last_3_overs_score
    - last_10_overs_score
    - boundary_rate
    - dot_ball_rate
    - recent_5_over_wickets

Both models use:
    - identical eligible rows
    - identical match-level train/test split
    - identical row sampling
    - identical XGBoost configuration

This allows us to measure the effect of the temporal
features separately.
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
    / "temporal_experiment_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 100_000


# ============================================================
# FEATURE DEFINITIONS
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

TEMPORAL_FEATURES = [
    "last_3_overs_score",
    "last_10_overs_score",
    "boundary_rate",
    "dot_ball_rate",
    "recent_5_over_wickets",
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
        f"{df['match_id'].nunique():,}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    all_numeric = (
        BASE_NUMERIC_FEATURES
        + STATE_FEATURES
        + TEMPORAL_FEATURES
        + [TARGET]
    )

    required = (
        all_numeric
        + CATEGORICAL_FEATURES
        + [GROUP_COLUMN]
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
    # Numeric columns
    # --------------------------------------------------------

    for column in all_numeric:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # CRR is undefined on the first delivery of some innings.
    #
    # Remove these rows so both models use exactly the same
    # eligible population.
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=(
            BASE_NUMERIC_FEATURES
            + STATE_FEATURES
            + TEMPORAL_FEATURES
            + [TARGET]
        )
    )

    removed = before - len(df)

    print(
        f"\nRows removed because required "
        f"features were unavailable: {removed:,}"
    )

    print(
        f"Rows available for experiment: "
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
        df[GROUP_COLUMN].isin(
            train_matches
        )
    ].copy()

    test_df = df[
        df[GROUP_COLUMN].isin(
            test_matches
        )
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
        f"Testing matches:  "
        f"{len(test_matches):,}"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Testing rows:  "
        f"{len(test_df):,}"
    )

    print(
        f"Overlapping matches: "
        f"{len(overlap)}"
    )

    return train_df, test_df


# ============================================================
# LIMIT ROWS
# ============================================================

def limit_rows(
    train_df,
    test_df,
):

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

    print("\nROWS USED FOR CONTROLLED EXPERIMENT")
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

def build_preprocessor(
    numeric_features,
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
# XGBOOST MODEL
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
    numeric_features,
    train_df,
    test_df,
):

    print("\n")
    print("=" * 70)

    print(
        f"TRAINING: {experiment_name}"
    )

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

    # --------------------------------------------------------
    # Training
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
    # Prediction
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
    # SAME ROW SAMPLE FOR BOTH MODELS
    # --------------------------------------------------------

    train_df, test_df = limit_rows(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # MODEL A
    #
    # Baseline + state features
    # --------------------------------------------------------

    state_numeric_features = (
        BASE_NUMERIC_FEATURES
        + STATE_FEATURES
    )

    state_result = train_and_evaluate(
        experiment_name=
            "State Features",
        numeric_features=
            state_numeric_features,
        train_df=train_df,
        test_df=test_df,
    )

    # --------------------------------------------------------
    # MODEL B
    #
    # Baseline + state + temporal features
    # --------------------------------------------------------

    full_numeric_features = (
        BASE_NUMERIC_FEATURES
        + STATE_FEATURES
        + TEMPORAL_FEATURES
    )

    temporal_result = train_and_evaluate(
        experiment_name=
            "State + Temporal Features",
        numeric_features=
            full_numeric_features,
        train_df=train_df,
        test_df=test_df,
    )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            state_result,
            temporal_result,
        ]
    )

    print("\n")
    print("=" * 70)
    print("TEMPORAL FEATURE EXPERIMENT")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FEATURE IMPACT
    # --------------------------------------------------------

    state_rmse = (
        state_result["RMSE"]
    )

    temporal_rmse = (
        temporal_result["RMSE"]
    )

    state_mae = (
        state_result["MAE"]
    )

    temporal_mae = (
        temporal_result["MAE"]
    )

    state_r2 = (
        state_result["R2"]
    )

    temporal_r2 = (
        temporal_result["R2"]
    )

    rmse_change = (
        state_rmse
        - temporal_rmse
    )

    mae_change = (
        state_mae
        - temporal_mae
    )

    r2_change = (
        temporal_r2
        - state_r2
    )

    print("\n")
    print("=" * 70)
    print("TEMPORAL FEATURE IMPACT")
    print("=" * 70)

    print(
        f"RMSE change: "
        f"{rmse_change:+.4f}"
    )

    print(
        f"MAE change : "
        f"{mae_change:+.4f}"
    )

    print(
        f"R2 change  : "
        f"{r2_change:+.4f}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_CSV,
        index=False,
    )

    print("\nResults saved to:")

    print(
        RESULTS_CSV
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()