"""
Chronological evaluation of the final ODI score prediction model.

Purpose:
    Evaluate the same temporal-feature XGBoost score prediction pipeline
    using a genuinely chronological train/test split.

Existing randomized evaluation is NOT modified.

Split:
    Older match dates -> training
    Newer match dates -> testing

Important:
    Complete matches are kept together.
    No match can appear in both training and testing.

Outputs:
    data/processed/chronological_evaluation_results.csv
    models/chronological_xgboost_model.pkl
"""

from __future__ import annotations

from pathlib import Path
import time
import pickle

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
    / "chronological_evaluation_results.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "chronological_xgboost_model.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 100_000


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "innings",
    "over",
    "ball",
    "total_score",
    "wickets",
    "last_5_overs_score",
    "current_run_rate",
    "overs_remaining",

    # Temporal features
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

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    print("=" * 70)
    print("CHRONOLOGICAL SCORE PREDICTION EVALUATION")
    print("=" * 70)

    print("\nLoading temporal dataset...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
    )

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df: pd.DataFrame) -> pd.DataFrame:

    print("\n")
    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    required_columns = (
        ALL_FEATURES
        + [
            TARGET,
            "match_id",
            "date",
        ]
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

    print("All required columns exist.")

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    missing_dates = int(
        df["date"].isna().sum()
    )

    print(
        f"Rows with missing dates: "
        f"{missing_dates:,}"
    )

    if missing_dates > 0:
        raise ValueError(
            "Missing or invalid match dates detected."
        )

    # --------------------------------------------------------
    # Check one date per match
    # --------------------------------------------------------

    dates_per_match = (
        df.groupby("match_id")["date"]
        .nunique()
    )

    inconsistent_matches = (
        dates_per_match > 1
    ).sum()

    print(
        f"Matches with multiple dates: "
        f"{inconsistent_matches:,}"
    )

    if inconsistent_matches > 0:
        raise ValueError(
            "Some matches contain multiple dates. "
            "Chronological splitting cannot proceed safely."
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES + [TARGET]:
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
    # Remove missing numerical rows
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=NUMERIC_FEATURES + [TARGET]
    ).copy()

    removed = before - len(df)

    print(
        f"Rows removed due to missing numeric values: "
        f"{removed:,}"
    )

    print(
        f"Rows available: {len(df):,}"
    )

    return df


# ============================================================
# CHRONOLOGICAL MATCH-LEVEL SPLIT
# ============================================================

def split_chronologically(
    df: pd.DataFrame,
):
    """
    Split complete matches chronologically.

    The split is performed at the DATE level rather than the
    delivery level.

    This ensures:

        max(training date) < min(test date)

    Therefore, no future match date can appear in training.
    """

    print("\n")
    print("=" * 70)
    print("CHRONOLOGICAL MATCH-LEVEL SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Create one row per match
    # --------------------------------------------------------

    match_dates = (
        df[
            [
                "match_id",
                "date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "date",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    total_matches = len(match_dates)

    print(
        f"Total matches: {total_matches:,}"
    )

    print(
        f"First match date: "
        f"{match_dates['date'].min().date()}"
    )

    print(
        f"Last match date: "
        f"{match_dates['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Find approximately 80/20 chronological boundary
    # --------------------------------------------------------

    target_train_matches = int(
        total_matches * (1 - TEST_SIZE)
    )

    # Initial candidate date boundary
    candidate_date = (
        match_dates.iloc[target_train_matches - 1]["date"]
    )

    # All matches on or before candidate date go to train.
    # All later matches go to test.
    train_match_dates = match_dates[
        match_dates["date"] <= candidate_date
    ]

    test_match_dates = match_dates[
        match_dates["date"] > candidate_date
    ]

    # --------------------------------------------------------
    # If the candidate creates an extremely unbalanced split,
    # use the closest possible chronological boundary.
    # --------------------------------------------------------

    if len(train_match_dates) == 0:
        raise ValueError(
            "Chronological training set is empty."
        )

    if len(test_match_dates) == 0:
        raise ValueError(
            "Chronological test set is empty."
        )

    train_matches = train_match_dates[
        "match_id"
    ].tolist()

    test_matches = test_match_dates[
        "match_id"
    ].tolist()

    # --------------------------------------------------------
    # Create row-level datasets
    # --------------------------------------------------------

    train_df = df[
        df["match_id"].isin(train_matches)
    ].copy()

    test_df = df[
        df["match_id"].isin(test_matches)
    ].copy()

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    overlap = (
        set(train_matches)
        &
        set(test_matches)
    )

    if len(overlap) != 0:
        raise ValueError(
            "DATA LEAKAGE: training and testing matches overlap."
        )

    max_train_date = train_df["date"].max()
    min_test_date = test_df["date"].min()

    if max_train_date >= min_test_date:
        raise ValueError(
            "CHRONOLOGICAL ORDER CHECK FAILED."
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        f"\nTraining matches: "
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
        f"\nTraining date range:"
        f"\n  {train_df['date'].min().date()}"
        f" → {train_df['date'].max().date()}"
    )

    print(
        f"\nTesting date range:"
        f"\n  {test_df['date'].min().date()}"
        f" → {test_df['date'].max().date()}"
    )

    print(
        f"\nChronological boundary:"
        f"\n  {max_train_date.date()}"
        f" < "
        f"{min_test_date.date()}"
    )

    print(
        f"\nOverlapping matches: "
        f"{len(overlap)}"
    )

    return train_df, test_df


# ============================================================
# LIMIT ROWS
# ============================================================

def limit_rows(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Keep the same row limits used by the existing final
    score-prediction experiment.

    IMPORTANT:
        Sampling rows does NOT change the chronological
        train/test separation because the match sets have
        already been separated.
    """

    print("\n")
    print("=" * 70)
    print("CHRONOLOGICAL EVALUATION SAMPLE")
    print("=" * 70)

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

    print(
        f"Training rows used: "
        f"{len(train_df):,}"
    )

    print(
        f"Testing rows used: "
        f"{len(test_df):,}"
    )

    return train_df, test_df


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                encoder,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                xgb,
            ),
        ]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    df = validate_data(df)

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    train_df, test_df = split_chronologically(
        df
    )

    # --------------------------------------------------------
    # LIMIT ROWS
    # --------------------------------------------------------

    train_df, test_df = limit_rows(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X_train = train_df[
        ALL_FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        ALL_FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # FINAL LEAKAGE CHECK
    # --------------------------------------------------------

    if TARGET in X_train.columns:
        raise ValueError(
            "FINAL LEAKAGE CHECK FAILED."
        )

    if TARGET in X_test.columns:
        raise ValueError(
            "FINAL LEAKAGE CHECK FAILED."
        )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BUILDING XGBOOST MODEL")
    print("=" * 70)

    model = build_model()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING CHRONOLOGICAL XGBOOST MODEL")
    print("=" * 70)

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    train_time = time.time() - start

    print(
        f"Training completed in "
        f"{train_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    print("\nGenerating chronological predictions...")

    start = time.time()

    predictions = model.predict(
        X_test
    )

    prediction_time = time.time() - start

    # --------------------------------------------------------
    # METRICS
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

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CHRONOLOGICAL SCORE MODEL RESULTS")
    print("=" * 70)

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
        f"Training time   : "
        f"{train_time:.2f} sec"
    )

    print(
        f"Prediction time : "
        f"{prediction_time:.2f} sec"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        [
            {
                "Evaluation": "Chronological",
                "Model": "XGBoost",
                "Train_Matches": train_df[
                    "match_id"
                ].nunique(),
                "Test_Matches": test_df[
                    "match_id"
                ].nunique(),
                "Train_Rows": len(train_df),
                "Test_Rows": len(test_df),
                "Train_Start_Date": train_df[
                    "date"
                ].min().date(),
                "Train_End_Date": train_df[
                    "date"
                ].max().date(),
                "Test_Start_Date": test_df[
                    "date"
                ].min().date(),
                "Test_End_Date": test_df[
                    "date"
                ].max().date(),
                "Features": len(ALL_FEATURES),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Train_Time_sec": train_time,
                "Prediction_Time_sec": prediction_time,
            }
        ]
    )

    results.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            model,
            file,
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CHRONOLOGICAL EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_CSV}"
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()