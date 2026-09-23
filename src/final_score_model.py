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
    r2_score
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
    / "final_model_results.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_xgboost_model.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 100_000

TEST_SIZE = 0.20


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

def load_data():

    print("=" * 65)
    print("FINAL SCORE PREDICTION MODEL")
    print("=" * 65)

    print("\nLoading dataset...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False
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
# VALIDATE FEATURES
# ============================================================

def validate_data(df):

    print("\n")
    print("=" * 65)
    print("FEATURE VALIDATION")
    print("=" * 65)

    required = (
        ALL_FEATURES
        + [TARGET, "match_id"]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("All required columns exist.")

    # Make absolutely sure target isn't in features
    if TARGET in ALL_FEATURES:
        raise ValueError(
            "LEAKAGE ERROR: final_score is being used as a feature."
        )

    print(
        "Target excluded from features: YES"
    )

    print(
        f"Number of input features: "
        f"{len(ALL_FEATURES)}"
    )

    print(
        f"Target: {TARGET}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing data...")

    # Numeric conversion
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Categorical conversion
    for col in CATEGORICAL_FEATURES:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )

    before = len(df)

    # Remove rows with missing numerical values
    df = df.dropna(
        subset=NUMERIC_FEATURES + [TARGET]
    )

    removed = before - len(df)

    print(
        f"Rows removed because of missing values: "
        f"{removed:,}"
    )

    print(
        f"Rows available: {len(df):,}"
    )

    return df


# ============================================================
# MATCH-LEVEL TRAIN / TEST SPLIT
# ============================================================

def split_by_match(df):

    print("\n")
    print("=" * 65)
    print("MATCH-LEVEL TRAIN / TEST SPLIT")
    print("=" * 65)

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    matches = df[
        "match_id"
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
        df["match_id"].isin(train_matches)
    ].copy()

    test_df = df[
        df["match_id"].isin(test_matches)
    ].copy()

    overlap = (
        set(train_matches)
        &
        set(test_matches)
    )

    if len(overlap) != 0:
        raise ValueError(
            "DATA LEAKAGE: training and testing "
            "matches overlap."
        )

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
# LIMIT ROWS
# ============================================================

def limit_rows(train_df, test_df):

    print("\n")
    print("=" * 65)
    print("FINAL EVALUATION SAMPLE")
    print("=" * 65)

    if len(train_df) > MAX_TRAIN_ROWS:

        train_df = train_df.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE
        )

    if len(test_df) > MAX_TEST_ROWS:

        test_df = test_df.sample(
            n=MAX_TEST_ROWS,
            random_state=RANDOM_STATE
        )

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
# BUILD MODEL
# ============================================================

def build_model():

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                encoder,
                CATEGORICAL_FEATURES
            )
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
        random_state=RANDOM_STATE
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                xgb
            )
        ]
    )

    return model


# ============================================================
# MAIN TRAINING
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
    # PREPARE
    # --------------------------------------------------------

    df = prepare_data(df)

    # --------------------------------------------------------
    # SPLIT BY MATCH
    # --------------------------------------------------------

    train_df, test_df = split_by_match(df)

    # --------------------------------------------------------
    # LIMIT TO SAME SIZE USED IN EXPERIMENTS
    # --------------------------------------------------------

    train_df, test_df = limit_rows(
        train_df,
        test_df
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

    # Final safety check
    if TARGET in X_train.columns:
        raise ValueError(
            "FINAL LEAKAGE CHECK FAILED."
        )

    print("\n")
    print("=" * 65)
    print("FEATURE SET")
    print("=" * 65)

    print("\nNumeric features:")
    for feature in NUMERIC_FEATURES:
        print("  -", feature)

    print("\nCategorical features:")
    for feature in CATEGORICAL_FEATURES:
        print("  -", feature)

    print(
        f"\nTotal features: {len(ALL_FEATURES)}"
    )

    print(
        f"Target: {TARGET}"
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TRAINING FINAL XGBOOST MODEL")
    print("=" * 65)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    train_time = (
        time.time() - start
    )

    print(
        f"Training completed in "
        f"{train_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    start = time.time()

    predictions = model.predict(
        X_test
    )

    prediction_time = (
        time.time() - start
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("FINAL SCORE MODEL RESULTS")
    print("=" * 65)

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
        exist_ok=True
    )

    results = pd.DataFrame([
        {
            "Model": "Final XGBoost",
            "Train_Matches": train_df[
                "match_id"
            ].nunique(),
            "Test_Matches": test_df[
                "match_id"
            ].nunique(),
            "Train_Rows": len(train_df),
            "Test_Rows": len(test_df),
            "Features": len(ALL_FEATURES),
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Train_Time_sec": train_time,
            "Prediction_Time_sec": prediction_time
        }
    ])

    results.to_csv(
        RESULTS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("FINAL SCORE MODEL COMPLETE")
    print("=" * 65)

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