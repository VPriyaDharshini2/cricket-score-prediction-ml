"""
Baseline ML models for cricket score prediction.

This is the first model-training stage of the project.

Models:
- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost Regression

Evaluation:
- MAE
- MSE
- RMSE
- R2

The train/test split is performed at MATCH level so that deliveries
from the same match cannot appear in both training and testing sets.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

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
    / "baseline_model_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Main models
MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 100_000

# RBF-SVR is computationally expensive.
SVM_TRAIN_ROWS = 10_000
SVM_TEST_ROWS = 20_000


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
    print("LOADING ENGINEERED DATASET")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Matches: {df['match_id'].nunique():,}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET, "match_id"]
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required].copy()

    # Target must exist.
    df = df.dropna(
        subset=[TARGET]
    )

    # Categorical columns.
    for col in CATEGORICAL_FEATURES:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )

    # Numeric columns.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Remove rows with unavailable numeric predictors.
    df = df.dropna(
        subset=NUMERIC_FEATURES
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

    overlap = set(train_matches) & set(test_matches)

    if overlap:
        raise RuntimeError(
            "Data leakage detected: "
            "some matches occur in both sets."
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
# LIMIT MAIN DATASET
# ============================================================

def limit_main_rows(train_df, test_df):

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

    print("\nROWS USED FOR MAIN MODELS")
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

def build_preprocessor():

    numeric_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            )
        ]
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {

        "Linear Regression": LinearRegression(),

        "SVM": SVR(
            kernel="rbf",
            C=10,
            epsilon=0.1,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    train_time = time.time() - start

    start = time.time()

    predictions = model.predict(
        X_test
    )

    prediction_time = time.time() - start

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

    return {
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

    df = prepare_data(df)

    print(
        f"\nRows after preparation: {len(df):,}"
    )

    # --------------------------------------------------------
    # MATCH-LEVEL SPLIT
    # --------------------------------------------------------

    train_df, test_df = split_by_match(
        df
    )

    # --------------------------------------------------------
    # MAIN MODEL DATA
    # --------------------------------------------------------

    train_main, test_main = limit_main_rows(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # SVM DATA
    # --------------------------------------------------------

    svm_train = train_main.sample(
        n=min(
            SVM_TRAIN_ROWS,
            len(train_main),
        ),
        random_state=RANDOM_STATE,
    )

    svm_test = test_main.sample(
        n=min(
            SVM_TEST_ROWS,
            len(test_main),
        ),
        random_state=RANDOM_STATE,
    )

    print("\nSVM SUBSET")
    print("-" * 70)

    print(
        f"SVM training rows: {len(svm_train):,}"
    )

    print(
        f"SVM testing rows:  {len(svm_test):,}"
    )

    # --------------------------------------------------------
    # MODEL LOOP
    # --------------------------------------------------------

    models = build_models()

    results = []

    print("\n")
    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    for name, estimator in models.items():

        print(
            f"\nTraining {name}..."
        )

        # Select appropriate dataset.
        if name == "SVM":
            train_data = svm_train
            test_data = svm_test
        else:
            train_data = train_main
            test_data = test_main

        X_train = train_data[
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
        ]

        y_train = train_data[TARGET]

        X_test = test_data[
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
        ]

        y_test = test_data[TARGET]

        preprocessor = build_preprocessor()

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        try:

            metrics = evaluate_model(
                pipeline,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            row = {
                "Model": name,
                "Train_Rows": len(train_data),
                "Test_Rows": len(test_data),
                **metrics,
            }

            results.append(row)

            print(
                f"MAE  : {metrics['MAE']:.4f}"
            )

            print(
                f"MSE  : {metrics['MSE']:.4f}"
            )

            print(
                f"RMSE : {metrics['RMSE']:.4f}"
            )

            print(
                f"R2   : {metrics['R2']:.4f}"
            )

            print(
                f"Train time: "
                f"{metrics['Train_Time_sec']:.2f}s"
            )

        except Exception as e:

            print(
                f"ERROR training {name}:"
            )

            print(e)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise RuntimeError(
            "No models completed successfully."
        )

    results_df = results_df.sort_values(
        "RMSE"
    )

    print("\n")
    print("=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
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