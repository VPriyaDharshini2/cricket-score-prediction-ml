import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_score_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_xgboost_model.pkl"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "short_innings_robustness.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

OUTPUT_FIGURE = (
    FIGURES_DIR
    / "18_error_by_innings_length.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_MATCHES = 2060
TEST_MATCHES = 516


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("SHORT-INNINGS ROBUSTNESS ANALYSIS")
    print("=" * 70)

    print("\nLoading temporal dataset...")

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Unique matches: {df['match_id'].nunique():,}")

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading saved XGBoost model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded: {MODEL_PATH}")

    return model


# ============================================================
# CREATE MATCH-LEVEL TEST SPLIT
# ============================================================

def create_test_split(df):

    print("\nCreating match-level test split...")

    matches = (
        df["match_id"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    rng = np.random.RandomState(RANDOM_STATE)

    shuffled_matches = matches.copy()

    rng.shuffle(shuffled_matches)

    train_matches = shuffled_matches[:TRAIN_MATCHES]
    test_matches = shuffled_matches[TRAIN_MATCHES:]

    train_match_set = set(train_matches)
    test_match_set = set(test_matches)

    train_df = df[
        df["match_id"].isin(train_match_set)
    ].copy()

    test_df = df[
        df["match_id"].isin(test_match_set)
    ].copy()

    print(f"Training matches: {len(train_match_set):,}")
    print(f"Testing matches : {len(test_match_set):,}")

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows : {len(test_df):,}")

    overlap = train_match_set.intersection(
        test_match_set
    )

    print(f"Overlapping matches: {len(overlap)}")

    if len(overlap) != 0:
        raise ValueError(
            "ERROR: Training and testing matches overlap."
        )

    return test_df


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_features(test_df, model):

    print("\nPreparing test features...")

    target = "final_score"

    # IMPORTANT:
    # Do NOT remove venue.
    # The saved final model expects venue as a feature.

    drop_columns = [
        target,
        "match_id",
        "date",
        "city",
        "team_1",
        "team_2",
        "toss_winner",
        "toss_decision",
        "actual_delivery",
        "player_out",
        "wicket_kind",
        "wicket_fielders",
    ]

    available_drop = [
        c for c in drop_columns
        if c in test_df.columns
    ]

    X = test_df.drop(
        columns=available_drop,
        errors="ignore"
    ).copy()

    y = test_df[target].copy()

    # --------------------------------------------------------
    # Replace infinite numerical values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Preserve categorical columns as strings
    # --------------------------------------------------------
    #
    # The saved model contains preprocessing for categorical
    # features. Therefore, do NOT convert categorical values
    # into category codes.
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = [
        c for c in X.columns
        if c not in categorical_columns
    ]

    for col in categorical_columns:

        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    for col in numerical_columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        X[col] = X[col].fillna(0)

    # --------------------------------------------------------
    # Match model feature order
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

        print("\nModel expects these features:")

        for feature in expected_features:
            print(f"  - {feature}")

        missing_features = [
            c
            for c in expected_features
            if c not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Dataset is missing model features:\n"
                + "\n".join(missing_features)
            )

        X = X[expected_features]

    print(
        f"\nNumber of features used: {X.shape[1]}"
    )

    return X, y


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, X):

    print("\nGenerating predictions...")

    predictions = model.predict(X)

    predictions = np.asarray(
        predictions
    )

    print(
        f"Predictions generated: "
        f"{len(predictions):,}"
    )

    return predictions


# ============================================================
# CREATE ERROR DATAFRAME
# ============================================================

def create_error_dataframe(
    test_df,
    predictions
):

    print("\nCreating prediction-error dataframe...")

    result = test_df.copy()

    result["predicted_score"] = predictions

    result["error"] = (
        result["predicted_score"]
        - result["final_score"]
    )

    result["absolute_error"] = (
        result["error"].abs()
    )

    return result


# ============================================================
# DETERMINE INNINGS LENGTH
# ============================================================

def add_innings_length(result):

    print("\nDetermining actual innings length...")

    # Each match + innings has a final row.
    # The maximum over represents the final over index.
    #
    # Cricsheet over numbering is zero-based.
    # Therefore:
    #
    # overs_completed = max_over + 1

    innings_info = (
        result
        .groupby(
            ["match_id", "innings"],
            as_index=False
        )
        .agg(
            final_score=("final_score", "max"),
            max_over=("over", "max"),
            batting_team=("batting_team", "first")
        )
    )

    innings_info["overs_completed"] = (
        innings_info["max_over"] + 1
    )

    # --------------------------------------------------------
    # Create innings-length categories
    # --------------------------------------------------------

    conditions = [
        innings_info["overs_completed"] < 10,

        (
            (innings_info["overs_completed"] >= 10)
            &
            (innings_info["overs_completed"] < 20)
        ),

        (
            (innings_info["overs_completed"] >= 20)
            &
            (innings_info["overs_completed"] < 30)
        ),

        innings_info["overs_completed"] >= 30
    ]

    labels = [
        "Very Short (<10 overs)",
        "Short (10-20 overs)",
        "Medium (20-30 overs)",
        "Long (30+ overs)"
    ]

    innings_info["innings_length_group"] = np.select(
        conditions,
        labels,
        default="Unknown"
    )

    # --------------------------------------------------------
    # Merge innings information back
    # --------------------------------------------------------

    result = result.merge(
        innings_info[
            [
                "match_id",
                "innings",
                "overs_completed",
                "innings_length_group"
            ]
        ],
        on=["match_id", "innings"],
        how="left"
    )

    print("\nInnings-length distribution:")

    distribution = (
        innings_info[
            "innings_length_group"
        ]
        .value_counts()
    )

    group_order = [
        "Very Short (<10 overs)",
        "Short (10-20 overs)",
        "Medium (20-30 overs)",
        "Long (30+ overs)"
    ]

    for group in group_order:

        count = distribution.get(
            group,
            0
        )

        print(
            f"  {group}: {count:,} innings"
        )

    return result


# ============================================================
# CALCULATE ROBUSTNESS METRICS
# ============================================================

def calculate_robustness_metrics(result):

    print(
        "\nCalculating metrics by innings length..."
    )

    group_order = [
        "Very Short (<10 overs)",
        "Short (10-20 overs)",
        "Medium (20-30 overs)",
        "Long (30+ overs)"
    ]

    rows = []

    for group_name in group_order:

        group = result[
            result["innings_length_group"]
            == group_name
        ].copy()

        if len(group) == 0:
            continue

        actual = group["final_score"]
        predicted = group["predicted_score"]

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        r2 = r2_score(
            actual,
            predicted
        )

        mean_error = (
            group["error"].mean()
        )

        mean_actual = (
            actual.mean()
        )

        mean_predicted = (
            predicted.mean()
        )

        unique_innings = (
            group[
                ["match_id", "innings"]
            ]
            .drop_duplicates()
            .shape[0]
        )

        rows.append({

            "innings_length_group":
                group_name,

            "innings_count":
                unique_innings,

            "prediction_rows":
                len(group),

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,

            "mean_error":
                mean_error,

            "mean_final_score":
                mean_actual,

            "mean_predicted_score":
                mean_predicted
        })

    results = pd.DataFrame(rows)

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    os.makedirs(
        OUTPUT_CSV.parent,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nSaved results to:\n"
        f"{OUTPUT_CSV}"
    )


# ============================================================
# CREATE FIGURE
# ============================================================

def create_figure(results):

    print("\nCreating visualization...")

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results["innings_length_group"],
        results["MAE"],
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Actual Innings Length"
    )

    plt.ylabel(
        "Mean Absolute Error (MAE)"
    )

    plt.title(
        "Prediction Error by Actual Innings Length"
    )

    plt.xticks(
        rotation=20
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved figure to:\n"
        f"{OUTPUT_FIGURE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    model = load_model()

    # --------------------------------------------------------
    # Reconstruct exact test split
    # --------------------------------------------------------

    test_df = create_test_split(
        df
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_test, y_test = prepare_features(
        test_df,
        model
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = generate_predictions(
        model,
        X_test
    )

    # --------------------------------------------------------
    # Error dataframe
    # --------------------------------------------------------

    result = create_error_dataframe(
        test_df,
        predictions
    )

    # --------------------------------------------------------
    # Add innings length
    # --------------------------------------------------------

    result = add_innings_length(
        result
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    results = calculate_robustness_metrics(
        result
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SHORT-INNINGS ROBUSTNESS RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    create_figure(
        results
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()