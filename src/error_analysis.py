"""
Final Model Error Analysis
==========================

Analyzes the existing final XGBoost score-prediction model.

IMPORTANT:
- Does NOT retrain the model.
- Does NOT modify any existing dataset.
- Uses the saved final XGBoost model.
- Uses a match-level train/test split.
- Produces numerical error-analysis CSV files and figures.

Outputs:
    data/processed/error_analysis_summary.csv
    data/processed/largest_prediction_errors.csv

Figures:
    results/figures/12_prediction_vs_actual.png
    results/figures/13_error_distribution.png
    results/figures/14_stage_mae.png
    results/figures/15_stage_rmse.png
    results/figures/16_score_range_mae.png
    results/figures/17_error_vs_overs_remaining.png
"""

from pathlib import Path
import pickle
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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "error_analysis_summary.csv"
)

LARGEST_ERRORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "largest_prediction_errors.csv"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# The final model experiment used:
# 2,060 training matches
#   516 testing matches
#
# Total:
# 2,576 matches
#
# Keep this random state identical to the final model experiment.
RANDOM_STATE = 42

TRAIN_MATCHES = 2060
TEST_MATCHES = 516


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("Loading engineered dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

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

    overlap = train_match_set.intersection(test_match_set)

    print(f"Overlapping matches: {len(overlap)}")

    if len(overlap) != 0:
        raise ValueError(
            "ERROR: Training and testing matches overlap."
        )

    return train_df, test_df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(test_df, model):

    print("\nPreparing test features...")

    target = "final_score"

    # ========================================================
    # COLUMNS NOT USED AS MODEL INPUT
    # ========================================================

    # IMPORTANT:
    # venue MUST remain because the final XGBoost model
    # was trained using venue as a categorical feature.
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

    # ========================================================
    # KEEP CATEGORICAL FEATURES IN THEIR ORIGINAL FORMAT
    # ========================================================
    #
    # DO NOT use:
    #
    #     .astype("category").cat.codes
    #
    # because the saved model contains a categorical
    # preprocessing pipeline that expects the original
    # categorical/string values.
    # ========================================================

    categorical_columns = []

    for col in X.columns:

        if (
            pd.api.types.is_object_dtype(X[col])
            or pd.api.types.is_categorical_dtype(X[col])
            or pd.api.types.is_string_dtype(X[col])
        ):
            categorical_columns.append(col)

    print("\nCategorical features:")

    for col in categorical_columns:
        print(f"  - {col}")

        # Keep categorical values as strings/objects.
        # Missing categorical values receive an explicit
        # string instead of numeric 0.
        X[col] = X[col].astype("object")

        X[col] = X[col].where(
            X[col].notna(),
            "__MISSING__"
        )

    # ========================================================
    # NUMERICAL FEATURES
    # ========================================================

    numerical_columns = [
        col for col in X.columns
        if col not in categorical_columns
    ]

    print("\nNumerical features:")

    for col in numerical_columns:
        print(f"  - {col}")

        # Convert infinite values to NaN.
        X[col] = X[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Fill missing numerical values.
        X[col] = X[col].fillna(0)

    # ========================================================
    # MATCH THE MODEL'S EXPECTED FEATURE ORDER
    # ========================================================

    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

        missing_features = [
            c for c in expected_features
            if c not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "The dataset is missing model features:\n"
                + "\n".join(missing_features)
            )

        # EXACT SAME FEATURE ORDER AS TRAINING
        X = X[expected_features]

    print(f"\nNumber of features: {X.shape[1]}")

    print("\nFeatures used:")

    for feature in X.columns:
        print(f"  - {feature}")

    return X, y
    
# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, X):

    print("\nGenerating predictions...")

    predictions = model.predict(X)

    predictions = np.asarray(predictions)

    print(
        f"Predictions generated: "
        f"{len(predictions):,}"
    )

    return predictions


# ============================================================
# CREATE ERROR COLUMNS
# ============================================================

def create_error_columns(test_df, predictions):

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
# OVERALL ERROR METRICS
# ============================================================

def calculate_overall_metrics(result):

    actual = result["final_score"]
    predicted = result["predicted_score"]

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

    mean_error = result["error"].mean()

    median_absolute_error = (
        result["absolute_error"].median()
    )

    maximum_absolute_error = (
        result["absolute_error"].max()
    )

    print("\n")
    print("=" * 64)
    print("OVERALL ERROR METRICS")
    print("=" * 64)

    print(f"MAE                    : {mae:.4f}")
    print(f"RMSE                   : {rmse:.4f}")
    print(f"R2                     : {r2:.4f}")
    print(f"Mean error             : {mean_error:.4f}")
    print(
        f"Median absolute error : "
        f"{median_absolute_error:.4f}"
    )
    print(
        f"Maximum absolute error: "
        f"{maximum_absolute_error:.4f}"
    )

    return {
        "group": "Overall",
        "count": len(result),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Mean_Error": mean_error,
        "Median_Absolute_Error": median_absolute_error,
        "Maximum_Absolute_Error": maximum_absolute_error,
    }


# ============================================================
# STAGE CLASSIFICATION
# ============================================================

def add_stage(result):

    def classify_stage(over):

        if over < 10:
            return "Powerplay (0-9)"

        elif over < 40:
            return "Middle (10-39)"

        else:
            return "Death (40-49)"

    result["innings_stage"] = (
        result["over"]
        .apply(classify_stage)
    )

    return result


# ============================================================
# GROUP ERROR METRICS
# ============================================================

def grouped_error_metrics(
    result,
    group_column,
    group_order=None
):

    rows = []

    grouped = result.groupby(
        group_column,
        dropna=False
    )

    for group_name, group in grouped:

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

        mean_error = group["error"].mean()

        mean_actual = actual.mean()
        mean_predicted = predicted.mean()

        rows.append({
            "group": group_name,
            "count": len(group),
            "MAE": mae,
            "RMSE": rmse,
            "Mean_Error": mean_error,
            "Mean_Actual": mean_actual,
            "Mean_Predicted": mean_predicted,
        })

    output = pd.DataFrame(rows)

    if group_order is not None:

        output["group"] = pd.Categorical(
            output["group"],
            categories=group_order,
            ordered=True
        )

        output = output.sort_values(
            "group"
        ).reset_index(drop=True)

        output["group"] = (
            output["group"].astype(str)
        )

    return output


# ============================================================
# SCORE RANGE
# ============================================================

def add_score_range(result):

    bins = [
        -np.inf,
        50,
        100,
        150,
        200,
        np.inf
    ]

    labels = [
        "<50",
        "50-99",
        "100-149",
        "150-199",
        "200+"
    ]

    result["score_range"] = pd.cut(
        result["total_score"],
        bins=bins,
        labels=labels,
        right=False
    )

    return result


# ============================================================
# OVERS REMAINING RANGE
# ============================================================

def add_overs_remaining_range(result):

    bins = [
        -np.inf,
        10,
        20,
        30,
        40,
        np.inf
    ]

    labels = [
        "0-10",
        ">10-20",
        ">20-30",
        ">30-40",
        ">40"
    ]

    result["overs_remaining_range"] = pd.cut(
        result["overs_remaining"],
        bins=bins,
        labels=labels,
        right=True
    )

    return result


# ============================================================
# LARGEST ERRORS
# ============================================================

def save_largest_errors(result):

    columns = [
        "match_id",
        "innings",
        "over",
        "ball",
        "batting_team",
        "total_score",
        "wickets",
        "overs_remaining",
        "final_score",
        "predicted_score",
        "error",
        "absolute_error",
    ]

    available_columns = [
        c for c in columns
        if c in result.columns
    ]

    largest = (
        result
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .loc[:, available_columns]
        .head(20)
    )

    largest.to_csv(
        LARGEST_ERRORS_PATH,
        index=False
    )

    print("\nLargest prediction errors saved to:")
    print(LARGEST_ERRORS_PATH)

    return largest


# ============================================================
# VISUALIZATION 1
# ============================================================

def plot_prediction_vs_actual(result):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        result["final_score"],
        result["predicted_score"],
        alpha=0.15,
        s=10
    )

    min_value = min(
        result["final_score"].min(),
        result["predicted_score"].min()
    )

    max_value = max(
        result["final_score"].max(),
        result["predicted_score"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.title("Actual vs Predicted Final Score")
    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "12_prediction_vs_actual.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# VISUALIZATION 2
# ============================================================

def plot_error_distribution(result):

    plt.figure(figsize=(8, 6))

    plt.hist(
        result["error"],
        bins=50
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Prediction Error "
        "(Predicted - Actual)"
    )

    plt.ylabel("Frequency")

    plt.title(
        "Final Score Prediction Error Distribution"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "13_error_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# VISUALIZATION 3
# ============================================================

def plot_stage_mae(stage_results):

    plt.figure(figsize=(8, 6))

    plt.bar(
        stage_results["group"],
        stage_results["MAE"]
    )

    plt.xlabel("Innings Stage")
    plt.ylabel("MAE")
    plt.title("MAE by Innings Stage")

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "14_stage_mae.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# VISUALIZATION 4
# ============================================================

def plot_stage_rmse(stage_results):

    plt.figure(figsize=(8, 6))

    plt.bar(
        stage_results["group"],
        stage_results["RMSE"]
    )

    plt.xlabel("Innings Stage")
    plt.ylabel("RMSE")
    plt.title("RMSE by Innings Stage")

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "15_stage_rmse.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# VISUALIZATION 5
# ============================================================

def plot_score_range_mae(score_results):

    plt.figure(figsize=(8, 6))

    plt.bar(
        score_results["group"],
        score_results["MAE"]
    )

    plt.xlabel("Current Score Range")
    plt.ylabel("MAE")
    plt.title("MAE by Current Innings Score")

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "16_score_range_mae.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# VISUALIZATION 6
# ============================================================

def plot_error_vs_overs_remaining(result):

    # Sample points to keep the figure manageable.
    plot_data = result.sample(
        min(100000, len(result)),
        random_state=RANDOM_STATE
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        plot_data["overs_remaining"],
        plot_data["absolute_error"],
        alpha=0.15,
        s=8
    )

    plt.xlabel("Overs Remaining")
    plt.ylabel("Absolute Prediction Error")

    plt.title(
        "Absolute Prediction Error vs Overs Remaining"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "17_error_vs_overs_remaining.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    overall,
    stage_results,
    score_results,
    overs_results
):

    rows = []

    # Overall
    rows.append({
        "analysis": "Overall",
        **overall
    })

    # Stage
    for _, row in stage_results.iterrows():

        rows.append({
            "analysis": "Innings Stage",
            **row.to_dict()
        })

    # Score range
    for _, row in score_results.iterrows():

        rows.append({
            "analysis": "Current Score Range",
            **row.to_dict()
        })

    # Overs remaining
    for _, row in overs_results.iterrows():

        rows.append({
            "analysis": "Overs Remaining",
            **row.to_dict()
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print("\nSummary saved to:")
    print(SUMMARY_PATH)

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 64)
    print("FINAL MODEL ERROR ANALYSIS")
    print("=" * 64)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    model = load_model()

    # --------------------------------------------------------
    # Match-level split
    # --------------------------------------------------------

    _, test_df = create_test_split(df)

    # --------------------------------------------------------
    # Features
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

    result = create_error_columns(
        test_df,
        predictions
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall = calculate_overall_metrics(
        result
    )

    # --------------------------------------------------------
    # Stage analysis
    # --------------------------------------------------------

    result = add_stage(result)

    stage_order = [
        "Powerplay (0-9)",
        "Middle (10-39)",
        "Death (40-49)"
    ]

    stage_results = grouped_error_metrics(
        result,
        "innings_stage",
        stage_order
    )

    print("\n")
    print("=" * 64)
    print("ERROR BY INNINGS STAGE")
    print("=" * 64)

    print(
        stage_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Score-range analysis
    # --------------------------------------------------------

    result = add_score_range(result)

    score_order = [
        "<50",
        "50-99",
        "100-149",
        "150-199",
        "200+"
    ]

    score_results = grouped_error_metrics(
        result,
        "score_range",
        score_order
    )

    print("\n")
    print("=" * 64)
    print("ERROR BY CURRENT SCORE RANGE")
    print("=" * 64)

    print(
        score_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Overs remaining analysis
    # --------------------------------------------------------

    result = add_overs_remaining_range(
        result
    )

    overs_order = [
        "0-10",
        ">10-20",
        ">20-30",
        ">30-40",
        ">40"
    ]

    overs_results = grouped_error_metrics(
        result,
        "overs_remaining_range",
        overs_order
    )

    print("\n")
    print("=" * 64)
    print("ERROR BY OVERS REMAINING")
    print("=" * 64)

    print(
        overs_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Largest errors
    # --------------------------------------------------------

    largest = save_largest_errors(
        result
    )

    print("\nTop 20 largest errors:")
    print(
        largest.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        overall,
        stage_results,
        score_results,
        overs_results
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    print("\n")
    print("=" * 64)
    print("CREATING ERROR ANALYSIS FIGURES")
    print("=" * 64)

    plot_prediction_vs_actual(
        result
    )

    plot_error_distribution(
        result
    )

    plot_stage_mae(
        stage_results
    )

    plot_stage_rmse(
        stage_results
    )

    plot_score_range_mae(
        score_results
    )

    plot_error_vs_overs_remaining(
        result
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 64)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 64)

    print("\nGenerated files:")

    print(
        f"\nSummary:\n{SUMMARY_PATH}"
    )

    print(
        f"\nLargest errors:\n"
        f"{LARGEST_ERRORS_PATH}"
    )

    print(
        f"\nFigures:\n{FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()