from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pickle


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def save_figure(filename):
    path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 1. BASELINE MODEL COMPARISON
# ============================================================

def plot_baseline_models():

    path = DATA_DIR / "baseline_model_results.csv"

    if not path.exists():
        print("Skipping baseline comparison: file not found.")
        return

    df = pd.read_csv(path)

    print("\nCreating baseline model comparison...")

    # RMSE
    plt.figure(figsize=(9, 6))

    plt.bar(
        df["Model"],
        df["RMSE"]
    )

    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.title("Baseline Score Prediction: RMSE Comparison")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    save_figure(
        "01_baseline_rmse.png"
    )

    # R2
    plt.figure(figsize=(9, 6))

    plt.bar(
        df["Model"],
        df["R2"]
    )

    plt.ylabel("R²")
    plt.xlabel("Model")
    plt.title("Baseline Score Prediction: R² Comparison")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    save_figure(
        "02_baseline_r2.png"
    )


# ============================================================
# 2. STATE FEATURE EXPERIMENT
# ============================================================

def plot_state_experiment():

    path = (
        DATA_DIR
        / "state_feature_experiment_results.csv"
    )

    if not path.exists():
        print("Skipping state experiment: file not found.")
        return

    df = pd.read_csv(path)

    print("\nCreating state feature comparison...")

    plt.figure(figsize=(8, 6))

    plt.bar(
        df["Experiment"],
        df["RMSE"]
    )

    plt.ylabel("RMSE")
    plt.xlabel("Experiment")
    plt.title("Effect of State Features on Score Prediction")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    save_figure(
        "03_state_feature_rmse.png"
    )


# ============================================================
# 3. TEMPORAL FEATURE EXPERIMENT
# ============================================================

def plot_temporal_experiment():

    path = (
        DATA_DIR
        / "temporal_experiment_results.csv"
    )

    if not path.exists():
        print("Skipping temporal experiment: file not found.")
        return

    df = pd.read_csv(path)

    print("\nCreating temporal feature comparison...")

    plt.figure(figsize=(8, 6))

    plt.bar(
        df["Experiment"],
        df["RMSE"]
    )

    plt.ylabel("RMSE")
    plt.xlabel("Experiment")
    plt.title("Effect of Temporal Features on Score Prediction")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    save_figure(
        "04_temporal_feature_rmse.png"
    )


# ============================================================
# 4. ABLATION STUDY
# ============================================================

def plot_ablation():

    path = (
        DATA_DIR
        / "ablation_results.csv"
    )

    if not path.exists():
        print("Skipping ablation study: file not found.")
        return

    df = pd.read_csv(path)

    print("\nCreating ablation study chart...")

    plt.figure(figsize=(10, 6))

    plt.bar(
        df["Experiment"],
        df["RMSE"]
    )

    plt.ylabel("RMSE")
    plt.xlabel("Feature Configuration")
    plt.title("Ablation Study: RMSE Across Feature Configurations")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    save_figure(
        "05_ablation_rmse.png"
    )

    # R2
    if "R2" in df.columns:

        plt.figure(figsize=(10, 6))

        plt.bar(
            df["Experiment"],
            df["R2"]
        )

        plt.ylabel("R²")
        plt.xlabel("Feature Configuration")
        plt.title("Ablation Study: R² Across Feature Configurations")

        plt.xticks(
            rotation=25,
            ha="right"
        )

        save_figure(
            "06_ablation_r2.png"
        )


# ============================================================
# 5. PLAYER CLASSIFICATION MODEL COMPARISON
# ============================================================

def plot_player_classification():

    path = (
        DATA_DIR
        / "player_classification_results.csv"
    )

    if not path.exists():
        print(
            "Skipping player classification comparison: "
            "file not found."
        )
        return

    df = pd.read_csv(path)

    print("\nCreating player classification comparison...")

    # Accuracy
    plt.figure(figsize=(9, 6))

    plt.bar(
        df["Model"],
        df["Accuracy"]
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Player Role Classification: Model Accuracy")

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    save_figure(
        "07_player_classification_accuracy.png"
    )

    # F1
    plt.figure(figsize=(9, 6))

    plt.bar(
        df["Model"],
        df["F1_weighted"]
    )

    plt.ylabel("Weighted F1 Score")
    plt.xlabel("Model")
    plt.title("Player Role Classification: Weighted F1")

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    save_figure(
        "08_player_classification_f1.png"
    )


# ============================================================
# 6. CROSS-VALIDATION RESULTS
# ============================================================

def plot_cross_validation():

    path = (
        DATA_DIR
        / "player_classification_cv_results.csv"
    )

    if not path.exists():
        print(
            "Skipping cross-validation chart: "
            "file not found."
        )
        return

    df = pd.read_csv(path)

    print("\nCreating cross-validation comparison...")

    plt.figure(figsize=(9, 6))

    plt.bar(
        df["Model"],
        df["Accuracy_Mean"],
        yerr=df["Accuracy_Std"],
        capsize=5
    )

    plt.ylabel(
        "Mean Accuracy"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Player Classification: 5-Fold Cross-Validation"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    save_figure(
        "09_player_classification_cv.png"
    )


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix():

    path = (
        DATA_DIR
        / "player_confusion_matrices.csv"
    )

    if not path.exists():
        print(
            "Skipping confusion matrix: "
            "file not found."
        )
        return

    df = pd.read_csv(path)

    print("\nCreating XGBoost confusion matrix...")

    # We only want XGBoost.
    xgb = df[
        df["Model"] == "XGBoost"
    ]

    if xgb.empty:
        print(
            "XGBoost confusion matrix not found."
        )
        return

    labels = sorted(
        set(xgb["Actual"]) |
        set(xgb["Predicted"])
    )

    matrix = pd.DataFrame(
        0,
        index=labels,
        columns=labels
    )

    for _, row in xgb.iterrows():

        matrix.loc[
            row["Actual"],
            row["Predicted"]
        ] = row["Count"]

    plt.figure(
        figsize=(8, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix.values,
        display_labels=matrix.index
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "XGBoost Player Role Classification Confusion Matrix"
    )

    save_figure(
        "10_xgboost_confusion_matrix.png"
    )


# ============================================================
# 8. XGBOOST FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance():

    model_path = (
        MODEL_DIR
        / "final_player_classifier.pkl"
    )

    if not model_path.exists():

        print(
            "Skipping feature importance: "
            "model file not found."
        )

        return

    print(
        "\nCreating XGBoost feature importance..."
    )

    with open(
        model_path,
        "rb"
    ) as file:

        pipeline = pickle.load(
            file
        )

    feature_names = [
        "innings",
        "runs",
        "average",
        "strike_rate",
        "wickets",
        "economy",
        "bowling_average",
        "bowling_strike_rate",
        "batting_basra",
        "bowling_basra"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "XGBoost Player Classification Feature Importance"
    )

    save_figure(
        "11_player_feature_importance.png"
    )


# ============================================================
# 9. FINAL MODEL SUMMARY
# ============================================================

def create_summary():

    print("\n")
    print("=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)

    files = sorted(
        OUTPUT_DIR.glob("*.png")
    )

    print(
        f"\nFigures created: {len(files)}"
    )

    for file in files:

        print(
            f"  - {file.name}"
        )

    print(
        f"\nSaved to:\n{OUTPUT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ML PROJECT - RESULTS VISUALIZATION")
    print("=" * 70)

    plot_baseline_models()

    plot_state_experiment()

    plot_temporal_experiment()

    plot_ablation()

    plot_player_classification()

    plot_cross_validation()

    plot_confusion_matrix()

    plot_feature_importance()

    create_summary()


if __name__ == "__main__":
    main()