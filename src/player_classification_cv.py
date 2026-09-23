from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_classification_dataset.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_classification_cv_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "innings",
    "runs",
    "average",
    "strike_rate",
    "wickets",
    "economy",
    "bowling_average",
    "bowling_strike_rate",
    "batting_basra",
    "bowling_basra",
]

TARGET = "role"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("PLAYER CLASSIFICATION - 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    print("\nLoading dataset...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print(
        f"Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Convert features to numeric
    for column in FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Encode target consistently
    class_names = sorted(
        y.unique()
    )

    class_to_number = {
        name: index
        for index, name in enumerate(class_names)
    }

    y_encoded = y.map(
        class_to_number
    )

    print("\nClass mapping:")

    for name, number in class_to_number.items():

        print(
            f"  {number} -> {name}"
        )

    return X, y_encoded


# ============================================================
# BUILD MODELS
# ============================================================

def build_models():

    scaled_preprocessor = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    unscaled_preprocessor = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    models = {

        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    scaled_preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

        "Naive Bayes": Pipeline(
            steps=[
                (
                    "preprocessor",
                    scaled_preprocessor
                ),
                (
                    "model",
                    GaussianNB()
                )
            ]
        ),

        "SVM": Pipeline(
            steps=[
                (
                    "preprocessor",
                    scaled_preprocessor
                ),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    unscaled_preprocessor
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        ),

        "XGBoost": Pipeline(
            steps=[
                (
                    "preprocessor",
                    unscaled_preprocessor
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="multi:softmax",
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        )
    }

    return models


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    models,
    X,
    y
):

    print("\n")
    print("=" * 70)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for name, model in models.items():

        print("\n")
        print("-" * 70)
        print(f"VALIDATING: {name}")
        print("-" * 70)

        start = time.time()

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=[
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted"
            ],
            n_jobs=1,
            return_train_score=False
        )

        elapsed = (
            time.time() - start
        )

        accuracy_mean = np.mean(
            scores["test_accuracy"]
        )

        accuracy_std = np.std(
            scores["test_accuracy"]
        )

        precision_mean = np.mean(
            scores[
                "test_precision_weighted"
            ]
        )

        precision_std = np.std(
            scores[
                "test_precision_weighted"
            ]
        )

        recall_mean = np.mean(
            scores[
                "test_recall_weighted"
            ]
        )

        recall_std = np.std(
            scores[
                "test_recall_weighted"
            ]
        )

        f1_mean = np.mean(
            scores[
                "test_f1_weighted"
            ]
        )

        f1_std = np.std(
            scores[
                "test_f1_weighted"
            ]
        )

        print(
            f"Accuracy : "
            f"{accuracy_mean:.4f} "
            f"+/- {accuracy_std:.4f}"
        )

        print(
            f"Precision: "
            f"{precision_mean:.4f} "
            f"+/- {precision_std:.4f}"
        )

        print(
            f"Recall   : "
            f"{recall_mean:.4f} "
            f"+/- {recall_std:.4f}"
        )

        print(
            f"F1 Score : "
            f"{f1_mean:.4f} "
            f"+/- {f1_std:.4f}"
        )

        print(
            f"CV time  : "
            f"{elapsed:.2f}s"
        )

        results.append(
            {
                "Model": name,

                "Folds": N_SPLITS,

                "Accuracy_Mean":
                    accuracy_mean,

                "Accuracy_Std":
                    accuracy_std,

                "Precision_Mean":
                    precision_mean,

                "Precision_Std":
                    precision_std,

                "Recall_Mean":
                    recall_mean,

                "Recall_Std":
                    recall_std,

                "F1_Mean":
                    f1_mean,

                "F1_Std":
                    f1_std,

                "CV_Time_sec":
                    elapsed
            }
        )

    return pd.DataFrame(
        results
    )


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

    X, y = prepare_data(
        df
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = build_models()

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    results_df = run_cross_validation(
        models,
        X,
        y
    )

    # --------------------------------------------------------
    # SORT BY F1
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            "F1_Mean",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    display_columns = [
        "Model",
        "Accuracy_Mean",
        "Accuracy_Std",
        "Precision_Mean",
        "Recall_Mean",
        "F1_Mean",
        "F1_Std",
        "CV_Time_sec"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = results_df.iloc[0]

    print("\n")
    print("=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"\nBest model by mean weighted F1: "
        f"{best_model['Model']}"
    )

    print(
        f"Mean Accuracy: "
        f"{best_model['Accuracy_Mean']:.4f}"
    )

    print(
        f"Accuracy Std: "
        f"{best_model['Accuracy_Std']:.4f}"
    )

    print(
        f"Mean F1: "
        f"{best_model['F1_Mean']:.4f}"
    )

    print(
        f"F1 Std: "
        f"{best_model['F1_Std']:.4f}"
    )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()