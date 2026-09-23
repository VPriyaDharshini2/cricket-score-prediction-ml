from pathlib import Path
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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

RESULTS_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_classification_results.csv"
)

CONFUSION_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_confusion_matrices.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_player_classifier.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


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
    print("PLAYER CLASSIFICATION EXPERIMENT")
    print("=" * 70)

    print("\nLoading classification dataset...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing features...")

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET
    ].copy()

    # Convert all features to numeric
    for column in FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Target: {TARGET}"
    )

    return X, y


# ============================================================
# ENCODE TARGET
# ============================================================

def encode_target(y):

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y
    )

    print("\nClass mapping:")

    for index, label in enumerate(
        encoder.classes_
    ):

        print(
            f"  {index} -> {label}"
        )

    return y_encoded, encoder


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n")
    print("=" * 70)
    print("STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows: {len(X_test):,}"
    )

    print(
        f"Training percentage: "
        f"{len(X_train) / len(X) * 100:.1f}%"
    )

    print(
        f"Testing percentage: "
        f"{len(X_test) / len(X) * 100:.1f}%"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():

    # Models that benefit from standardized features
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

    # Models that don't require scaling
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
# TRAIN ONE MODEL
# ============================================================

def train_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    label_encoder
):

    print("\n")
    print("-" * 70)
    print(f"TRAINING: {name}")
    print("-" * 70)

    start = time.time()

    model.fit(
        X_train,
        y_train
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

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"Train time: {train_time:.2f}s"
    )

    print(
        f"Prediction time: "
        f"{prediction_time:.4f}s"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    result = {
        "Model": name,
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Features": len(FEATURES),
        "Accuracy": accuracy,
        "Precision_weighted": precision,
        "Recall_weighted": recall,
        "F1_weighted": f1,
        "Train_Time_sec": train_time,
        "Prediction_Time_sec": prediction_time
    }

    return (
        model,
        result,
        cm
    )


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(
    matrices,
    label_encoder
):

    rows = []

    class_names = (
        label_encoder.classes_
    )

    for model_name, matrix in matrices.items():

        for i, actual in enumerate(
            class_names
        ):

            for j, predicted in enumerate(
                class_names
            ):

                rows.append(
                    {
                        "Model": model_name,
                        "Actual": actual,
                        "Predicted": predicted,
                        "Count": int(
                            matrix[i][j]
                        )
                    }
                )

    cm_df = pd.DataFrame(
        rows
    )

    CONFUSION_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cm_df.to_csv(
        CONFUSION_CSV,
        index=False
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    models,
    results
):

    results_df = pd.DataFrame(
        results
    )

    best_index = (
        results_df[
            "F1_weighted"
        ]
        .idxmax()
    )

    best_model_name = (
        results_df.loc[
            best_index,
            "Model"
        ]
    )

    best_model = models[
        best_model_name
    ]

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            best_model,
            file
        )

    return (
        best_model_name,
        results_df
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
    # ENCODE TARGET
    # --------------------------------------------------------

    y_encoded, label_encoder = (
        encode_target(y)
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y_encoded
    )

    # --------------------------------------------------------
    # BUILD MODELS
    # --------------------------------------------------------

    models = build_models()

    results = []
    trained_models = {}
    confusion_matrices = {}

    # --------------------------------------------------------
    # TRAIN ALL MODELS
    # --------------------------------------------------------

    for name, model in models.items():

        (
            trained_model,
            result,
            cm
        ) = train_model(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            label_encoder
        )

        results.append(
            result
        )

        trained_models[name] = (
            trained_model
        )

        confusion_matrices[name] = cm

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "F1_weighted",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n")
    print("=" * 70)
    print("PLAYER CLASSIFICATION MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SAVE CONFUSION MATRICES
    # --------------------------------------------------------

    save_confusion_matrices(
        confusion_matrices,
        label_encoder
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    (
        best_model_name,
        _
    ) = save_best_model(
        trained_models,
        results
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PLAYER CLASSIFICATION COMPLETE")
    print("=" * 70)

    print(
        f"\nBest model by weighted F1: "
        f"{best_model_name}"
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_CSV}"
    )

    print(
        f"\nConfusion matrices saved to:"
        f"\n{CONFUSION_CSV}"
    )

    print(
        f"\nBest model saved to:"
        f"\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()