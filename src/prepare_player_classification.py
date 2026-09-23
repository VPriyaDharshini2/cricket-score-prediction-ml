from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_statistics.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_classification_dataset.csv"
)


# ============================================================
# PAPER-BASED FEATURES
# ============================================================

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "role"

VALID_ROLES = [
    "Batsman",
    "Bowler",
    "Batting All-rounder",
    "Bowling All-rounder",
]


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("=" * 65)
    print("PLAYER CLASSIFICATION DATASET PREPARATION")
    print("=" * 65)

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nOriginal dataset shape: "
        f"{df.shape}"
    )

    return df


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(df):

    required = (
        ["player"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("\nAll required columns found.")


# ============================================================
# FILTER FOUR CLASSES
# ============================================================

def create_classification_dataset(df):

    print("\nFiltering four player categories...")

    classification_df = df[
        df[TARGET_COLUMN].isin(
            VALID_ROLES
        )
    ].copy()

    excluded = (
        ~df[TARGET_COLUMN].isin(
            VALID_ROLES
        )
    ).sum()

    print(
        f"Players excluded due to insufficient "
        f"evidence: {excluded:,}"
    )

    print(
        f"Players retained for classification: "
        f"{len(classification_df):,}"
    )

    return classification_df


# ============================================================
# SELECT FEATURES
# ============================================================

def select_features(df):

    columns = (
        ["player"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    df = df[
        columns
    ].copy()

    return df


# ============================================================
# CHECK DUPLICATES
# ============================================================

def check_duplicates(df):

    duplicate_players = (
        df["player"]
        .duplicated()
        .sum()
    )

    duplicate_rows = (
        df.duplicated()
        .sum()
    )

    print("\nDuplicate checks:")

    print(
        f"Duplicate player names: "
        f"{duplicate_players}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows}"
    )

    if duplicate_rows > 0:

        raise ValueError(
            "Duplicate classification rows detected."
        )


# ============================================================
# ROLE DISTRIBUTION
# ============================================================

def print_role_distribution(df):

    print("\n")
    print("=" * 65)
    print("CLASS DISTRIBUTION")
    print("=" * 65)

    counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .reindex(VALID_ROLES)
    )

    percentages = (
        counts
        / len(df)
        * 100
    )

    distribution = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    print(
        distribution.to_string()
    )


# ============================================================
# MISSING VALUES
# ============================================================

def print_missing_values(df):

    print("\n")
    print("=" * 65)
    print("MISSING VALUES")
    print("=" * 65)

    missing = (
        df[
            FEATURE_COLUMNS
        ]
        .isna()
        .sum()
    )

    print(
        missing.to_string()
    )

    print(
        "\nMissing values will be handled by "
        "the ML preprocessing pipeline."
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

def print_feature_statistics(df):

    print("\n")
    print("=" * 65)
    print("FEATURE STATISTICS")
    print("=" * 65)

    print(
        df[
            FEATURE_COLUMNS
        ]
        .describe()
        .to_string()
    )


# ============================================================
# SAVE
# ============================================================

def save_dataset(df):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n")
    print("=" * 65)
    print("CLASSIFICATION DATASET CREATED")
    print("=" * 65)

    print(
        f"\nShape: {df.shape}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    validate_columns(
        df
    )

    df = create_classification_dataset(
        df
    )

    df = select_features(
        df
    )

    check_duplicates(
        df
    )

    print_role_distribution(
        df
    )

    print_missing_values(
        df
    )

    print_feature_statistics(
        df
    )

    save_dataset(
        df
    )


if __name__ == "__main__":
    main()