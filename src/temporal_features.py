"""
Create leakage-safe temporal and momentum features for cricket
score prediction.

New features:
    - last_3_overs_score
    - last_10_overs_score
    - boundary_rate
    - dot_ball_rate
    - recent_5_over_wickets

All predictor features use only the current delivery and
earlier deliveries in the same match innings.

final_score remains the prediction target and is allowed
to use the complete innings.

IMPORTANT:
    This script does NOT modify:
        data/processed/score_prediction_dataset.csv

    It creates:
        data/processed/temporal_score_dataset.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


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

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_score_dataset.csv"
)


# ============================================================
# SETTINGS
# ============================================================

GROUP_COLUMNS = [
    "match_id",
    "innings",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    return df


# ============================================================
# SORT DELIVERIES
# ============================================================

def sort_deliveries(df):

    return (
        df.sort_values(
            [
                "match_id",
                "innings",
                "over",
                "ball",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ============================================================
# LEGAL DELIVERY
# ============================================================

def add_legal_delivery(df):

    wides = pd.to_numeric(
        df["extras_wides"],
        errors="coerce",
    ).fillna(0)

    noballs = pd.to_numeric(
        df["extras_noballs"],
        errors="coerce",
    ).fillna(0)

    df["is_legal_for_temporal"] = (
        wides.eq(0)
        & noballs.eq(0)
    )

    return df


# ============================================================
# ROLLING OVER SCORE
# ============================================================

def add_rolling_over_score(
    df,
    n,
    output_column,
):
    """
    Calculate runs scored during the current over plus
    the previous n-1 completed overs.

    Example:
        n = 3
        current over = 10

    Feature contains:
        over 8
        over 9
        current over 10

    Calculation:

        current cumulative score
        -
        cumulative score at the end of over 7

    This uses only information available up to the
    current delivery and therefore avoids future leakage.
    """

    # --------------------------------------------------------
    # Get cumulative score at the end of every over.
    # --------------------------------------------------------

    over_end = (
        df.groupby(
            GROUP_COLUMNS + ["over"],
            sort=False,
        )["total_score"]
        .last()
        .reset_index()
    )

    over_end = over_end.rename(
        columns={
            "over": "cutoff_over",
            "total_score": "score_before_window",
        }
    )

    # --------------------------------------------------------
    # For current over o:
    #
    # cutoff = o - n
    #
    # Example:
    # n = 3, current over = 10
    # cutoff = 7
    #
    # current score - score at end of over 7
    # = runs in overs 8, 9 and current 10.
    # --------------------------------------------------------

    df["cutoff_over"] = (
        df["over"] - n
    )

    df = df.merge(
        over_end[
            GROUP_COLUMNS
            + [
                "cutoff_over",
                "score_before_window",
            ]
        ],
        on=(
            GROUP_COLUMNS
            + ["cutoff_over"]
        ),
        how="left",
    )

    df[output_column] = (
        df["total_score"]
        - df["score_before_window"].fillna(0)
    )

    # --------------------------------------------------------
    # Safety check.
    # --------------------------------------------------------

    negative_count = (
        df[output_column] < 0
    ).sum()

    if negative_count > 0:

        raise ValueError(
            f"{output_column} contains "
            f"{negative_count:,} negative values."
        )

    df = df.drop(
        columns=[
            "cutoff_over",
            "score_before_window",
        ]
    )

    return df


# ============================================================
# BOUNDARY + DOT BALL FEATURES
# ============================================================

def add_ball_quality_features(df):
    """
    Create cumulative boundary and dot-ball rates.

    boundary_rate:
        cumulative boundary deliveries
        /
        cumulative deliveries

    dot_ball_rate:
        cumulative dot balls
        /
        cumulative legal deliveries

    Both features use only the current delivery and
    earlier deliveries.
    """

    grouped = df.groupby(
        GROUP_COLUMNS,
        sort=False,
    )

    # --------------------------------------------------------
    # Total deliveries observed up to current delivery
    #
    # Every row represents one recorded delivery.
    # --------------------------------------------------------

    df["temporal_delivery_count"] = (
        grouped["match_id"]
        .cumcount()
        .add(1)
        .astype("int64")
    )

    # --------------------------------------------------------
    # Legal deliveries up to current delivery
    # --------------------------------------------------------

    df["temporal_legal_balls"] = (
        grouped[
            "is_legal_for_temporal"
        ]
        .cumsum()
        .astype("int64")
    )

    # --------------------------------------------------------
    # Batter runs
    # --------------------------------------------------------

    runs_batter = pd.to_numeric(
        df["runs_batter"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # Boundary event
    #
    # A boundary is counted when batter runs are exactly
    # 4 or 6.
    # --------------------------------------------------------

    df["boundary_event"] = (
        runs_batter
        .isin([4, 6])
        .astype("int64")
    )

    # --------------------------------------------------------
    # Cumulative boundaries
    # --------------------------------------------------------

    df["cumulative_boundaries"] = (
        grouped[
            "boundary_event"
        ]
        .cumsum()
        .astype("int64")
    )

    # --------------------------------------------------------
    # Boundary rate
    #
    # IMPORTANT:
    # Use total deliveries rather than legal deliveries.
    #
    # This prevents a boundary on a no-ball from producing
    # a rate greater than 1.
    # --------------------------------------------------------

    df["boundary_rate"] = np.where(
        df["temporal_delivery_count"] > 0,

        (
            df["cumulative_boundaries"]
            / df["temporal_delivery_count"]
        ),

        0.0,
    )

    # --------------------------------------------------------
    # Total runs on delivery
    # --------------------------------------------------------

    runs_total = pd.to_numeric(
        df["runs_total"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # Dot-ball event
    #
    # Legal delivery AND zero total runs.
    # --------------------------------------------------------

    df["dot_ball_event"] = (
        df["is_legal_for_temporal"]
        & runs_total.eq(0)
    ).astype("int64")

    # --------------------------------------------------------
    # Cumulative dot balls
    # --------------------------------------------------------

    df["cumulative_dot_balls"] = (
        grouped[
            "dot_ball_event"
        ]
        .cumsum()
        .astype("int64")
    )

    # --------------------------------------------------------
    # Dot-ball rate
    #
    # Dot balls / legal deliveries
    # --------------------------------------------------------

    df["dot_ball_rate"] = np.where(
        df["temporal_legal_balls"] > 0,

        (
            df["cumulative_dot_balls"]
            / df["temporal_legal_balls"]
        ),

        0.0,
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    invalid_boundary = (
        (df["boundary_rate"] < 0)
        | (df["boundary_rate"] > 1)
    ).sum()

    if invalid_boundary > 0:

        raise ValueError(
            "boundary_rate contains values "
            "outside [0, 1]."
        )

    invalid_dot = (
        (df["dot_ball_rate"] < 0)
        | (df["dot_ball_rate"] > 1)
    ).sum()

    if invalid_dot > 0:

        raise ValueError(
            "dot_ball_rate contains values "
            "outside [0, 1]."
        )

    return df

# ============================================================
# RECENT WICKET FEATURE
# ============================================================

def add_recent_wickets(df):
    """
    Calculate wickets taken during the current over and
    previous four completed overs.

    For current over o:

        wickets at current delivery
        -
        wickets at end of over o-5

    The result can never be negative.
    """

    # --------------------------------------------------------
    # Wickets at the end of each over
    # --------------------------------------------------------

    over_wickets = (
        df.groupby(
            GROUP_COLUMNS + ["over"],
            sort=False,
        )["wickets"]
        .last()
        .reset_index()
    )

    over_wickets = over_wickets.rename(
        columns={
            "over": "cutoff_over",
            "wickets": "wickets_before_recent_window",
        }
    )

    # --------------------------------------------------------
    # Current over o:
    #
    # cutoff = o - 5
    #
    # Example:
    # current over = 20
    # cutoff = 15
    #
    # wickets now - wickets at end of over 15
    # --------------------------------------------------------

    df["cutoff_over"] = (
        df["over"] - 5
    )

    df = df.merge(
        over_wickets[
            GROUP_COLUMNS
            + [
                "cutoff_over",
                "wickets_before_recent_window",
            ]
        ],
        on=(
            GROUP_COLUMNS
            + ["cutoff_over"]
        ),
        how="left",
    )

    df["recent_5_over_wickets"] = (
        df["wickets"]
        - df[
            "wickets_before_recent_window"
        ].fillna(0)
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    negative_count = (
        df["recent_5_over_wickets"] < 0
    ).sum()

    if negative_count > 0:

        raise ValueError(
            "recent_5_over_wickets contains "
            f"{negative_count:,} negative values."
        )

    df = df.drop(
        columns=[
            "cutoff_over",
            "wickets_before_recent_window",
        ]
    )

    return df


# ============================================================
# CLEAN HELPER COLUMNS
# ============================================================

def clean_dataset(df):

    helper_columns = [
        "is_legal_for_temporal",
        "temporal_delivery_count",
        "temporal_legal_balls",
        "boundary_event",
        "cumulative_boundaries",
        "dot_ball_event",
        "cumulative_dot_balls",
    ]

    return df.drop(
        columns=helper_columns,
        errors="ignore",
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_temporal_features(df):
    """
    Final safety audit before the dataset is saved.
    """

    temporal_features = [
        "last_3_overs_score",
        "last_10_overs_score",
        "boundary_rate",
        "dot_ball_rate",
        "recent_5_over_wickets",
    ]

    print("\n")
    print("=" * 70)
    print("FINAL TEMPORAL FEATURE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    for column in temporal_features:

        missing = int(
            df[column].isna().sum()
        )

        print(
            f"{column}: {missing:,}"
        )

        if missing > 0:

            raise ValueError(
                f"{column} contains "
                f"{missing:,} missing values."
            )

    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    print("\nNegative values:")

    for column in temporal_features:

        negative = int(
            (df[column] < 0).sum()
        )

        print(
            f"{column}: {negative:,}"
        )

        # Score and wicket features cannot be negative.
        if column in [
            "last_3_overs_score",
            "last_10_overs_score",
            "recent_5_over_wickets",
        ]:

            if negative > 0:

                raise ValueError(
                    f"{column} contains "
                    f"negative values."
                )

    # --------------------------------------------------------
    # Rate validation
    # --------------------------------------------------------

    print("\nRate validation:")

    for column in [
        "boundary_rate",
        "dot_ball_rate",
    ]:

        below_zero = int(
            (df[column] < 0).sum()
        )

        above_one = int(
            (df[column] > 1).sum()
        )

        print(
            f"{column}: "
            f"below 0 = {below_zero:,}, "
            f"above 1 = {above_one:,}"
        )

        if (
            below_zero > 0
            or above_one > 0
        ):

            raise ValueError(
                f"{column} is outside "
                f"the valid [0, 1] range."
            )

    # --------------------------------------------------------
    # Final score relationship
    # --------------------------------------------------------

    if (
        "total_score" in df.columns
        and "final_score" in df.columns
    ):

        invalid_total = (
            df["total_score"]
            > df["final_score"]
        ).sum()

        print(
            "\nRows where total_score > final_score:",
            f"{invalid_total:,}",
        )

        if invalid_total > 0:

            raise ValueError(
                "total_score exceeds final_score."
            )

    print(
        "\nAll temporal feature checks passed."
    )


# ============================================================
# REPORT
# ============================================================

def report(df):

    temporal_features = [
        "last_3_overs_score",
        "last_10_overs_score",
        "boundary_rate",
        "dot_ball_rate",
        "recent_5_over_wickets",
    ]

    print("\n")
    print("=" * 70)
    print("TEMPORAL FEATURE REPORT")
    print("=" * 70)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Matches: "
        f"{df['match_id'].nunique():,}"
    )

    print("\nNew feature statistics:")

    print(
        df[temporal_features]
        .describe()
        .to_string()
    )

    print("\nMaximum values:")

    for column in temporal_features:

        print(
            f"{column}: "
            f"{df[column].max()}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading engineered dataset..."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = sort_deliveries(df)

    # --------------------------------------------------------
    # LEGAL DELIVERY
    # --------------------------------------------------------

    df = add_legal_delivery(df)

    # --------------------------------------------------------
    # RECENT SCORING FEATURES
    # --------------------------------------------------------

    print(
        "Creating last_3_overs_score..."
    )

    df = add_rolling_over_score(
        df,
        n=3,
        output_column=
            "last_3_overs_score",
    )

    print(
        "Creating last_10_overs_score..."
    )

    df = add_rolling_over_score(
        df,
        n=10,
        output_column=
            "last_10_overs_score",
    )

    # --------------------------------------------------------
    # BOUNDARY + DOT BALL FEATURES
    # --------------------------------------------------------

    print(
        "Creating boundary_rate and dot_ball_rate..."
    )

    df = add_ball_quality_features(
        df
    )

    # --------------------------------------------------------
    # RECENT WICKET FEATURE
    # --------------------------------------------------------

    print(
        "Creating recent_5_over_wickets..."
    )

    df = add_recent_wickets(
        df
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = clean_dataset(df)

    # --------------------------------------------------------
    # RESTORE CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    df = sort_deliveries(df)

    # --------------------------------------------------------
    # VALIDATE BEFORE SAVING
    # --------------------------------------------------------

    validate_temporal_features(
        df
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report(df)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "TEMPORAL DATASET CREATED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_CSV
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()