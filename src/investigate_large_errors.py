from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPORAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_score_dataset.csv"
)

LARGEST_ERRORS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "largest_prediction_errors.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "large_error_match_analysis.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_N_MATCHES = 10


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LARGE PREDICTION ERROR INVESTIGATION")
    print("=" * 70)

    print("\nLoading largest-error file...")

    errors = pd.read_csv(
        LARGEST_ERRORS
    )

    print(
        f"Largest-error rows: {len(errors):,}"
    )

    print("\nLoading temporal dataset...")

    df = pd.read_csv(
        TEMPORAL_DATASET,
        low_memory=False
    )

    print(
        f"Temporal dataset shape: {df.shape}"
    )

    return errors, df


# ============================================================
# IDENTIFY WORST MATCHES
# ============================================================

def identify_worst_matches(errors):

    print("\n")
    print("=" * 70)
    print("WORST MATCHES")
    print("=" * 70)

    match_summary = (
        errors
        .groupby(
            ["match_id", "innings"],
            as_index=False
        )
        .agg(
            max_absolute_error=(
                "absolute_error",
                "max"
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean"
            ),
            error_count=(
                "absolute_error",
                "count"
            ),
            actual_final_score=(
                "final_score",
                "first"
            ),
            predicted_score_max=(
                "predicted_score",
                "max"
            ),
            batting_team=(
                "batting_team",
                "first"
            )
        )
        .sort_values(
            "max_absolute_error",
            ascending=False
        )
        .head(TOP_N_MATCHES)
    )

    print(
        match_summary.to_string(
            index=False
        )
    )

    return match_summary


# ============================================================
# DETAILED MATCH ANALYSIS
# ============================================================

def analyze_match(
    df,
    match_id,
    innings
):

    match = df[
        (df["match_id"] == match_id)
        & (df["innings"] == innings)
    ].copy()

    if match.empty:
        return None

    match = match.sort_values(
        ["over", "ball"]
    )

    # --------------------------------------------------------
    # Basic innings information
    # --------------------------------------------------------

    final_score = (
        match["final_score"]
        .iloc[-1]
    )

    max_total_score = (
        match["total_score"]
        .max()
    )

    max_over = (
        match["over"]
        .max()
    )

    number_of_deliveries = len(
        match
    )

    wickets = (
        match["wickets"]
        .max()
    )

    batting_team = (
        match["batting_team"]
        .iloc[0]
        if "batting_team" in match.columns
        else "Unknown"
    )

    bowling_team = (
        match["bowling_team"]
        .iloc[0]
        if "bowling_team" in match.columns
        else "Unknown"
    )

    # --------------------------------------------------------
    # Last observed delivery
    # --------------------------------------------------------

    last_row = match.iloc[-1]

    # --------------------------------------------------------
    # Runs by over
    # --------------------------------------------------------

    over_scores = (
        match
        .groupby("over")["runs_total"]
        .sum()
    )

    # --------------------------------------------------------
    # Maximum score before final innings result
    # --------------------------------------------------------

    max_score_before_end = (
        match["total_score"]
        .max()
    )

    # --------------------------------------------------------
    # Identify whether innings was extremely short
    # --------------------------------------------------------

    legal_balls = (
        match["legal_balls"].max()
        if "legal_balls" in match.columns
        else np.nan
    )

    overs_completed = (
        legal_balls / 6
        if pd.notna(legal_balls)
        else np.nan
    )

    return {
        "match_id": match_id,
        "innings": innings,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "final_score": final_score,
        "max_total_score": max_total_score,
        "deliveries": number_of_deliveries,
        "legal_balls": legal_balls,
        "overs_completed": overs_completed,
        "max_over_index": max_over,
        "wickets": wickets,
        "first_total_score": match["total_score"].iloc[0],
        "last_total_score": match["total_score"].iloc[-1],
        "last_overs_remaining": (
            last_row["overs_remaining"]
            if "overs_remaining" in match.columns
            else np.nan
        ),
        "max_current_run_rate": (
            match["current_run_rate"].max()
            if "current_run_rate" in match.columns
            else np.nan
        ),
        "number_of_overs_with_runs": (
            (over_scores > 0).sum()
        ),
    }


# ============================================================
# ANALYZE ALL WORST MATCHES
# ============================================================

def analyze_worst_matches(
    df,
    match_summary
):

    rows = []

    for _, row in match_summary.iterrows():

        analysis = analyze_match(
            df,
            row["match_id"],
            row["innings"]
        )

        if analysis is not None:

            analysis[
                "max_prediction_error"
            ] = row[
                "max_absolute_error"
            ]

            analysis[
                "mean_prediction_error"
            ] = row[
                "mean_absolute_error"
            ]

            rows.append(
                analysis
            )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    errors, df = load_data()

    match_summary = identify_worst_matches(
        errors
    )

    detailed = analyze_worst_matches(
        df,
        match_summary
    )

    print("\n")
    print("=" * 70)
    print("DETAILED LARGE-ERROR MATCH ANALYSIS")
    print("=" * 70)

    print(
        detailed.to_string(
            index=False
        )
    )

    detailed.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved analysis to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()