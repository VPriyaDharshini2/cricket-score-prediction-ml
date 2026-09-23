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
    / "odi_deliveries.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_statistics.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Minimum evidence used when creating reproducible role labels.
# These are implementation thresholds, NOT thresholds stated
# explicitly in the base paper.
MIN_BATTING_INNINGS = 5
MIN_BOWLING_INNINGS = 5

# Minimum career batting strike rate considered meaningful.
# This is only used for role labeling.
BATTING_SR_THRESHOLD = 50.0

# Minimum wickets per bowling innings considered meaningful.
WICKETS_PER_INNINGS_THRESHOLD = 0.5


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 65)
    print("PLAYER FEATURE ENGINEERING")
    print("=" * 65)

    print("\nLoading Cricsheet delivery dataset...")

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
        f"Matches: "
        f"{df['match_id'].nunique():,}"
    )

    return df


# ============================================================
# NUMERIC CLEANING
# ============================================================

def prepare_numeric_columns(df):

    numeric_columns = [
        "runs_batter",
        "runs_total",
        "extras_wides",
        "extras_noballs",
        "extras_byes",
        "extras_legbyes",
        "extras_penalty",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)

    return df


# ============================================================
# LEGAL DELIVERY
# ============================================================

def add_legal_delivery(df):

    wides = df[
        "extras_wides"
    ]

    noballs = df[
        "extras_noballs"
    ]

    df["is_legal_delivery"] = (
        (wides == 0)
        &
        (noballs == 0)
    )

    return df


# ============================================================
# WICKET COUNT
# ============================================================

NON_WICKET_KINDS = {
    "retired hurt"
}


def count_wickets(value):

    if pd.isna(value):
        return 0

    text = str(value).strip().lower()

    if not text or text == "nan":
        return 0

    parts = [
        part.strip()
        for part in text.split(";")
    ]

    count = 0

    for part in parts:

        if part in NON_WICKET_KINDS:
            continue

        count += 1

    return count


def add_wicket_count(df):

    df["wicket_count"] = (
        df["wicket_kind"]
        .apply(count_wickets)
        .astype(int)
    )

    return df


# ============================================================
# BATTER STATISTICS
# ============================================================

def calculate_batting_statistics(df):

    print("\nCalculating batting statistics...")

    batting = (
        df.groupby("batter")
        .agg(
            runs=(
                "runs_batter",
                "sum"
            ),
            batting_balls=(
                "is_legal_delivery",
                "sum"
            ),
            batting_matches=(
                "match_id",
                "nunique"
            ),
        )
        .reset_index()
    )

    batting = batting.rename(
        columns={
            "batter": "player"
        }
    )

    # Number of innings:
    # count unique match + innings combinations
    innings = (
        df.groupby("batter")
        .apply(
            lambda x:
            x[
                ["match_id", "innings"]
            ]
            .drop_duplicates()
            .shape[0]
        )
        .reset_index(
            name="innings"
        )
    )

    innings = innings.rename(
        columns={
            "batter": "player"
        }
    )

    batting = batting.merge(
        innings,
        on="player",
        how="left"
    )

    # Batting average:
    # Runs / innings.
    #
    # NOTE:
    # The paper describes Average as runs and innings,
    # but does not specify a dismissal-based cricket average.
    batting["average"] = np.where(
        batting["innings"] > 0,
        batting["runs"]
        / batting["innings"],
        0
    )

    # Strike rate:
    # runs / legal balls * 100
    batting["strike_rate"] = np.where(
        batting["batting_balls"] > 0,
        (
            batting["runs"]
            / batting["batting_balls"]
        ) * 100,
        0
    )

    return batting


# ============================================================
# BOWLER STATISTICS
# ============================================================

def calculate_bowling_statistics(df):

    print("\nCalculating bowling statistics...")

    bowling = (
        df.groupby("bowler")
        .agg(
            wickets=(
                "wicket_count",
                "sum"
            ),
            bowling_balls=(
                "is_legal_delivery",
                "sum"
            ),
            runs_conceded=(
                "runs_total",
                "sum"
            ),
            bowling_matches=(
                "match_id",
                "nunique"
            ),
        )
        .reset_index()
    )

    bowling = bowling.rename(
        columns={
            "bowler": "player"
        }
    )

    # Remove runs belonging to batsman's extras that aren't
    # charged to the bowler.
    #
    # Byes, leg-byes and penalty runs should not be charged.
    non_bowler_extras = (
        df["extras_byes"]
        + df["extras_legbyes"]
        + df["extras_penalty"]
    )

    df["bowler_runs_conceded"] = (
        df["runs_total"]
        - non_bowler_extras
    ).clip(lower=0)

    bowling_runs = (
        df.groupby("bowler")[
            "bowler_runs_conceded"
        ]
        .sum()
        .reset_index()
    )

    bowling_runs = bowling_runs.rename(
        columns={
            "bowler": "player",
            "bowler_runs_conceded":
                "runs_conceded"
        }
    )

    bowling = bowling.drop(
        columns=["runs_conceded"]
    )

    bowling = bowling.merge(
        bowling_runs,
        on="player",
        how="left"
    )

    # Bowling innings = number of match/innings combinations
    bowling_innings = (
        df.groupby("bowler")
        .apply(
            lambda x:
            x[
                ["match_id", "innings"]
            ]
            .drop_duplicates()
            .shape[0]
        )
        .reset_index(
            name="bowling_innings"
        )
    )

    bowling_innings = bowling_innings.rename(
        columns={
            "bowler": "player"
        }
    )

    bowling = bowling.merge(
        bowling_innings,
        on="player",
        how="left"
    )

    # Overs = legal balls / 6
    bowling["overs"] = (
        bowling["bowling_balls"] / 6.0
    )

    # Economy = runs conceded / overs
    bowling["economy"] = np.where(
        bowling["overs"] > 0,
        bowling["runs_conceded"]
        / bowling["overs"],
        0
    )

    # Bowling average = runs conceded / wickets
    bowling["bowling_average"] = np.where(
        bowling["wickets"] > 0,
        bowling["runs_conceded"]
        / bowling["wickets"],
        np.nan
    )

    # Bowling strike rate = balls / wickets
    bowling["bowling_strike_rate"] = np.where(
        bowling["wickets"] > 0,
        bowling["bowling_balls"]
        / bowling["wickets"],
        np.nan
    )

    return bowling


# ============================================================
# MERGE PLAYER STATISTICS
# ============================================================

def merge_player_statistics(
    batting,
    bowling
):

    print("\nMerging batting and bowling statistics...")

    batting_players = set(
        batting["player"]
    )

    bowling_players = set(
        bowling["player"]
    )

    all_players = sorted(
        batting_players
        | bowling_players
    )

    players = pd.DataFrame(
        {
            "player": all_players
        }
    )

    players = players.merge(
        batting,
        on="player",
        how="left"
    )

    players = players.merge(
        bowling,
        on="player",
        how="left",
        suffixes=(
            "_batting",
            "_bowling"
        )
    )

    # Fill statistics that don't exist for a player's
    # opposite discipline.
    zero_columns = [
        "runs",
        "batting_balls",
        "batting_matches",
        "innings",
        "wickets",
        "bowling_balls",
        "bowling_matches",
        "runs_conceded",
        "bowling_innings",
        "overs",
        "economy",
    ]

    for column in zero_columns:

        if column in players.columns:

            players[column] = (
                players[column]
                .fillna(0)
            )

    return players


# ============================================================
# BASRA FEATURES
# ============================================================

def add_basra_features(df):

    print("\nCalculating Batting Basra and Bowling Basra...")

    # Base-paper definition:
    #
    # Batting Basra = aggregate of batting average
    #                 and strike rate
    #
    # We interpret "aggregate" as the sum because
    # the paper does not provide another formula.
    df["batting_basra"] = (
        df["average"].fillna(0)
        +
        df["strike_rate"].fillna(0)
    )

    # Base-paper definition:
    #
    # Bowling Basra = ratio of wickets taken
    #                 to number of matches played.
    #
    df["bowling_basra"] = np.where(
        df["bowling_matches"] > 0,
        df["wickets"]
        / df["bowling_matches"],
        0
    )

    return df


# ============================================================
# REPRODUCIBLE ROLE LABEL
# ============================================================

def assign_role_labels(df):

    print("\nCreating reproducible player-role labels...")

    # --------------------------------------------------------
    # IMPORTANT:
    # The base paper specifies four categories but does not
    # provide explicit numerical thresholds for generating
    # ground-truth labels.
    #
    # Therefore these thresholds are implementation choices
    # for reproducing the experiment on Cricsheet data.
    # --------------------------------------------------------

    batting_capable = (
        (df["innings"] >= MIN_BATTING_INNINGS)
        &
        (df["runs"] > 0)
        &
        (df["strike_rate"] >= BATTING_SR_THRESHOLD)
    )

    bowling_capable = (
        (df["bowling_innings"] >= MIN_BOWLING_INNINGS)
        &
        (df["wickets"] > 0)
        &
        (
            df["wickets"]
            / df["bowling_innings"]
            >= WICKETS_PER_INNINGS_THRESHOLD
        )
    )

    # --------------------------------------------------------
    # Calculate normalized batting and bowling strength.
    #
    # We DO NOT compare raw Batting Basra and Bowling Basra
    # because they are on very different numerical scales.
    # --------------------------------------------------------

    batting_strength = (
        df["batting_basra"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    bowling_strength = (
        df["bowling_basra"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Normalize using percentile rank across players.
    # This puts both measures approximately on [0, 1].
    batting_rank = (
        batting_strength
        .rank(pct=True)
        .fillna(0)
    )

    bowling_rank = (
        bowling_strength
        .rank(pct=True)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Assign the four roles.
    # --------------------------------------------------------

    df["role"] = "Limited Evidence"

    # Batting only
    df.loc[
        batting_capable & ~bowling_capable,
        "role"
    ] = "Batsman"

    # Bowling only
    df.loc[
        ~batting_capable & bowling_capable,
        "role"
    ] = "Bowler"

    # Both batting + bowling
    both = (
        batting_capable
        &
        bowling_capable
    )

    # Stronger normalized batting profile
    df.loc[
        both
        &
        (batting_rank >= bowling_rank),
        "role"
    ] = "Batting All-rounder"

    # Stronger normalized bowling profile
    df.loc[
        both
        &
        (bowling_rank > batting_rank),
        "role"
    ] = "Bowling All-rounder"

    return df

# ============================================================
# FINAL COLUMN ORDER
# ============================================================

def select_output_columns(df):

    columns = [
        "player",

        # Batting
        "innings",
        "runs",
        "average",
        "strike_rate",

        # Bowling
        "wickets",
        "overs",
        "economy",
        "bowling_average",
        "bowling_strike_rate",

        # Paper's added features
        "batting_basra",
        "bowling_basra",

        # Supporting statistics
        "batting_balls",
        "bowling_balls",
        "runs_conceded",
        "bowling_innings",
        "batting_matches",
        "bowling_matches",

        # Reproducible implementation label
        "role",
    ]

    return df[
        [
            col
            for col in columns
            if col in df.columns
        ]
    ]


# ============================================================
# REPORT
# ============================================================

def print_report(df):

    print("\n")
    print("=" * 65)
    print("PLAYER STATISTICS REPORT")
    print("=" * 65)

    print(
        f"\nNumber of players: "
        f"{len(df):,}"
    )

    print("\nColumns:")
    print(
        "\n".join(
            f"  - {column}"
            for column in df.columns
        )
    )

    print("\nMissing values:")

    missing = (
        df.isna()
        .sum()
    )

    print(
        missing[
            missing > 0
        ].to_string()
        if (missing > 0).any()
        else "None"
    )

    print("\nRole distribution:")

    print(
        df["role"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSample players:")

    print(
        df.head(15)
        .to_string(index=False)
    )

    print("\nBasic statistics:")

    stats = [
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

    print(
        df[stats]
        .describe()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = prepare_numeric_columns(df)

    df = add_legal_delivery(df)

    df = add_wicket_count(df)

    batting = calculate_batting_statistics(
        df
    )

    bowling = calculate_bowling_statistics(
        df
    )

    players = merge_player_statistics(
        batting,
        bowling
    )

    players = add_basra_features(
        players
    )

    players = assign_role_labels(
        players
    )

    players = select_output_columns(
        players
    )

    print_report(players)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    players.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n")
    print("=" * 65)
    print("PLAYER STATISTICS DATASET CREATED")
    print("=" * 65)

    print(
        f"\nSaved to:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()