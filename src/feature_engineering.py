"""
Build a delivery-level score-prediction table from the parsed Cricsheet ODI data.

Leakage rule: every feature except final_score uses only information from the
current delivery and earlier deliveries in the same match innings.

This script reads data/processed/odi_deliveries.csv and writes a new file. It
does not overwrite the raw delivery CSV and does not train models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "odi_deliveries.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "score_prediction_dataset.csv"

ODI_SCHEDULED_OVERS = 50.0
SUPER_OVER_SCHEDULED_OVERS = 1.0
LAST_N_OVERS = 5

# Inspected wicket_kind values in odi_deliveries.csv:
#   caught, bowled, lbw, run out, caught and bowled, stumped, hit wicket,
#   obstructing the field, timed out, retired hurt,
#   and two dual events: "caught; timed out", "caught; retired hurt".
# Retired hurt is not a dismissal under the Laws of Cricket (the batter may
# return). The IEEE paper also stores retired hurt under "Other wickets",
# separate from the cumulative Wickets column. Timed out is a real wicket.
NON_WICKET_KINDS = frozenset({"retired hurt"})

NEW_FEATURE_COLUMNS = [
    "wicket_event",
    "total_score",
    "final_score",
    "wickets",
    "legal_balls",
    "overs_bowled",
    "overs_remaining",
    "current_run_rate",
    "last_5_overs_score",
]


def load_deliveries(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Raw delivery dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def sort_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological order within each innings. ball already counts extras."""
    return df.sort_values(
        ["match_id", "innings", "over", "ball"],
        kind="mergesort",
    ).reset_index(drop=True)


def wicket_event_counts(wicket_kind: pd.Series) -> pd.Series:
    """
    Count actual dismissals on a delivery.

    Multiple kinds can appear on one ball, separated by '; '.
    'retired hurt' is ignored; every other recorded kind is counted.
    """
    text = wicket_kind.fillna("").astype(str).str.strip()
    empty = text.eq("") | text.str.lower().eq("nan")

    # Number of kind tokens, then subtract non-wicket tokens.
    n_tokens = text.str.count(";").add(1)
    n_tokens = n_tokens.mask(empty, 0)

    lowered = text.str.lower()
    n_non_wicket = pd.Series(0, index=wicket_kind.index, dtype="int64")
    for kind in NON_WICKET_KINDS:
        n_non_wicket = n_non_wicket + lowered.str.count(rf"(?:^|;\s*){kind}(?:\s*;|$)")

    return (n_tokens - n_non_wicket).clip(lower=0).astype("int64")


def is_legal_delivery(df: pd.DataFrame) -> pd.Series:
    """
    A legal ball is one that is not a wide and not a no-ball.

    Byes, leg-byes and penalty extras still count as a ball of the over.
    """
    wides = pd.to_numeric(df["extras_wides"], errors="coerce").fillna(0)
    noballs = pd.to_numeric(df["extras_noballs"], errors="coerce").fillna(0)
    return (wides.eq(0) & noballs.eq(0))


def scheduled_overs(df: pd.DataFrame) -> pd.Series:
    """50-over allocation for a normal ODI innings; 1 over for a super over."""
    super_over = df["super_over"].isin([True, "True", "true", 1, "1"])
    return np.where(super_over, SUPER_OVER_SCHEDULED_OVERS, ODI_SCHEDULED_OVERS)


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    innings_key = ["match_id", "innings"]
    grouped = df.groupby(innings_key, sort=False)

    # Current innings score after this delivery (includes this ball).
    df["total_score"] = grouped["runs_total"].cumsum()

    # Regression target: last cumulative score of the innings.
    # This is the only feature allowed to use later deliveries.
    df["final_score"] = grouped["total_score"].transform("max")

    df["wicket_event"] = wicket_event_counts(df["wicket_kind"])
    df["wickets"] = grouped["wicket_event"].cumsum()

    df["is_legal_delivery"] = is_legal_delivery(df)
    df["legal_balls"] = grouped["is_legal_delivery"].cumsum().astype("int64")

    # Cricket overs as completed overs + fraction of the current over.
    df["overs_bowled"] = df["legal_balls"] / 6.0
    df["overs_remaining"] = (scheduled_overs(df) - df["overs_bowled"]).clip(lower=0)

    # Safe CRR: undefined until at least one legal ball has been bowled.
    df["current_run_rate"] = np.where(
        df["legal_balls"] > 0,
        df["total_score"] / df["overs_bowled"],
        np.nan,
    )
    return df


def add_last_5_overs_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs in the current over so far plus the previous four completed overs.

    At over o (0-indexed) the window is overs max(0, o-4) ... o, using only
    balls up to the current delivery. Implemented as:

        last_5_overs_score = total_score - score at the end of over (o-5)

    The end-of-over score for over (o-5) is entirely in the past, so this
    does not leak future balls of the current over.
    """
    end_of_over = (
        df.groupby(["match_id", "innings", "over"], sort=False)["total_score"]
        .last()
        .rename("score_before_window")
        .reset_index()
    )
    # Align that completed-over score with the start of the 5-over window.
    end_of_over = end_of_over.rename(columns={"over": "cutoff_over"})

    df["cutoff_over"] = df["over"] - LAST_N_OVERS
    df = df.merge(
        end_of_over,
        on=["match_id", "innings", "cutoff_over"],
        how="left",
    )
    df["last_5_overs_score"] = (
        df["total_score"] - df["score_before_window"].fillna(0)
    ).astype("int64")
    df = df.drop(columns=["cutoff_over", "score_before_window"])
    return sort_deliveries(df)


def drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["is_legal_delivery"], errors="ignore")


def print_report(df: pd.DataFrame) -> None:
    n_matches = df["match_id"].nunique()
    n_innings = df.groupby(["match_id", "innings"]).ngroups

    print("Output:", OUTPUT_CSV)
    print(f"Final shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"Unique matches: {n_matches:,}")
    print(f"Unique innings (match_id, innings): {n_innings:,}")
    print("Innings value counts:")
    print(df.groupby(["match_id", "innings"])["innings"].first().value_counts().sort_index().to_string())

    print("\nSample rows (5):")
    sample_cols = [
        "match_id",
        "innings",
        "over",
        "ball",
        "runs_total",
        "total_score",
        "final_score",
        "wicket_event",
        "wickets",
        "legal_balls",
        "overs_bowled",
        "overs_remaining",
        "current_run_rate",
        "last_5_overs_score",
    ]
    print(df.loc[:, sample_cols].head(12).to_string(index=False))

    print("\nMissing values for newly created features:")
    missing = df[NEW_FEATURE_COLUMNS].isna().sum()
    missing_pct = (missing / len(df) * 100).round(4)
    summary = pd.DataFrame(
        {
            "column": NEW_FEATURE_COLUMNS,
            "missing_count": missing.values,
            "missing_percent": missing_pct.values,
        }
    )
    print(summary.to_string(index=False))

    print("\nBasic statistics:")
    stats_cols = [
        "total_score",
        "final_score",
        "wickets",
        "current_run_rate",
        "last_5_overs_score",
        "overs_remaining",
        "overs_bowled",
    ]
    print(df[stats_cols].describe().to_string())

    print("\nWicket sanity checks:")
    print(f"max cumulative wickets: {df['wickets'].max()}")
    print(f"deliveries with wicket_event > 0: {(df['wicket_event'] > 0).sum():,}")
    print(f"retired-hurt rows counted as 0 events: {int(((df['wicket_kind'] == 'retired hurt') & (df['wicket_event'] == 0)).sum())}")

    print("\nDid not modify:", INPUT_CSV)


def main() -> None:
    raw = load_deliveries(INPUT_CSV)
    df = sort_deliveries(raw)
    df = add_cumulative_features(df)
    df = add_last_5_overs_score(df)
    df = drop_helper_columns(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print_report(df)


if __name__ == "__main__":
    main()
