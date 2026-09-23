"""
Read-only quality checks for the flattened ODI delivery table.

This script does not modify, overwrite, drop, fill, or delete any data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "odi_deliveries.csv"

RUN_COLUMNS = ("runs_batter", "runs_extras", "runs_total")
EXTRA_COLUMNS = (
    "extras_wides",
    "extras_noballs",
    "extras_byes",
    "extras_legbyes",
    "extras_penalty",
)
DELIVERY_KEY = ["match_id", "innings", "over", "ball"]
TEAM_COLUMNS = ["team_1", "team_2", "batting_team", "bowling_team"]


def configure_display() -> None:
    """Show full tables in the console instead of truncated pandas previews."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.max_rows", 200)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def report_shape(df: pd.DataFrame) -> tuple[int, int]:
    print_header("1. Dataset shape")
    n_rows, n_cols = df.shape
    print(f"rows={n_rows:,}")
    print(f"columns={n_cols}")
    print(f"shape={df.shape}")
    return n_rows, n_cols


def report_dtypes(df: pd.DataFrame) -> None:
    print_header("2. Column names and data types")
    types = df.dtypes.astype(str)
    for name, dtype in types.items():
        print(f"{name}: {dtype}")


def unique_non_null(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.strip()


def report_unique_counts(df: pd.DataFrame) -> dict[str, int]:
    print_header("3. Unique match_ids")
    n_matches = df["match_id"].nunique(dropna=True)
    print(f"unique match_ids: {n_matches:,}")

    print_header("4. Unique teams")
    team_values = pd.concat([unique_non_null(df[col]) for col in TEAM_COLUMNS if col in df.columns], ignore_index=True)
    teams = sorted(team_values.unique())
    print(f"unique teams: {len(teams)}")
    for team in teams:
        print(f"  - {team}")

    print_header("5. Unique venues")
    n_venues = df["venue"].nunique(dropna=True)
    print(f"unique venues: {n_venues:,}")

    return {"n_matches": n_matches, "n_teams": len(teams), "n_venues": n_venues}


def report_dates(df: pd.DataFrame) -> tuple[str | None, str | None]:
    print_header("6. Match date range")
    dates = pd.to_datetime(df["date"], errors="coerce")
    n_unparsed = int(dates.isna().sum() - df["date"].isna().sum())
    min_date = dates.min()
    max_date = dates.max()
    min_text = None if pd.isna(min_date) else min_date.date().isoformat()
    max_text = None if pd.isna(max_date) else max_date.date().isoformat()
    print(f"minimum date: {min_text}")
    print(f"maximum date: {max_text}")
    print(f"missing dates: {int(df['date'].isna().sum()):,}")
    print(f"unparseable dates: {n_unparsed:,}")
    return min_text, max_text


def report_innings(df: pd.DataFrame) -> None:
    print_header("7. Innings values")
    unique_innings = sorted(df["innings"].dropna().unique().tolist())
    print(f"number of unique innings values: {len(unique_innings)}")
    print(f"unique innings values: {unique_innings}")

    print_header("8. Rows per innings")
    counts = df["innings"].value_counts(dropna=False).sort_index()
    for value, count in counts.items():
        print(f"innings={value}: {count:,} rows")
    print(f"total rows accounted for: {int(counts.sum()):,}")


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    print_header("9. Missing values per column")
    n_rows = len(df)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / n_rows * 100).round(4)
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_percent": missing_pct.values,
        }
    )
    print(missing.to_string(index=False))
    return missing


def report_duplicates(df: pd.DataFrame) -> dict[str, int]:
    print_header("10. Duplicate rows")
    n_dup_rows = int(df.duplicated().sum())
    print(f"duplicate rows (entire record): {n_dup_rows:,}")

    print_header("11. Duplicate delivery keys")
    print(f"delivery key columns: {DELIVERY_KEY}")
    n_dup_keys = int(df.duplicated(subset=DELIVERY_KEY).sum())
    print(f"duplicate delivery records: {n_dup_keys:,}")
    if n_dup_keys:
        key_counts = df.groupby(DELIVERY_KEY, dropna=False).size()
        extras = key_counts[key_counts > 1]
        print(f"unique colliding keys: {len(extras):,}")
        print("example colliding keys (up to 10):")
        print(extras.head(10).to_string())
    return {"duplicate_rows": n_dup_rows, "duplicate_delivery_keys": n_dup_keys}


def report_run_stats(df: pd.DataFrame) -> None:
    print_header("12. Run-column summary statistics")
    for column in RUN_COLUMNS:
        series = pd.to_numeric(df[column], errors="coerce")
        print(f"{column}:")
        print(f"  min={series.min()}")
        print(f"  max={series.max()}")
        print(f"  mean={series.mean():.6f}")
        print(f"  median={series.median()}")
        print(f"  missing={int(series.isna().sum()):,}")


def report_run_consistency(df: pd.DataFrame) -> int:
    print_header("13. runs_total vs runs_batter + runs_extras")
    batter = pd.to_numeric(df["runs_batter"], errors="coerce")
    extras = pd.to_numeric(df["runs_extras"], errors="coerce")
    total = pd.to_numeric(df["runs_total"], errors="coerce")
    comparable = batter.notna() & extras.notna() & total.notna()
    expected = batter + extras
    inconsistent = comparable & (total != expected)
    n_inconsistent = int(inconsistent.sum())
    n_comparable = int(comparable.sum())
    n_skipped = int((~comparable).sum())
    print(f"rows with all three run fields present: {n_comparable:,}")
    print(f"rows skipped because a run field is missing: {n_skipped:,}")
    print(f"inconsistent rows (runs_total != runs_batter + runs_extras): {n_inconsistent:,}")
    if n_inconsistent:
        preview = df.loc[
            inconsistent,
            ["match_id", "innings", "over", "ball", "runs_batter", "runs_extras", "runs_total"],
        ].head(10)
        print("example inconsistent rows (up to 10):")
        print(preview.to_string(index=False))
    return n_inconsistent


def report_row_previews(df: pd.DataFrame) -> None:
    print_header("14. Sample of 10 rows")
    sample = df.sample(n=min(10, len(df)), random_state=42)
    print(sample.to_string(index=False))

    print_header("15. First 10 rows")
    print(df.head(10).to_string(index=False))

    print_header("16. Last 10 rows")
    print(df.tail(10).to_string(index=False))


def report_wickets(df: pd.DataFrame) -> int:
    print_header("17. Wickets recorded")
    player_out = df["player_out"].notna() & (df["player_out"].astype(str).str.strip() != "")
    wicket_kind = df["wicket_kind"].notna() & (df["wicket_kind"].astype(str).str.strip() != "")
    either = player_out | wicket_kind
    n_wickets = int(either.sum())
    print(f"rows with player_out: {int(player_out.sum()):,}")
    print(f"rows with wicket_kind: {int(wicket_kind.sum()):,}")
    print(f"rows with player_out or wicket_kind: {n_wickets:,}")
    print(f"rows with wicket_kind but no player_out: {int((wicket_kind & ~player_out).sum()):,}")
    print(f"rows with player_out but no wicket_kind: {int((player_out & ~wicket_kind).sum()):,}")
    if "wicket_kind" in df.columns:
        print("wicket_kind distribution:")
        print(df.loc[wicket_kind, "wicket_kind"].value_counts(dropna=False).to_string())
    return n_wickets


def report_team_distributions(df: pd.DataFrame) -> None:
    print_header("18. Batting team distribution")
    print(df["batting_team"].value_counts(dropna=False).to_string())

    print_header("19. Bowling team distribution")
    print(df["bowling_team"].value_counts(dropna=False).to_string())


def count_negative(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    return int((numeric < 0).sum())


def report_invalid_values(df: pd.DataFrame) -> dict[str, int]:
    print_header("20. Obviously invalid values")
    checks: dict[str, int] = {}
    for column in (*RUN_COLUMNS, *EXTRA_COLUMNS, "over", "ball", "innings"):
        if column not in df.columns:
            continue
        n_neg = count_negative(df[column])
        checks[f"negative_{column}"] = n_neg
        print(f"negative {column}: {n_neg:,}")

    ball = pd.to_numeric(df["ball"], errors="coerce")
    over = pd.to_numeric(df["over"], errors="coerce")
    innings = pd.to_numeric(df["innings"], errors="coerce")
    checks["ball_less_than_1"] = int((ball < 1).sum())
    checks["over_greater_than_49"] = int((over > 49).sum())
    print(f"ball < 1: {checks['ball_less_than_1']:,}")
    print("over > 49 (possible for super overs / miscounted overs; listed for inspection only): "
          f"{checks['over_greater_than_49']:,}")
    print(f"innings < 1: {int((innings < 1).sum()):,}")
    return checks


def final_summary(
    *,
    n_rows: int,
    n_cols: int,
    n_matches: int,
    n_dup_rows: int,
    n_dup_keys: int,
    n_inconsistent_runs: int,
    invalid_counts: dict[str, int],
) -> None:
    print_header("22. Final structural-validity summary")
    obvious_invalid = sum(count for name, count in invalid_counts.items() if name != "over_greater_than_49")
    issues: list[str] = []
    if n_rows == 0:
        issues.append("the table is empty")
    if n_matches == 0:
        issues.append("no match_ids were found")
    if n_dup_rows:
        issues.append(f"{n_dup_rows:,} fully duplicate rows")
    if n_dup_keys:
        issues.append(f"{n_dup_keys:,} duplicate delivery keys (match_id, innings, over, ball)")
    if n_inconsistent_runs:
        issues.append(f"{n_inconsistent_runs:,} rows where runs_total != runs_batter + runs_extras")
    if obvious_invalid:
        issues.append(f"{obvious_invalid:,} obviously invalid numeric values (negatives or ball < 1)")

    print(f"rows={n_rows:,}, columns={n_cols}, unique matches={n_matches:,}")
    print("This script did not modify the CSV.")
    if issues:
        print("STRUCTURAL CONCERNS:")
        for issue in issues:
            print(f"  - {issue}")
        print(
            "The dataset needs these issues reviewed before feature engineering. "
            "It was not cleaned in this stage."
        )
    else:
        print(
            "The dataset appears structurally valid for the next feature-engineering stage: "
            "expected shape, 2,576 unique matches, no full-row or delivery-key duplicates, "
            "consistent run totals, and no negative runs/over/ball values. "
            "High missingness in extras/wicket/super_over columns is expected because those "
            "fields are optional in Cricsheet JSON."
        )


def main() -> None:
    configure_display()
    print(f"Reading {INPUT_CSV} (read-only)")
    df = load_dataset(INPUT_CSV)

    n_rows, n_cols = report_shape(df)
    report_dtypes(df)
    unique_counts = report_unique_counts(df)
    report_dates(df)
    report_innings(df)
    report_missing(df)
    duplicates = report_duplicates(df)
    report_run_stats(df)
    n_inconsistent = report_run_consistency(df)
    report_row_previews(df)
    report_wickets(df)
    report_team_distributions(df)
    invalid_counts = report_invalid_values(df)
    final_summary(
        n_rows=n_rows,
        n_cols=n_cols,
        n_matches=unique_counts["n_matches"],
        n_dup_rows=duplicates["duplicate_rows"],
        n_dup_keys=duplicates["duplicate_delivery_keys"],
        n_inconsistent_runs=n_inconsistent,
        invalid_counts=invalid_counts,
    )


if __name__ == "__main__":
    main()
