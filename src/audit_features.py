from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "processed" / "score_prediction_dataset.csv"

df = pd.read_csv(CSV, low_memory=False)

print("=" * 60)
print("FEATURE AUDIT")
print("=" * 60)

# ---------------------------------------------------------
# 1. Basic information
# ---------------------------------------------------------

print(f"\nDataset shape: {df.shape}")
print(f"Unique matches: {df['match_id'].nunique():,}")
print(
    f"Unique innings: "
    f"{df.groupby(['match_id', 'innings']).ngroups:,}"
)

# ---------------------------------------------------------
# 2. Check that cumulative score starts correctly
# ---------------------------------------------------------

first_rows = (
    df.sort_values(["match_id", "innings", "over", "ball"])
      .groupby(["match_id", "innings"], as_index=False)
      .first()
)

print("\nFirst delivery of each innings:")
print(
    first_rows[
        [
            "match_id",
            "innings",
            "over",
            "ball",
            "total_score",
            "wickets",
            "last_5_overs_score",
        ]
    ].head(10).to_string(index=False)
)

# ---------------------------------------------------------
# 3. Check final score consistency
# ---------------------------------------------------------

last_rows = (
    df.sort_values(["match_id", "innings", "over", "ball"])
      .groupby(["match_id", "innings"], as_index=False)
      .last()
)

final_score_mismatch = (
    last_rows["total_score"] != last_rows["final_score"]
).sum()

print("\nFinal-score consistency:")
print(f"Mismatched innings: {final_score_mismatch}")

# ---------------------------------------------------------
# 4. Check last_5_overs_score is never greater
#    than the current total score
# ---------------------------------------------------------

invalid_recent_score = (
    df["last_5_overs_score"] > df["total_score"]
).sum()

print("\nLast-5-overs sanity:")
print(
    f"Rows where last_5_overs_score > total_score: "
    f"{invalid_recent_score}"
)

# ---------------------------------------------------------
# 5. Check negative values
# ---------------------------------------------------------

for column in [
    "total_score",
    "final_score",
    "wickets",
    "legal_balls",
    "overs_bowled",
    "overs_remaining",
    "last_5_overs_score",
]:
    negative = (df[column] < 0).sum()
    print(f"Negative {column}: {negative}")

# ---------------------------------------------------------
# 6. Check wickets never exceed 10
# ---------------------------------------------------------

print(f"\nMaximum wickets: {df['wickets'].max()}")

# ---------------------------------------------------------
# 7. Check current score never exceeds final score
# ---------------------------------------------------------

score_ahead = (
    df["total_score"] > df["final_score"]
).sum()

print(
    f"Rows where total_score > final_score: "
    f"{score_ahead}"
)

# ---------------------------------------------------------
# 8. Check innings are independent
# ---------------------------------------------------------

first_by_innings = (
    df.sort_values(["match_id", "innings", "over", "ball"])
      .groupby(["match_id", "innings"])
      .first()
)

bad_start_score = (
    first_by_innings["total_score"] !=
    first_by_innings["runs_total"]
).sum()

print(
    f"\nFirst-delivery cumulative-score mismatches: "
    f"{bad_start_score}"
)

# ---------------------------------------------------------
# 9. Check duplicate delivery identifiers
# ---------------------------------------------------------

duplicate_keys = df.duplicated(
    subset=[
        "match_id",
        "innings",
        "over",
        "ball",
        "actual_delivery",
    ]
).sum()

print(
    f"Duplicate delivery identifiers: "
    f"{duplicate_keys}"
)

# ---------------------------------------------------------
# 10. Check missing engineered features
# ---------------------------------------------------------

engineered = [
    "total_score",
    "final_score",
    "wickets",
    "legal_balls",
    "overs_bowled",
    "overs_remaining",
    "current_run_rate",
    "last_5_overs_score",
]

print("\nMissing engineered values:")

for column in engineered:
    print(
        f"{column}: "
        f"{df[column].isna().sum():,}"
    )

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)