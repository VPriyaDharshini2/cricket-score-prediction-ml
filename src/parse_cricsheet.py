"""
Flatten Cricsheet Men's ODI JSON match files into a delivery-level table.

This script only parses and flattens raw JSON. It does not engineer ML
features, create synthetic rows, or modify the original match files.

Inspected dataset: Cricsheet JSON 1.2.0. In that format each match is:

    { meta, info, innings: [ { team, overs: [ { over, deliveries: [...] } ] } ] }

Overs are 0-indexed. Each delivery typically includes batter, non_striker,
bowler, runs, and optional extras / wickets. Files currently live in
data/raw/; the script also looks in data/raw/odis_male_json/ if present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Paths are resolved from the project root (parent of src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREFERRED_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "odis_male_json"
FALLBACK_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "odi_deliveries.csv"

# Extra kinds present in this dump: wides, noballs, byes, legbyes, penalty.
EXTRA_KINDS = ("wides", "noballs", "byes", "legbyes", "penalty")


def resolve_raw_dir() -> Path:
    """Prefer data/raw/odis_male_json/; fall back to data/raw/ if needed."""
    if PREFERRED_RAW_DIR.is_dir():
        return PREFERRED_RAW_DIR
    return FALLBACK_RAW_DIR


def find_json_files(raw_dir: Path) -> list[Path]:
    """Return sorted .json match files. README.txt and other files are ignored."""
    return sorted(path for path in raw_dir.glob("*.json") if path.is_file())


def match_id_from_filename(path: Path) -> str:
    """Cricsheet match id is the JSON filename without the extension."""
    return path.stem


def _first_or_none(values: Any) -> Any:
    """Return the first element of a non-empty list, else None."""
    if isinstance(values, list) and values:
        return values[0]
    return None


def extract_match_metadata(info: dict[str, Any] | None) -> dict[str, Any]:
    """Pull optional match-level fields. Missing keys become None."""
    info = info or {}
    teams = info.get("teams")
    if not isinstance(teams, list):
        teams = []

    toss = info.get("toss") if isinstance(info.get("toss"), dict) else {}
    dates = info.get("dates") if isinstance(info.get("dates"), list) else []

    # ODIs are usually one day; multi-day files store every date. Use the start date.
    return {
        "season": info.get("season"),
        "date": dates[0] if dates else None,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team_1": teams[0] if len(teams) > 0 else None,
        "team_2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "_teams": teams,
    }


def bowling_team_for(batting_team: Any, teams: list[Any]) -> Any:
    """The bowling side is the other listed team. Do not guess if that is unclear."""
    if batting_team is None or not teams:
        return None
    others = [team for team in teams if team != batting_team]
    if len(others) == 1:
        return others[0]
    return None


def _join_names(values: Iterable[Any] | None) -> Any:
    """Join a list of names with '; '. Return None when there is nothing to join."""
    if not values:
        return None
    names = [str(item) for item in values if item is not None and str(item) != ""]
    return "; ".join(names) if names else None


def parse_wickets(wickets: Any) -> dict[str, Any]:
    """
    Flatten optional wickets (a list; at most two in this dump).

    JSON 1.2.0 wicket objects look like:
        { kind, player_out, fielders?: [{ name, substitute? }] }
    """
    if not isinstance(wickets, list) or not wickets:
        return {
            "player_out": None,
            "wicket_kind": None,
            "wicket_fielders": None,
        }

    kinds: list[str] = []
    players: list[str] = []
    fielders: list[str] = []

    for wicket in wickets:
        if not isinstance(wicket, dict):
            continue
        kind = wicket.get("kind")
        player_out = wicket.get("player_out")
        if kind is not None:
            kinds.append(str(kind))
        if player_out is not None:
            players.append(str(player_out))

        raw_fielders = wicket.get("fielders") or []
        if isinstance(raw_fielders, list):
            for fielder in raw_fielders:
                # 1.2.0 uses objects; older dumps used plain strings.
                if isinstance(fielder, dict):
                    name = fielder.get("name")
                    if name:
                        label = str(name)
                        if fielder.get("substitute"):
                            label = f"{label} (sub)"
                        fielders.append(label)
                elif fielder:
                    fielders.append(str(fielder))

    return {
        "player_out": _join_names(players),
        "wicket_kind": _join_names(kinds),
        "wicket_fielders": _join_names(fielders),
    }


def parse_extras(extras: Any) -> dict[str, Any]:
    """Copy extra-run kinds that exist on the delivery. Absent kinds stay None."""
    extras = extras if isinstance(extras, dict) else {}
    return {f"extras_{kind}": extras.get(kind) for kind in EXTRA_KINDS}


def parse_delivery(
    delivery: dict[str, Any],
    *,
    ball_in_over: int,
    over_number: Any,
) -> dict[str, Any]:
    """Flatten one delivery object into a single row dict (no match/innings keys yet)."""
    runs = delivery.get("runs") if isinstance(delivery.get("runs"), dict) else {}

    # non_boundary is only present when Cricsheet marks a four/six that was not a boundary.
    row: dict[str, Any] = {
        "over": over_number,
        "ball": ball_in_over,
        "actual_delivery": delivery.get("actual_delivery"),
        "batter": delivery.get("batter"),
        "non_striker": delivery.get("non_striker"),
        "bowler": delivery.get("bowler"),
        "runs_batter": runs.get("batter"),
        "runs_extras": runs.get("extras"),
        "runs_total": runs.get("total"),
        "non_boundary": runs.get("non_boundary"),
    }
    row.update(parse_extras(delivery.get("extras")))
    row.update(parse_wickets(delivery.get("wickets")))
    return row


def iter_overs(innings_obj: dict[str, Any]) -> Iterable[tuple[Any, list[Any]]]:
    """
    Yield (over_number, deliveries) for JSON 1.2.0 overs lists.

    Also tolerate an older deliveries-as-list-of-single-key-dicts shape if it appears.
    """
    overs = innings_obj.get("overs")
    if isinstance(overs, list):
        for over_obj in overs:
            if not isinstance(over_obj, dict):
                continue
            deliveries = over_obj.get("deliveries")
            if not isinstance(deliveries, list):
                deliveries = []
            yield over_obj.get("over"), deliveries
        return

    # Older Cricsheet JSON stored balls as [{"0.1": {...}}, ...].
    deliveries = innings_obj.get("deliveries")
    if isinstance(deliveries, list):
        grouped: dict[Any, list[Any]] = {}
        for item in deliveries:
            if not isinstance(item, dict) or not item:
                continue
            label, payload = next(iter(item.items()))
            try:
                over_number = int(str(label).split(".")[0])
            except (TypeError, ValueError):
                over_number = None
            grouped.setdefault(over_number, []).append(payload)
        for over_number, balls in grouped.items():
            yield over_number, balls


def unwrap_innings_entry(entry: Any, index: int) -> tuple[int, dict[str, Any]] | None:
    """
    Return (1-based innings number, innings dict).

    JSON 1.2.0 innings is a list of objects with a 'team' key. Older files used
    {"1st innings": {...}} wrappers.
    """
    if not isinstance(entry, dict):
        return None

    if "team" in entry or "overs" in entry or "deliveries" in entry:
        return index + 1, entry

    for key, value in entry.items():
        if isinstance(value, dict):
            innings_number = index + 1
            prefix = str(key).split()[0]
            if prefix.endswith("st") or prefix.endswith("nd") or prefix.endswith("rd") or prefix.endswith("th"):
                try:
                    innings_number = int("".join(ch for ch in prefix if ch.isdigit()) or innings_number)
                except ValueError:
                    pass
            return innings_number, value
    return None


def parse_match(path: Path) -> list[dict[str, Any]]:
    """Read one match JSON file and return a list of delivery rows."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    match_id = match_id_from_filename(path)
    metadata = extract_match_metadata(payload.get("info") if isinstance(payload, dict) else None)
    teams = metadata.pop("_teams")

    shared = {
        "match_id": match_id,
        "season": metadata["season"],
        "date": metadata["date"],
        "venue": metadata["venue"],
        "city": metadata["city"],
        "team_1": metadata["team_1"],
        "team_2": metadata["team_2"],
        "toss_winner": metadata["toss_winner"],
        "toss_decision": metadata["toss_decision"],
    }

    innings_list = payload.get("innings") if isinstance(payload, dict) else None
    if not isinstance(innings_list, list):
        return []

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(innings_list):
        unwrapped = unwrap_innings_entry(entry, index)
        if unwrapped is None:
            continue
        innings_number, innings_obj = unwrapped
        batting_team = innings_obj.get("team")
        bowling_team = bowling_team_for(batting_team, teams)
        super_over = innings_obj.get("super_over")

        for over_number, deliveries in iter_overs(innings_obj):
            for ball_index, delivery in enumerate(deliveries, start=1):
                if not isinstance(delivery, dict):
                    continue
                row = dict(shared)
                row["innings"] = innings_number
                row["batting_team"] = batting_team
                row["bowling_team"] = bowling_team
                row["super_over"] = super_over
                row.update(
                    parse_delivery(
                        delivery,
                        ball_in_over=ball_index,
                        over_number=over_number,
                    )
                )
                rows.append(row)
    return rows


COLUMN_ORDER = [
    "match_id",
    "season",
    "date",
    "venue",
    "city",
    "team_1",
    "team_2",
    "toss_winner",
    "toss_decision",
    "innings",
    "batting_team",
    "bowling_team",
    "super_over",
    "over",
    "ball",
    "actual_delivery",
    "batter",
    "non_striker",
    "bowler",
    "runs_batter",
    "runs_extras",
    "runs_total",
    "non_boundary",
    "extras_wides",
    "extras_noballs",
    "extras_byes",
    "extras_legbyes",
    "extras_penalty",
    "player_out",
    "wicket_kind",
    "wicket_fielders",
]


def parse_all_matches(json_files: list[Path]) -> tuple[pd.DataFrame, int, int, list[tuple[str, str]]]:
    """
    Parse every match file.

    Returns the combined dataframe, success count, failure count, and
    (filename, error) pairs for files that failed.
    """
    all_rows: list[dict[str, Any]] = []
    n_success = 0
    failures: list[tuple[str, str]] = []

    for path in json_files:
        try:
            all_rows.extend(parse_match(path))
            n_success += 1
        except Exception as exc:  # keep going; one bad file must not stop the dump
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
            print(f"Failed to parse {path.name}: {type(exc).__name__}: {exc}")

    dataframe = pd.DataFrame(all_rows, columns=COLUMN_ORDER)
    return dataframe, n_success, len(failures), failures


def save_deliveries(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Create data/processed/ if needed and write the CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def print_summary(
    *,
    n_found: int,
    n_success: int,
    n_failed: int,
    dataframe: pd.DataFrame,
    output_path: Path,
    raw_dir: Path,
) -> None:
    print(f"Raw JSON directory: {raw_dir}")
    print(f"Output CSV: {output_path}")
    print(f"JSON files found: {n_found}")
    print(f"Files successfully processed: {n_success}")
    print(f"Failed files: {n_failed}")
    print(f"Delivery rows generated: {len(dataframe)}")
    print(f"Final dataframe shape: {dataframe.shape}")
    print(f"Final column names: {list(dataframe.columns)}")


def main() -> None:
    raw_dir = resolve_raw_dir()
    json_files = find_json_files(raw_dir)
    n_found = len(json_files)

    dataframe, n_success, n_failed, _failures = parse_all_matches(json_files)
    save_deliveries(dataframe, OUTPUT_CSV)
    print_summary(
        n_found=n_found,
        n_success=n_success,
        n_failed=n_failed,
        dataframe=dataframe,
        output_path=OUTPUT_CSV,
        raw_dir=raw_dir,
    )


if __name__ == "__main__":
    main()
