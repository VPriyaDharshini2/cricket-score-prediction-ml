"""
Input validation utilities for the Cricket Analytics ML application.

These checks validate user-entered ODI match-state values before they
are passed to the trained score-prediction model.

This module does not modify the trained model or its preprocessing.
"""

from __future__ import annotations


def validate_score_inputs(
    innings,
    over,
    ball,
    total_score,
    wickets,
    last_5_overs_score,
    current_run_rate,
    overs_remaining,
    last_3_overs_score,
    last_10_overs_score,
    boundary_rate,
    dot_ball_rate,
    recent_5_over_wickets,
    season,
    venue,
    batting_team,
    bowling_team,
    batter,
    non_striker,
    bowler,
):
    """
    Validate all score-prediction inputs.

    Returns:
        list[str]: validation error messages.

    An empty list means all inputs are valid.
    """

    errors = []

    # --------------------------------------------------------
    # Basic numeric validation
    # --------------------------------------------------------

    try:
        innings = int(innings)
    except (TypeError, ValueError):
        errors.append("Innings must be a valid integer.")
        innings = None

    if innings is not None and innings not in (1, 2):
        errors.append("Innings must be either 1 or 2.")

    try:
        over = float(over)
    except (TypeError, ValueError):
        errors.append("Over must be a valid number.")
        over = None

    if over is not None and not (0 <= over <= 49):
        errors.append("Over must be between 0 and 49.")

    try:
        ball = float(ball)
    except (TypeError, ValueError):
        errors.append("Ball must be a valid number.")
        ball = None

    if ball is not None and ball <= 0:
        errors.append("Ball must be greater than 0.")

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    numeric_non_negative = {
        "Total score": total_score,
        "Last 5 overs score": last_5_overs_score,
        "Current run rate": current_run_rate,
        "Overs remaining": overs_remaining,
        "Last 3 overs score": last_3_overs_score,
        "Last 10 overs score": last_10_overs_score,
        "Recent 5-over wickets": recent_5_over_wickets,
    }

    converted_values = {}

    for name, value in numeric_non_negative.items():

        try:
            value = float(value)
            converted_values[name] = value
        except (TypeError, ValueError):
            errors.append(
                f"{name} must be a valid number."
            )
            continue

        if value < 0:
            errors.append(
                f"{name} cannot be negative."
            )

    # --------------------------------------------------------
    # Wickets
    # --------------------------------------------------------

    try:
        wickets = int(wickets)
    except (TypeError, ValueError):
        errors.append(
            "Wickets must be a valid integer."
        )
        wickets = None

    if wickets is not None and not (0 <= wickets <= 10):
        errors.append(
            "Wickets must be between 0 and 10."
        )

    # --------------------------------------------------------
    # Overs remaining
    # --------------------------------------------------------

    if "Overs remaining" in converted_values:

        if converted_values["Overs remaining"] > 50:
            errors.append(
                "Overs remaining cannot exceed 50."
            )

    # --------------------------------------------------------
    # Rate validation
    # --------------------------------------------------------

    if "Boundary rate" not in numeric_non_negative:
        pass

    try:
        boundary_rate = float(boundary_rate)

        if not (0 <= boundary_rate <= 1):
            errors.append(
                "Boundary rate must be between 0 and 1."
            )

    except (TypeError, ValueError):

        errors.append(
            "Boundary rate must be a valid number."
        )

    try:
        dot_ball_rate = float(dot_ball_rate)

        if not (0 <= dot_ball_rate <= 1):
            errors.append(
                "Dot-ball rate must be between 0 and 1."
            )

    except (TypeError, ValueError):

        errors.append(
            "Dot-ball rate must be a valid number."
        )

    # --------------------------------------------------------
    # Recent wickets
    # --------------------------------------------------------

    if (
        "Recent 5-over wickets"
        in converted_values
    ):

        if converted_values[
            "Recent 5-over wickets"
        ] > 10:

            errors.append(
                "Recent 5-over wickets cannot exceed 10."
            )

    # --------------------------------------------------------
    # Categorical fields
    # --------------------------------------------------------

    categorical_fields = {
        "Season": season,
        "Venue": venue,
        "Batting team": batting_team,
        "Bowling team": bowling_team,
        "Batter": batter,
        "Non-striker": non_striker,
        "Bowler": bowler,
    }

    for name, value in categorical_fields.items():

        if value is None or not str(value).strip():

            errors.append(
                f"{name} cannot be empty."
            )

    # --------------------------------------------------------
    # Logical consistency checks
    # --------------------------------------------------------

    if (
        innings == 1
        and wickets is not None
        and wickets > 10
    ):
        errors.append(
            "Invalid wicket count for the innings."
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return errors


def is_valid_score_input(**kwargs):
    """
    Convenience function.

    Returns:
        True  -> all inputs are valid
        False -> at least one validation error exists
    """

    return len(
        validate_score_inputs(**kwargs)
    ) == 0


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("INPUT VALIDATION TEST")
    print("=" * 60)

    valid_input = {
        "innings": 1,
        "over": 30,
        "ball": 3,
        "total_score": 145,
        "wickets": 3,
        "last_5_overs_score": 35,
        "current_run_rate": 4.79,
        "overs_remaining": 19.5,
        "last_3_overs_score": 20,
        "last_10_overs_score": 65,
        "boundary_rate": 0.12,
        "dot_ball_rate": 0.45,
        "recent_5_over_wickets": 1,
        "season": "2024",
        "venue": "Example Stadium",
        "batting_team": "India",
        "bowling_team": "Australia",
        "batter": "Example Batter",
        "non_striker": "Example Non-Striker",
        "bowler": "Example Bowler",
    }

    errors = validate_score_inputs(
        **valid_input
    )

    print("\nValid input test:")

    if not errors:
        print("PASS - Input is valid.")
    else:
        print("FAIL")
        for error in errors:
            print(f"- {error}")

    invalid_input = {
        **valid_input,
        "innings": 3,
        "over": 55,
        "total_score": -10,
        "wickets": 12,
        "boundary_rate": 1.5,
    }

    errors = validate_score_inputs(
        **invalid_input
    )

    print("\nInvalid input test:")

    if errors:
        print("PASS - Invalid input was detected.")

        for error in errors:
            print(f"- {error}")

    else:
        print("FAIL - Invalid input was not detected.")