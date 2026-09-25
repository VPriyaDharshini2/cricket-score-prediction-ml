"""
Prediction reliability context for ODI score prediction.

This module does NOT change the trained ML model.

It provides historical error context based on the project's
existing match-level error analysis.

The output is descriptive:
    - innings stage
    - historical MAE
    - reliability label

The reliability label is a UI interpretation and should not
be treated as a probability or confidence percentage.
"""

from __future__ import annotations


# ============================================================
# HISTORICAL ERROR VALUES
# ============================================================

# MAE by innings stage from the completed error analysis.
STAGE_MAE = {
    "Powerplay": 47.8774,
    "Middle Overs": 31.8969,
    "Death Overs": 12.5847,
}


# MAE by overs remaining from the completed error analysis.
OVERS_REMAINING_MAE = {
    "0-10": 12.7464,
    "10-20": 23.2308,
    "20-30": 31.6104,
    "30-40": 39.8694,
    "40+": 47.9123,
}


# ============================================================
# INNINGS STAGE
# ============================================================

def get_innings_stage(over: float) -> str:
    """
    Determine the ODI innings stage from the current over.

    0-9   -> Powerplay
    10-39 -> Middle Overs
    40-49 -> Death Overs
    """

    if over < 0:
        raise ValueError(
            "Over cannot be negative."
        )

    if over < 10:
        return "Powerplay"

    if over < 40:
        return "Middle Overs"

    return "Death Overs"


# ============================================================
# OVERS REMAINING CATEGORY
# ============================================================

def get_overs_remaining_category(
    overs_remaining: float,
) -> str:
    """
    Convert overs remaining into the same categories used
    in the project's error analysis.
    """

    if overs_remaining < 0:
        raise ValueError(
            "Overs remaining cannot be negative."
        )

    if overs_remaining <= 10:
        return "0-10"

    if overs_remaining <= 20:
        return "10-20"

    if overs_remaining <= 30:
        return "20-30"

    if overs_remaining <= 40:
        return "30-40"

    return "40+"


# ============================================================
# RELIABILITY LABEL
# ============================================================

def get_reliability_label(
    historical_mae: float,
) -> str:
    """
    Convert historical MAE into a simple descriptive label.

    This is NOT a probability or confidence score.

    Lower historical MAE:
        Higher Reliability

    Medium historical MAE:
        Moderate Reliability

    Higher historical MAE:
        Lower Reliability
    """

    if historical_mae < 20:
        return "Higher Reliability"

    if historical_mae < 35:
        return "Moderate Reliability"

    return "Lower Reliability"


# ============================================================
# MAIN RELIABILITY FUNCTION
# ============================================================

def get_prediction_reliability(
    over: float,
    overs_remaining: float,
) -> dict:
    """
    Return historical error context for the current match state.

    The final MAE shown to the user is based primarily on the
    overs-remaining category because this directly represents
    how much of the innings is still available for scoring.

    The innings-stage MAE is also returned for transparency.
    """

    stage = get_innings_stage(
        over
    )

    overs_category = (
        get_overs_remaining_category(
            overs_remaining
        )
    )

    historical_mae = (
        OVERS_REMAINING_MAE[
            overs_category
        ]
    )

    stage_mae = (
        STAGE_MAE[
            stage
        ]
    )

    reliability = (
        get_reliability_label(
            historical_mae
        )
    )

    return {
        "stage": stage,
        "overs_remaining_category": overs_category,
        "historical_mae": historical_mae,
        "stage_mae": stage_mae,
        "reliability": reliability,
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    examples = [
        {
            "over": 5,
            "overs_remaining": 44.0,
        },
        {
            "over": 25,
            "overs_remaining": 24.0,
        },
        {
            "over": 45,
            "overs_remaining": 4.0,
        },
    ]

    print("=" * 60)
    print("PREDICTION RELIABILITY TEST")
    print("=" * 60)

    for example in examples:

        result = get_prediction_reliability(
            over=example["over"],
            overs_remaining=example[
                "overs_remaining"
            ],
        )

        print("\nInput:")
        print(example)

        print("\nResult:")
        print(
            f"Stage: "
            f"{result['stage']}"
        )

        print(
            f"Overs remaining category: "
            f"{result['overs_remaining_category']}"
        )

        print(
            f"Historical MAE: "
            f"{result['historical_mae']:.2f} runs"
        )

        print(
            f"Stage MAE: "
            f"{result['stage_mae']:.2f} runs"
        )

        print(
            f"Reliability: "
            f"{result['reliability']}"
        )