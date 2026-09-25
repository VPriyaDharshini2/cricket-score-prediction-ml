from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pickle
import pandas as pd
import streamlit as st

from src.input_validation import validate_score_inputs
from src.prediction_reliability import get_prediction_reliability


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cricket Analytics ML",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SCORE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_xgboost_model.pkl"
)

PLAYER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_player_classifier.pkl"
)


# ============================================================
# MODEL METRICS
# ============================================================

SCORE_MAE = 32.9531
SCORE_RMSE = 44.2526
SCORE_R2 = 0.5229

CHRONOLOGICAL_MAE = 36.4162
CHRONOLOGICAL_RMSE = 48.2668
CHRONOLOGICAL_R2 = 0.4892

PLAYER_TEST_ACCURACY = 0.9925
PLAYER_TEST_F1 = 0.9925

PLAYER_CV_ACCURACY = 0.9761
PLAYER_CV_STD = 0.0090


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

SCORE_DEFAULTS = {
    "score_innings": 1,
    "score_over": 30,
    "score_ball": 3.0,
    "score_total": 145.0,
    "score_wickets": 3,
    "score_last5": 35.0,
    "score_crr": 4.79,
    "score_remaining": 19.5,
    "score_last3": 20.0,
    "score_last10": 65.0,
    "score_boundary": 0.12,
    "score_dot": 0.45,
    "score_recent_wickets": 1.0,
    "score_season": "2024",
    "score_venue": "Example Stadium",
    "score_batting_team": "India",
    "score_bowling_team": "Australia",
    "score_batter": "Example Batter",
    "score_non_striker": "Example Non-Striker",
    "score_bowler": "Example Bowler",
}

PLAYER_DEFAULTS = {
    "player_innings": 50,
    "player_runs": 1500.0,
    "player_average": 35.0,
    "player_strike_rate": 85.0,
    "player_wickets": 10.0,
    "player_economy": 5.0,
    "player_bowling_average": 35.0,
    "player_bowling_strike_rate": 42.0,
    "player_batting_basra": 0.0,
    "player_bowling_basra": 0.0,
}


def initialize_session_state():
    """Initialize all Streamlit session-state values."""

    for key, value in SCORE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for key, value in PLAYER_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if "last_prediction_signature" not in st.session_state:
        st.session_state.last_prediction_signature = None

    if "last_score_prediction" not in st.session_state:
        st.session_state.last_score_prediction = None

    if "last_player_prediction" not in st.session_state:
        st.session_state.last_player_prediction = None


def reset_score_inputs():
    """Reset score-prediction inputs."""

    for key, value in SCORE_DEFAULTS.items():
        st.session_state[key] = value

    st.session_state.last_score_prediction = None
    st.session_state.last_prediction_signature = None


def reset_player_inputs():
    """Reset player-classification inputs."""

    for key, value in PLAYER_DEFAULTS.items():
        st.session_state[key] = value

    st.session_state.last_player_prediction = None


initialize_session_state()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ==========================================================
       ORCA-INSPIRED / AURORA INDIGO
       Professional SaaS analytics aesthetic based on the
       supplied reference: deep navy surfaces, luminous indigo
       accents, bright typography, compact cards and clear
       visual hierarchy.
       ========================================================== */

    :root {
        --bg: #080914;
        --bg-2: #0d0e1c;
        --sidebar: #0b0b17;

        --panel: #121222;
        --panel-2: #17172a;
        --panel-3: #1d1c35;

        --border: #282744;
        --border-soft: #1f1f36;

        --text: #f4f2ff;
        --text-soft: #b9b7ca;
        --muted: #85839b;
        --faint: #5f5d73;

        --purple: #8b5cf6;
        --purple-bright: #a66cff;
        --purple-dark: #5b2bbd;

        --lavender: #c7b5ff;
        --cyan: #62d9d0;

        --success: #58c78c;
        --warning: #e8a85c;
        --danger: #e36d7d;
    }

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 78% 0%,
                rgba(139, 92, 246, 0.075),
                transparent 25%
            ),
            radial-gradient(
                circle at 15% 90%,
                rgba(98, 217, 208, 0.035),
                transparent 22%
            ),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    /* ----------------------------------------------------------
       Sidebar
       ---------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid #1e1e32;
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    [data-testid="stSidebar"] .stCaption {
        color: var(--muted);
    }

    [data-testid="stSidebar"] hr {
        border-color: #222238;
        margin: 1.2rem 0;
    }

    [data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #17172a,
            #121222
        );
        border: 1px solid var(--border);
        box-shadow: none;
    }

    [data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    /* ----------------------------------------------------------
       Hero
       ---------------------------------------------------------- */

    .hero {
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                135deg,
                #15142b 0%,
                #17152e 55%,
                #111225 100%
            );

        border: 1px solid #302c54;
        border-radius: 12px;

        padding: 2rem 2.2rem;
        margin-bottom: 1.45rem;

        box-shadow:
            0 18px 45px rgba(0, 0, 0, 0.28);
    }

    .hero::before {
        content: "";
        position: absolute;

        width: 260px;
        height: 260px;

        right: -80px;
        top: -120px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(139, 92, 246, 0.20),
                transparent 68%
            );

        pointer-events: none;
    }

    .hero::after {
        content: "";
        position: absolute;

        width: 140px;
        height: 140px;

        right: 145px;
        bottom: -90px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(98, 217, 208, 0.07),
                transparent 68%
            );

        pointer-events: none;
    }

    .hero-title {
        position: relative;
        z-index: 2;

        color: var(--text);

        font-size: 2.2rem;
        font-weight: 780;

        letter-spacing: -0.04em;

        margin: 0;
    }

    .hero-subtitle {
        position: relative;
        z-index: 2;

        color: var(--text-soft);

        font-size: 0.96rem;
        line-height: 1.55;

        margin-top: 0.45rem;
        margin-bottom: 0;
    }

    .hero-tag {
        position: relative;
        z-index: 2;

        display: inline-block;

        margin-top: 0.9rem;

        padding: 0.34rem 0.68rem;

        border: 1px solid #51427f;
        border-radius: 5px;

        color: var(--lavender);

        background: rgba(139, 92, 246, 0.10);

        font-size: 0.66rem;
        font-weight: 750;

        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    /* ----------------------------------------------------------
       Typography
       ---------------------------------------------------------- */

    .section-title {
        color: var(--text);

        font-size: 1.35rem;
        font-weight: 760;

        letter-spacing: -0.025em;

        margin-top: 0.65rem;
        margin-bottom: 0.25rem;
    }

    .section-description {
        color: var(--muted);

        font-size: 0.86rem;
        line-height: 1.55;

        margin-bottom: 1.05rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }

    p, li {
        color: var(--text-soft);
    }

    /* ----------------------------------------------------------
       Cards
       ---------------------------------------------------------- */

    .card {
        background:
            linear-gradient(
                145deg,
                var(--panel-2),
                var(--panel)
            );

        border: 1px solid var(--border);
        border-radius: 10px;

        padding: 1.2rem 1.3rem;

        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.16);

        margin-bottom: 0.9rem;
    }

    .card-title {
        color: var(--muted);

        font-size: 0.66rem;
        font-weight: 800;

        letter-spacing: 0.095em;
        text-transform: uppercase;

        margin-bottom: 0.45rem;
    }

    .card-value {
        color: var(--text);

        font-size: 1.45rem;
        font-weight: 780;
    }

    .card-caption {
        color: var(--muted);

        font-size: 0.78rem;
        line-height: 1.55;

        margin-top: 0.3rem;
    }

    .card-caption strong {
        color: var(--text-soft);
    }

    /* ----------------------------------------------------------
       Prediction result
       ---------------------------------------------------------- */

    .prediction-card {
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                145deg,
                #19182f 0%,
                #121222 100%
            );

        border: 1px solid #343056;
        border-radius: 12px;

        padding: 2.15rem;

        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.30);

        text-align: center;

        margin: 1.1rem 0 1.35rem 0;
    }

    .prediction-card::after {
        content: "";
        position: absolute;

        width: 260px;
        height: 260px;

        left: 50%;
        top: 50%;

        transform: translate(-50%, -50%);

        background:
            radial-gradient(
                circle,
                rgba(139, 92, 246, 0.09),
                transparent 68%
            );

        pointer-events: none;
    }

    .prediction-label {
        position: relative;
        z-index: 2;

        color: var(--lavender);

        font-size: 0.66rem;
        font-weight: 800;

        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .prediction-score {
        position: relative;
        z-index: 2;

        color: var(--text);

        font-size: 4.25rem;
        font-weight: 850;

        line-height: 1.03;

        margin: 0.42rem 0;

        letter-spacing: -0.055em;
    }

    .prediction-unit {
        position: relative;
        z-index: 2;

        color: var(--muted);

        font-size: 0.84rem;
        font-weight: 650;
    }

    .score-flow {
        position: relative;
        z-index: 2;

        color: var(--muted);

        font-size: 0.94rem;

        margin-top: 0.9rem;
    }

    .score-flow strong {
        color: var(--text);
    }

    /* ----------------------------------------------------------
       Reliability
       ---------------------------------------------------------- */

    .reliability-card {
        background:
            linear-gradient(
                145deg,
                #151a1b,
                #121517
            );

        border: 1px solid #293b3b;
        border-left: 3px solid var(--cyan);

        border-radius: 9px;

        padding: 1.1rem 1.25rem;

        margin-top: 0.95rem;
    }

    .reliability-title {
        color: var(--muted);

        font-weight: 800;
        font-size: 0.66rem;

        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .reliability-value {
        color: var(--cyan);

        font-size: 1.18rem;
        font-weight: 800;

        margin-top: 0.25rem;
    }

    .reliability-text {
        color: var(--muted);

        font-size: 0.81rem;
        line-height: 1.6;

        margin-top: 0.3rem;
    }

    .reliability-text strong {
        color: var(--text-soft);
    }

    .stage-badge {
        display: inline-block;

        background: rgba(139, 92, 246, 0.10);

        color: var(--lavender);

        border: 1px solid #4b3e72;

        border-radius: 5px;

        padding: 0.35rem 0.68rem;

        font-size: 0.73rem;
        font-weight: 750;
    }

    /* ----------------------------------------------------------
       Inputs
       ---------------------------------------------------------- */

    label,
    [data-testid="stWidgetLabel"] p {
        color: var(--text-soft) !important;
    }

    div[data-baseweb="input"] {
        background: #141421 !important;

        border: 1px solid #34334a !important;

        border-radius: 7px !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: var(--purple) !important;

        box-shadow:
            0 0 0 1px var(--purple) !important;
    }

    div[data-baseweb="input"] input {
        color: var(--text) !important;
        background: transparent !important;
    }

    div[data-baseweb="select"] > div {
        background: #141421 !important;

        border: 1px solid #34334a !important;

        border-radius: 7px !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }

    [data-baseweb="popover"] {
        background: #17172a !important;

        border: 1px solid var(--border) !important;
    }

    [role="option"] {
        background: #17172a !important;
        color: var(--text) !important;
    }

    [role="option"]:hover {
        background: #282544 !important;
    }

    /* ----------------------------------------------------------
       Buttons
       ---------------------------------------------------------- */

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 7px;

        font-weight: 720;

        min-height: 2.6rem;

        border: 1px solid #393752;

        background: #1a1a2b;

        color: var(--text);

        transition:
            background 0.18s ease,
            border-color 0.18s ease,
            transform 0.14s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #24233b;

        border-color: #5a527c;

        transform: translateY(-1px);
    }

    button[kind="primary"] {
        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #8b5cf6
            ) !important;

        border: 1px solid #9b72f5 !important;

        color: #ffffff !important;

        box-shadow:
            0 8px 24px rgba(124, 58, 237, 0.23);
    }

    button[kind="primary"]:hover {
        background:
            linear-gradient(
                135deg,
                #8b5cf6,
                #a66cff
            ) !important;

        border-color: #b18aff !important;
    }

    /* ----------------------------------------------------------
       Metrics
       ---------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                #18182b,
                #131322
            );

        border: 1px solid var(--border);

        padding: 0.95rem;

        border-radius: 9px;

        box-shadow: none;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--success) !important;
    }

    /* ----------------------------------------------------------
       Dataframe
       ---------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);

        border-radius: 9px;

        overflow: hidden;
    }

    [data-testid="stDataFrame"] * {
        color: var(--text-soft);
    }

    /* ----------------------------------------------------------
       Alerts
       ---------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 8px;

        border: 1px solid var(--border);

        background: #151522;

        color: var(--text-soft);
    }

    div[data-testid="stAlert"] p {
        color: var(--text-soft) !important;
    }

    /* ----------------------------------------------------------
       Tabs
       ---------------------------------------------------------- */

    button[data-baseweb="tab"] {
        color: var(--muted) !important;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--text) !important;
    }

    div[data-baseweb="tab-highlight"] {
        background: var(--purple) !important;
    }

    /* ----------------------------------------------------------
       Expanders
       ---------------------------------------------------------- */

    [data-testid="stExpander"] {
        background: #141421;

        border: 1px solid var(--border);

        border-radius: 8px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: var(--text-soft) !important;
    }

    /* ----------------------------------------------------------
       Dividers / footer
       ---------------------------------------------------------- */

    hr {
        border-color: var(--border-soft) !important;
    }

    .footer {
        text-align: center;

        color: var(--faint);

        font-size: 0.73rem;
        line-height: 1.7;

        margin-top: 3rem;

        padding-top: 1.35rem;

        border-top: 1px solid var(--border-soft);
    }

    header[data-testid="stHeader"] {
        background: rgba(8, 9, 20, 0.94);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """Load trained models once per Streamlit session."""

    if not SCORE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Score model not found: {SCORE_MODEL_PATH}"
        )

    if not PLAYER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Player model not found: {PLAYER_MODEL_PATH}"
        )

    with open(SCORE_MODEL_PATH, "rb") as file:
        score_model = pickle.load(file)

    with open(PLAYER_MODEL_PATH, "rb") as file:
        player_model = pickle.load(file)

    return score_model, player_model


try:
    score_model, player_model = load_models()
except Exception as exc:
    st.error(
        "Unable to load the trained models."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
             Cricket Analytics ML
        </div>
        <div class="hero-subtitle">
            ODI Score Prediction & Player Role Classification
            using Machine Learning
        </div>
        <div class="hero-tag">
            Temporal Feature-Driven ML Framework
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "##  Cricket Analytics ML"
    )

    st.caption(
        "Machine Learning Demonstration"
    )

    st.markdown("---")

    st.markdown("### Score Prediction")

    st.metric(
        "MAE",
        f"{SCORE_MAE:.2f}",
        help="Mean Absolute Error from the randomized match-level evaluation.",
    )

    st.metric(
        "RMSE",
        f"{SCORE_RMSE:.2f}",
        help="Root Mean Squared Error from the randomized match-level evaluation.",
    )

    st.metric(
        "R²",
        f"{SCORE_R2:.4f}",
        help="Coefficient of determination from the randomized match-level evaluation.",
    )

    st.markdown("---")

    st.markdown("### Player Classification")

    st.metric(
        "Test Accuracy",
        f"{PLAYER_TEST_ACCURACY * 100:.2f}%",
    )

    st.metric(
        "5-Fold CV",
        f"{PLAYER_CV_ACCURACY * 100:.2f}%",
    )

    st.caption(
        f"CV variation: ±{PLAYER_CV_STD * 100:.2f}%"
    )

    st.markdown("---")

    st.caption(
        "Final trained models are used for inference."
    )


# ============================================================
# MAIN NAVIGATION
# ============================================================

score_tab, player_tab, info_tab = st.tabs(
    [
        " Score Prediction",
        " Player Classification",
        " Model & Dataset",
    ]
)


# ============================================================
# SCORE PREDICTION TAB
# ============================================================

with score_tab:

    st.markdown(
        '<div class="section-title">ODI Score Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Enter the current match state and recent performance
            information to estimate the final innings score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    reset_col, history_col = st.columns(
        [1, 4]
    )

    with reset_col:

        st.button(
            "Reset Inputs",
            on_click=reset_score_inputs,
            width="stretch",
            key="score_reset_inputs",
        )

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form(
        "score_prediction_form",
        clear_on_submit=False,
    ):

        st.markdown(
            "### Match State"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            innings = st.number_input(
                "Innings",
                min_value=1,
                max_value=2,
                step=1,
                key="score_innings",
            )

        with c2:
            over = st.number_input(
                "Current Over",
                min_value=0,
                max_value=49,
                step=1,
                key="score_over",
            )

        with c3:
            ball = st.number_input(
                "Ball",
                min_value=0.1,
                step=0.1,
                format="%.1f",
                key="score_ball",
            )

        with c4:
            wickets = st.number_input(
                "Wickets Lost",
                min_value=0,
                max_value=10,
                step=1,
                key="score_wickets",
            )

        c5, c6, c7, c8 = st.columns(4)

        with c5:
            total_score = st.number_input(
                "Current Score",
                min_value=0.0,
                step=1.0,
                key="score_total",
            )

        with c6:
            current_run_rate = st.number_input(
                "Current Run Rate",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="score_crr",
            )

        with c7:
            overs_remaining = st.number_input(
                "Overs Remaining",
                min_value=0.0,
                max_value=50.0,
                step=0.1,
                format="%.1f",
                key="score_remaining",
            )

        with c8:
            recent_5_over_wickets = st.number_input(
                "Recent 5-Over Wickets",
                min_value=0.0,
                max_value=10.0,
                step=1.0,
                key="score_recent_wickets",
            )

        st.markdown(
            "### Recent Performance"
        )

        r1, r2, r3, r4, r5 = st.columns(5)

        with r1:
            last_3_overs_score = st.number_input(
                "Last 3 Overs",
                min_value=0.0,
                step=1.0,
                key="score_last3",
            )

        with r2:
            last_5_overs_score = st.number_input(
                "Last 5 Overs",
                min_value=0.0,
                step=1.0,
                key="score_last5",
            )

        with r3:
            last_10_overs_score = st.number_input(
                "Last 10 Overs",
                min_value=0.0,
                step=1.0,
                key="score_last10",
            )

        with r4:
            boundary_rate = st.number_input(
                "Boundary Rate",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key="score_boundary",
            )

        with r5:
            dot_ball_rate = st.number_input(
                "Dot-Ball Rate",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key="score_dot",
            )

        st.markdown(
            "### Match Context"
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            season = st.text_input(
                "Season",
                key="score_season",
            )

        with m2:
            venue = st.text_input(
                "Venue",
                key="score_venue",
            )

        with m3:
            batting_team = st.text_input(
                "Batting Team",
                key="score_batting_team",
            )

        m4, m5, m6, m7 = st.columns(4)

        with m4:
            bowling_team = st.text_input(
                "Bowling Team",
                key="score_bowling_team",
            )

        with m5:
            batter = st.text_input(
                "Batter",
                key="score_batter",
            )

        with m6:
            non_striker = st.text_input(
                "Non-Striker",
                key="score_non_striker",
            )

        with m7:
            bowler = st.text_input(
                "Bowler",
                key="score_bowler",
            )

        st.markdown("")

        predict_score = st.form_submit_button(
            " Predict Final Score",
            width="stretch",
            type="primary",
        )

    # --------------------------------------------------------
    # SCORE PREDICTION
    # --------------------------------------------------------

    if predict_score:

        validation_errors = validate_score_inputs(
            innings=innings,
            over=over,
            ball=ball,
            total_score=total_score,
            wickets=wickets,
            last_5_overs_score=last_5_overs_score,
            current_run_rate=current_run_rate,
            overs_remaining=overs_remaining,
            last_3_overs_score=last_3_overs_score,
            last_10_overs_score=last_10_overs_score,
            boundary_rate=boundary_rate,
            dot_ball_rate=dot_ball_rate,
            recent_5_over_wickets=recent_5_over_wickets,
            season=season,
            venue=venue,
            batting_team=batting_team,
            bowling_team=bowling_team,
            batter=batter,
            non_striker=non_striker,
            bowler=bowler,
        )

        if validation_errors:

            st.error(
                "Please correct the following inputs:"
            )

            for error in validation_errors:
                st.write(f" {error}")

            st.session_state.last_score_prediction = None

        else:

            prediction_input = pd.DataFrame(
                [
                    {
                        "innings": innings,
                        "over": over,
                        "ball": ball,
                        "total_score": total_score,
                        "wickets": wickets,
                        "last_5_overs_score": last_5_overs_score,
                        "current_run_rate": current_run_rate,
                        "overs_remaining": overs_remaining,
                        "last_3_overs_score": last_3_overs_score,
                        "last_10_overs_score": last_10_overs_score,
                        "boundary_rate": boundary_rate,
                        "dot_ball_rate": dot_ball_rate,
                        "recent_5_over_wickets": recent_5_over_wickets,
                        "season": str(season),
                        "venue": str(venue),
                        "batting_team": str(batting_team),
                        "bowling_team": str(bowling_team),
                        "batter": str(batter),
                        "non_striker": str(non_striker),
                        "bowler": str(bowler),
                    }
                ]
            )

            try:

                predicted_score = float(
                    score_model.predict(
                        prediction_input
                    )[0]
                )

                predicted_score = max(
                    0.0,
                    predicted_score,
                )

                reliability = (
                    get_prediction_reliability(
                        over=float(over),
                        overs_remaining=float(
                            overs_remaining
                        ),
                    )
                )

                current_score = float(
                    total_score
                )

                projected_change = (
                    predicted_score
                    - current_score
                )

                timestamp = datetime.now().strftime(
                    "%H:%M:%S"
                )

                # Streamlit reruns the script after widget interactions.
                # Use the complete model input as a deterministic signature
                # so the same prediction cannot be inserted twice.
                prediction_signature = prediction_input.to_json(
                    orient="records"
                )

                history_item = {
                    "Time": timestamp,
                    "Current Score": round(
                        current_score
                    ),
                    "Wickets": int(wickets),
                    "Over": f"{float(over):g}",
                    "Predicted Score": round(
                        predicted_score
                    ),
                    "Stage": reliability[
                        "stage"
                    ],
                    "Reliability": reliability[
                        "reliability"
                    ],
                }

                if (
                    prediction_signature
                    != st.session_state.last_prediction_signature
                ):
                    st.session_state.prediction_history.insert(
                        0,
                        history_item,
                    )

                    # Keep the history compact.
                    st.session_state.prediction_history = (
                        st.session_state.prediction_history[
                            :10
                        ]
                    )

                    st.session_state.last_prediction_signature = (
                        prediction_signature
                    )

                st.session_state.last_score_prediction = {
                    "predicted_score": predicted_score,
                    "current_score": current_score,
                    "projected_change": projected_change,
                    "reliability": reliability,
                    "timestamp": timestamp,
                }

            except Exception as exc:

                st.error(
                    "The prediction could not be generated."
                )

                st.exception(exc)

                st.session_state.last_score_prediction = None

    # --------------------------------------------------------
    # DISPLAY SCORE RESULT
    # --------------------------------------------------------

    result = (
        st.session_state.last_score_prediction
    )

    if result is not None:

        predicted_score = result[
            "predicted_score"
        ]

        current_score = result[
            "current_score"
        ]

        projected_change = result[
            "projected_change"
        ]

        reliability = result[
            "reliability"
        ]

        st.markdown(
            "---"
        )

        st.markdown(
            '<div class="section-title">Prediction Summary</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Model output and historical error context for the '
            'current match state.'
            '</div>',
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(3)

        with p1:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        Current Score
                    </div>
                    <div class="card-value">
                        {current_score:.0f}
                    </div>
                    <div class="card-caption">
                        Runs scored so far
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p2:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        Prediction Stage
                    </div>
                    <div class="card-value">
                        <span class="stage-badge">
                            {reliability["stage"]}
                        </span>
                    </div>
                    <div class="card-caption">
                        {overs_remaining:.1f} overs remaining
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p3:

            change_text = (
                f"+{projected_change:.0f}"
                if projected_change >= 0
                else f"{projected_change:.0f}"
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        Projected Change
                    </div>
                    <div class="card-value">
                        {change_text}
                    </div>
                    <div class="card-caption">
                        Runs relative to current score
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Predicted Final Score
                </div>
                <div class="prediction-score">
                    {predicted_score:.0f}
                </div>
                <div class="prediction-unit">
                    runs
                </div>
                <div class="score-flow">
                    Current score
                    <strong>{current_score:.0f}</strong>
                    &nbsp;→&nbsp;
                    Predicted final
                    <strong>{predicted_score:.0f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="reliability-card">
                <div class="reliability-title">
                    Prediction Reliability
                </div>
                <div class="reliability-value">
                    {reliability["reliability"]}
                </div>
                <div class="reliability-text">
                    Historical MAE for similar overs-remaining
                    situations: approximately
                    <strong>
                        ±{reliability["historical_mae"]:.2f} runs
                    </strong>.
                    <br>
                    Historical MAE for this innings stage:
                    <strong>
                        ±{reliability["stage_mae"]:.2f} runs
                    </strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------------

    st.markdown(
        "---"
    )

    history_header_col, history_button_col = st.columns(
        [5, 1]
    )

    with history_header_col:

        st.markdown(
            '<div class="section-title">Prediction History</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Predictions generated during this Streamlit session.'
            '</div>',
            unsafe_allow_html=True,
        )

    with history_button_col:

        if st.button(
            "Clear History",
            width="stretch",
            key="clear_prediction_history",
        ):

            st.session_state.prediction_history = []
            st.session_state.last_prediction_signature = None

            st.rerun()

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn(
                    "Time", width="small"
                ),
                "Current Score": st.column_config.NumberColumn(
                    "Current Score", format="%d"
                ),
                "Wickets": st.column_config.NumberColumn(
                    "Wickets", format="%d"
                ),
                "Over": st.column_config.TextColumn(
                    "Over", width="small"
                ),
                "Predicted Score": st.column_config.NumberColumn(
                    "Predicted Score", format="%d"
                ),
                "Stage": st.column_config.TextColumn("Stage"),
                "Reliability": st.column_config.TextColumn(
                    "Reliability"
                ),
            },
        )

    else:

        st.info(
            "No predictions have been generated yet. "
            "Your prediction history will appear here."
        )


# ============================================================
# PLAYER CLASSIFICATION TAB
# ============================================================

with player_tab:

    st.markdown(
        '<div class="section-title">'
        'Player Role Classification'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Enter batting and bowling statistics to classify
            the player's statistical role.
        </div>
        """,
        unsafe_allow_html=True,
    )

    reset_player_col, _ = st.columns(
        [1, 4]
    )

    with reset_player_col:

        st.button(
            "Reset Inputs",
            on_click=reset_player_inputs,
            width="stretch",
            key="player_reset_inputs",
        )

    with st.form(
        "player_classification_form",
        clear_on_submit=False,
    ):

        st.markdown(
            "### Player Statistics"
        )

        a1, a2, a3, a4, a5 = st.columns(5)

        with a1:
            player_innings = st.number_input(
                "Innings",
                min_value=0,
                step=1,
                key="player_innings",
            )

        with a2:
            player_runs = st.number_input(
                "Runs",
                min_value=0.0,
                step=10.0,
                key="player_runs",
            )

        with a3:
            player_average = st.number_input(
                "Batting Average",
                min_value=0.0,
                step=0.1,
                key="player_average",
            )

        with a4:
            player_strike_rate = st.number_input(
                "Strike Rate",
                min_value=0.0,
                step=0.1,
                key="player_strike_rate",
            )

        with a5:
            player_wickets = st.number_input(
                "Wickets",
                min_value=0.0,
                step=1.0,
                key="player_wickets",
            )

        b1, b2, b3, b4, b5 = st.columns(5)

        with b1:
            player_economy = st.number_input(
                "Economy",
                min_value=0.0,
                step=0.1,
                key="player_economy",
            )

        with b2:
            player_bowling_average = st.number_input(
                "Bowling Average",
                min_value=0.0,
                step=0.1,
                key="player_bowling_average",
            )

        with b3:
            player_bowling_strike_rate = st.number_input(
                "Bowling Strike Rate",
                min_value=0.0,
                step=0.1,
                key="player_bowling_strike_rate",
            )

        with b4:
            player_batting_basra = st.number_input(
                "Batting BASRA",
                step=0.1,
                key="player_batting_basra",
            )

        with b5:
            player_bowling_basra = st.number_input(
                "Bowling BASRA",
                step=0.1,
                key="player_bowling_basra",
            )

        predict_player = st.form_submit_button(
            " Classify Player Role",
            width="stretch",
            type="primary",
        )

    if predict_player:

        if (
            player_innings < 0
            or player_runs < 0
            or player_average < 0
            or player_strike_rate < 0
            or player_wickets < 0
            or player_economy < 0
            or player_bowling_average < 0
            or player_bowling_strike_rate < 0
        ):

            st.error(
                "Player statistics cannot be negative."
            )

            st.session_state.last_player_prediction = None

        else:

            player_input = pd.DataFrame(
                [
                    {
                        "innings": player_innings,
                        "runs": player_runs,
                        "average": player_average,
                        "strike_rate": player_strike_rate,
                        "wickets": player_wickets,
                        "economy": player_economy,
                        "bowling_average": player_bowling_average,
                        "bowling_strike_rate": player_bowling_strike_rate,
                        "batting_basra": player_batting_basra,
                        "bowling_basra": player_bowling_basra,
                    }
                ]
            )

            try:

                role_prediction = int(
                    player_model.predict(
                        player_input
                    )[0]
                )

                role_mapping = {
                    0: "Batsman",
                    1: "Batting All-rounder",
                    2: "Bowler",
                    3: "Bowling All-rounder",
                }

                role = role_mapping.get(
                    role_prediction,
                    "Unknown",
                )

                st.session_state.last_player_prediction = role

            except Exception as exc:

                st.error(
                    "The player role could not be classified."
                )

                st.exception(exc)

                st.session_state.last_player_prediction = None

    # --------------------------------------------------------
    # PLAYER RESULT
    # --------------------------------------------------------

    player_result = (
        st.session_state.last_player_prediction
    )

    if player_result:

        role_explanations = {
            "Batsman": (
                "Primarily contributes through batting performance "
                "and has comparatively limited bowling contribution."
            ),
            "Batting All-rounder": (
                "Combines a strong batting contribution with "
                "a meaningful secondary bowling role."
            ),
            "Bowler": (
                "Primarily contributes through bowling performance "
                "with comparatively limited batting contribution."
            ),
            "Bowling All-rounder": (
                "Combines a strong bowling contribution with "
                "a meaningful secondary batting role."
            ),
        }

        explanation = role_explanations.get(
            player_result,
            "Role classification generated by the trained model.",
        )

        st.markdown(
            "---"
        )

        st.markdown(
            '<div class="section-title">'
            'Classification Result'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Predicted Player Role
                </div>
                <div class="prediction-score"
                     style="font-size: 3rem;">
                    {player_result}
                </div>
                <div class="prediction-unit">
                    XGBoost classification output
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="reliability-card">
                <div class="reliability-title">
                    Role Explanation
                </div>
                <div class="reliability-value">
                    {player_result}
                </div>
                <div class="reliability-text">
                    {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "The player-role labels are derived from statistical "
            "role categories. The classifier learns to reproduce "
            "those rule-based labels from player performance features."
        )


# ============================================================
# MODEL & DATASET TAB
# ============================================================

with info_tab:

    st.markdown(
        '<div class="section-title">'
        'Model & Dataset'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Technical information about the trained models,
            evaluation results and the ODI dataset used by the project.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### Model Information"
    )

    score_col, player_col = st.columns(2)

    with score_col:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    ODI Score Prediction
                </div>
                <div class="card-value">
                    XGBoost Regressor
                </div>
                <div class="card-caption">
                    20 model input features
                    <br><br>
                    Randomized match-level evaluation:
                    <br>
                    MAE: <strong>{SCORE_MAE:.2f}</strong>
                    &nbsp;|&nbsp;
                    RMSE: <strong>{SCORE_RMSE:.2f}</strong>
                    &nbsp;|&nbsp;
                    R²: <strong>{SCORE_R2:.4f}</strong>
                    <br><br>
                    Chronological evaluation:
                    <br>
                    MAE: <strong>{CHRONOLOGICAL_MAE:.2f}</strong>
                    &nbsp;|&nbsp;
                    RMSE: <strong>{CHRONOLOGICAL_RMSE:.2f}</strong>
                    &nbsp;|&nbsp;
                    R²: <strong>{CHRONOLOGICAL_R2:.4f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with player_col:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    Player Role Classification
                </div>
                <div class="card-value">
                    XGBoost Classifier
                </div>
                <div class="card-caption">
                    Four role categories
                    <br><br>
                    Test Accuracy:
                    <strong>
                        {PLAYER_TEST_ACCURACY * 100:.2f}%
                    </strong>
                    <br>
                    Weighted F1:
                    <strong>
                        {PLAYER_TEST_F1 * 100:.2f}%
                    </strong>
                    <br><br>
                    5-Fold CV Accuracy:
                    <strong>
                        {PLAYER_CV_ACCURACY * 100:.2f}%
                    </strong>
                    ± {PLAYER_CV_STD * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # FEATURE PIPELINE
    # --------------------------------------------------------

    st.markdown(
        "### ️ Score Prediction Feature Pipeline"
    )

    pipeline_cols = st.columns(5)

    pipeline_items = [
        ("01", "Match State", "Score, wickets, over and ball"),
        ("02", "Recent Form", "Recent scoring and wicket activity"),
        ("03", "Temporal Features", "3, 5 and 10-over scoring patterns"),
        ("04", "XGBoost", "Regression model"),
        ("05", "Final Score", "Predicted ODI innings total"),
    ]

    for column, item in zip(
        pipeline_cols,
        pipeline_items,
    ):

        number, title, description = item

        with column:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        {number}
                    </div>
                    <div style="
                        font-size:1.05rem;
                        font-weight:750;
                        color:#e5edf7;
                        margin-bottom:0.4rem;">
                        {title}
                    </div>
                    <div class="card-caption">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.markdown(
        "### ️ About the Dataset"
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "ODI Matches",
            "2,576",
        )

    with d2:
        st.metric(
            "Delivery Records",
            "1.37M",
        )

    with d3:
        st.metric(
            "Score Features",
            "20",
        )

    with d4:
        st.metric(
            "Player Roles",
            "4",
        )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                Data Source
            </div>
            <div class="card-value">
                Cricsheet ODI JSON
            </div>
            <div class="card-caption">
                The project processes ball-by-ball ODI match data
                into a delivery-level machine learning dataset.
                The score-prediction pipeline combines match-state,
                recent-performance and temporal features.
                The player-classification pipeline uses batting and
                bowling statistics to classify players into four
                statistical role categories.
                <br><br>
                The chronological robustness experiment evaluates
                historical matches against later matches, with the
                training period ending on 5 August 2022 and the
                testing period beginning on 7 August 2022.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # ROLE INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### Player Role Categories"
    )

    role_data = pd.DataFrame(
        {
            "Role": [
                "Batsman",
                "Batting All-rounder",
                "Bowler",
                "Bowling All-rounder",
            ],
            "Description": [
                "Primary batting contribution.",
                "Strong batting contribution with secondary bowling ability.",
                "Primary bowling contribution.",
                "Strong bowling contribution with secondary batting ability.",
            ],
        }
    )

    st.dataframe(
        role_data,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    st.markdown(
        "### ️ Technology Stack"
    )

    tech_cols = st.columns(5)

    technologies = [
        ("Python", "Data processing & ML"),
        ("Pandas", "Data preparation"),
        ("Scikit-learn", "Preprocessing & evaluation"),
        ("XGBoost", "Prediction & classification"),
        ("Streamlit", "Interactive frontend"),
    ]

    for column, (name, description) in zip(
        tech_cols,
        technologies,
    ):

        with column:

            st.markdown(
                f"""
                <div class="card">
                    <div style="
                        font-size:1rem;
                        font-weight:750;
                        color:#e5edf7;">
                        {name}
                    </div>
                    <div class="card-caption">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Cricket Analytics ML &nbsp;&nbsp;
        ODI Score Prediction & Player Role Classification
        <br>
        Temporal Feature-Driven Machine Learning Framework
    </div>
    """,
    unsafe_allow_html=True,
)