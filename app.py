import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cricket Analytics ML",
    page_icon="🏏",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #cccccc;
        margin-top: 20px;
    }

    .result-number {
        font-size: 42px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_score_model():

    with open(
        SCORE_MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


@st.cache_resource
def load_player_model():

    with open(
        PLAYER_MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


score_model = load_score_model()
player_model = load_player_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏏 Cricket Analytics ML</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning for ODI Score Prediction and Player Role Classification
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

tab_score, tab_player = st.tabs(
    [
        "🏏 Score Prediction",
        "👤 Player Classification"
    ]
)


# ============================================================
# SCORE PREDICTION
# ============================================================

with tab_score:

    st.header("🏏 Cricket Score Prediction")

    st.write(
        "Enter the current state of an ODI innings. "
        "The trained XGBoost model will estimate the final score."
    )

    # --------------------------------------------------------
    # MATCH STATE
    # --------------------------------------------------------

    st.subheader("Match State")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        innings = st.number_input(
            "Innings",
            min_value=1,
            max_value=2,
            value=1,
            step=1
        )

    with col2:

        over = st.number_input(
            "Over",
            min_value=0,
            max_value=49,
            value=20,
            step=1
        )

    with col3:

        ball = st.number_input(
            "Ball",
            min_value=1,
            max_value=6,
            value=1,
            step=1
        )

    with col4:

        total_score = st.number_input(
            "Current Score",
            min_value=0,
            max_value=600,
            value=100,
            step=1
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        wickets = st.number_input(
            "Wickets Lost",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

    with col2:

        current_run_rate = st.number_input(
            "Current Run Rate",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1
        )

    with col3:

        overs_remaining = st.number_input(
            "Overs Remaining",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=0.1
        )

    with col4:

        season = st.number_input(
            "Season",
            value=2024        )

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    st.subheader("Recent Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        last_3_overs_score = st.number_input(
            "Last 3 Overs Score",
            min_value=0,
            max_value=100,
            value=18,
            step=1
        )

    with col2:

        last_5_overs_score = st.number_input(
            "Last 5 Overs Score",
            min_value=0,
            max_value=150,
            value=30,
            step=1
        )

    with col3:

        last_10_overs_score = st.number_input(
            "Last 10 Overs Score",
            min_value=0,
            max_value=250,
            value=50,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        boundary_rate = st.number_input(
            "Boundary Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01
        )

    with col2:

        dot_ball_rate = st.number_input(
            "Dot Ball Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.01
        )

    with col3:

        recent_5_over_wickets = st.number_input(
            "Wickets in Last 5 Overs",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    # --------------------------------------------------------
    # MATCH INFORMATION
    # --------------------------------------------------------

    st.subheader("Match Information")

    col1, col2 = st.columns(2)

    with col1:

        venue = st.text_input(
            "Venue",
            value="Unknown"
        )

    with col2:

        batting_team = st.text_input(
            "Batting Team",
            value="India"
        )

    col1, col2 = st.columns(2)

    with col1:

        bowling_team = st.text_input(
            "Bowling Team",
            value="Australia"
        )

    with col2:

        batter = st.text_input(
            "Current Batter",
            value="Unknown"
        )

    col1, col2 = st.columns(2)

    with col1:

        non_striker = st.text_input(
            "Non-Striker",
            value="Unknown"
        )

    with col2:

        bowler = st.text_input(
            "Current Bowler",
            value="Unknown"
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    st.divider()

    predict_score = st.button(
        "🏏 Predict Final Score",
        type="primary",
        use_container_width=True
    )

    if predict_score:

        score_features = pd.DataFrame(
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
                    "bowler": str(bowler)
                }
            ]
        )

        try:

            prediction = score_model.predict(
                score_features
            )[0]

            prediction = max(
                0,
                round(float(prediction))
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div>Predicted Final Score</div>
                    <div class="result-number">
                        {prediction} runs
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)


# ============================================================
# PLAYER CLASSIFICATION
# ============================================================

with tab_player:

    st.header("👤 Player Role Classification")

    st.write(
        "Enter a player's performance statistics "
        "to classify their functional cricket role."
    )

    st.subheader("Player Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        p_innings = st.number_input(
            "Innings",
            min_value=0,
            max_value=500,
            value=100,
            step=1,
            key="p_innings"
        )

    with col2:

        runs = st.number_input(
            "Runs",
            min_value=0.0,
            max_value=30000.0,
            value=3000.0,
            step=10.0
        )

    with col3:

        average = st.number_input(
            "Batting Average",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            step=0.1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        strike_rate = st.number_input(
            "Strike Rate",
            min_value=0.0,
            max_value=300.0,
            value=85.0,
            step=0.1
        )

    with col2:

        p_wickets = st.number_input(
            "Wickets",
            min_value=0,
            max_value=1000,
            value=20,
            step=1,
            key="p_wickets"
        )

    with col3:

        economy = st.number_input(
            "Economy Rate",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        bowling_average = st.number_input(
            "Bowling Average",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=0.1
        )

    with col2:

        bowling_strike_rate = st.number_input(
            "Bowling Strike Rate",
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=0.1
        )

    with col3:

        batting_basra = st.number_input(
            "Batting BASRA",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1
        )

    bowling_basra = st.number_input(
        "Bowling BASRA",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )

    st.divider()

    classify_player = st.button(
        "👤 Classify Player",
        type="primary",
        use_container_width=True
    )

    if classify_player:

        player_features = pd.DataFrame(
            [
                {
                    "innings": p_innings,
                    "runs": runs,
                    "average": average,
                    "strike_rate": strike_rate,
                    "wickets": p_wickets,
                    "economy": economy,
                    "bowling_average": bowling_average,
                    "bowling_strike_rate": bowling_strike_rate,
                    "batting_basra": batting_basra,
                    "bowling_basra": bowling_basra
                }
            ]
        )

        try:

            prediction = player_model.predict(
                player_features
            )[0]

            role_mapping = {
                0: "Batsman",
                1: "Batting All-rounder",
                2: "Bowler",
                3: "Bowling All-rounder"
            }

            role = role_mapping.get(
                int(prediction),
                "Unknown"
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div>Predicted Player Role</div>
                    <div class="result-number">
                        {role}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as error:

            st.error(
                "Classification failed."
            )

            st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cricket Analytics ML | "
    "ODI Score Prediction and Player Role Classification"
)