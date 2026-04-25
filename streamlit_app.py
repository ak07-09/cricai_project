"""
streamlit_app.py — Cricket Win Predictor (fixed)
=================================================
Changes vs. original
---------------------
* Calls predict() with the correct parameter names that match the training
  pipeline  (current_score, target, overs_completed, wickets, match_type).
* Displays a probability curve over overs (requires repeated calls — fast
  because no retraining happens).
* Shows raw feature values so users can sanity-check the model inputs.
* No hardcoded win_index or simulation logic.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import get_logger

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏏 CricAI — Win Probability",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .main-header {
        text-align: center; color: white; padding: 40px 20px 20px;
    }
    .main-header h1 { font-size: 46px; font-weight: 900; }
    .content-box {
        background: rgba(255,255,255,0.95); border-radius: 16px;
        padding: 30px; margin: 20px auto; max-width: 950px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47) !important;
        color: white !important; font-weight: 800 !important;
        font-size: 15px !important; padding: 14px 35px !important;
        border-radius: 10px !important; border: none !important;
        width: 100%;
    }
    .result-bar {
        height: 28px; border-radius: 6px; display: flex;
        align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

try:
    pipeline = load_pipeline()
    model_ok = True
except Exception as exc:
    st.error(f"⚠️ Model not loaded: {exc}. Run the training pipeline first.")
    model_ok = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏏 CricAI — Live Win Probability</h1>
    <p style="opacity:0.8">Data-driven • Ball-by-ball trained • Calibrated XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### ⚡ Match Snapshot")

    TEAMS = sorted([
        'Afghanistan', 'Australia', 'Bangladesh', 'CSK', 'DC', 'England',
        'GT', 'India', 'KKR', 'LSG', 'MI', 'New Zealand', 'Pakistan',
        'PBKS', 'RCB', 'RR', 'South Africa', 'Sri Lanka', 'SRH', 'West Indies',
    ])
    VENUES = sorted([
        'Auckland', 'Bangalore', 'Brisbane', 'Cape Town', 'Chennai', 'Colombo',
        'Delhi', 'Dubai', 'Durban', 'Hyderabad', 'Johannesburg', 'Kolkata',
        'London', 'Lucknow', 'Melbourne', 'Mumbai', 'Perth', 'Pune',
        'Sydney', 'Wankhede Stadium',
    ])

    c1, c2 = st.columns(2)
    with c1:
        batting_team = st.selectbox("🏏 Batting Team (Chasing)", TEAMS, index=TEAMS.index("India") if "India" in TEAMS else 0)
    with c2:
        bowling_team = st.selectbox("🎯 Bowling Team (Defending)", TEAMS, index=TEAMS.index("Australia") if "Australia" in TEAMS else 1)

    c3, c4 = st.columns(2)
    with c3:
        venue = st.selectbox("📍 Venue", VENUES)
    with c4:
        match_type = st.selectbox("🏆 Match Type", ["T20"])

    st.markdown("---")
    st.markdown("### 📊 Current Match State")

    c5, c6 = st.columns(2)
    with c5:
        target = st.number_input("🎯 Target (runs to win)", min_value=50, max_value=300, value=180)
    with c6:
        current_score = st.number_input("📈 Current Score", min_value=0, max_value=300, value=100)

    c7, c8 = st.columns(2)
    with c7:
        overs_completed = st.number_input(
            "⏱️ Overs Completed (e.g. 10.3 = 10 overs 3 balls)",
            min_value=0.0, max_value=19.5, step=0.1, value=10.0
        )
    with c8:
        wickets_lost = st.number_input("💀 Wickets Lost", min_value=0, max_value=10, value=3)

    predict_btn = st.button("🎯 GET WIN PROBABILITY")

    if predict_btn:
        if not model_ok:
            st.error("Train the model first!")
        elif batting_team == bowling_team:
            st.error("Batting and bowling teams must be different.")
        else:
            with st.spinner("Running model …"):
                features = {
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "venue": venue,
                    "match_type": match_type,
                    "target": target,
                    "current_score": current_score,
                    "overs_completed": overs_completed,
                    "wickets": wickets_lost,
                }
                try:
                    result = pipeline.predict(features)
                    bat_pct = float(result["batting_win"])
                    bowl_pct = float(result["bowling_win"])

                    st.markdown("---")
                    st.markdown("### 🏆 Win Probability")

                    st.markdown(f"""
                    <div style="margin-bottom:12px">
                        <b>{batting_team} (Batting)</b>
                        <div class="result-bar" style="
                            width:{bat_pct}%; max-width:100%;
                            background:linear-gradient(90deg,#4CAF50,#81C784);
                            margin-top:4px">
                            {bat_pct:.1f}%
                        </div>
                    </div>
                    <div>
                        <b>{bowling_team} (Bowling)</b>
                        <div class="result-bar" style="
                            width:{bowl_pct}%; max-width:100%;
                            background:linear-gradient(90deg,#e94560,#FF8A80);
                            margin-top:4px">
                            {bowl_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    winner = batting_team if bat_pct > 50 else bowling_team
                    st.success(f"🏆 Model favours **{winner}** — {max(bat_pct, bowl_pct):.1f}%")

                    # ── Probability Curve over Overs ──────────────────────────
                    st.markdown("---")
                    st.markdown("### 📈 Win Probability Curve (this match state across overs)")
                    curve_data = []
                    total_legal_balls = int(overs_completed) * 6 + round((overs_completed % 1) * 10)
                    for ball in range(6, total_legal_balls + 1, 6):
                        ov = ball / 6
                        # scale score proportionally (simplistic projection)
                        score_at = int(current_score * ball / max(total_legal_balls, 1))
                        try:
                            p = pipeline.predict({
                                **features,
                                "overs_completed": ov,
                                "current_score": score_at,
                            })
                            curve_data.append({"Over": ov, "Win %": p["batting_win"]})
                        except Exception:
                            pass

                    if curve_data:
                        curve_df = pd.DataFrame(curve_data).set_index("Over")
                        st.line_chart(curve_df, height=220)

                    # ── Raw feature debug panel ───────────────────────────────
                    with st.expander("🔍 Model Input Features (debug)"):
                        legal_balls = int(overs_completed) * 6 + round((overs_completed % 1) * 10)
                        balls_left = max(1, 120 - legal_balls)
                        runs_req = max(0, target - current_score)
                        crr = (current_score * 6) / max(legal_balls, 1)
                        rrr = (runs_req * 6) / balls_left
                        debug = {
                            "balls_bowled": legal_balls,
                            "balls_left": balls_left,
                            "runs_required": runs_req,
                            "current_run_rate": round(crr, 2),
                            "required_run_rate": round(rrr, 2),
                            "rr_ratio": round(min(crr / max(rrr, 0.1), 10), 3),
                            "wickets_in_hand": 10 - wickets_lost,
                            "match_progress": round(legal_balls / 120, 3),
                        }
                        st.json(debug)

                except Exception as exc:
                    st.error(f"Prediction error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.6); padding:20px; font-size:12px">
    🏏 CricAI | XGBoost + Isotonic Calibration | Trained on 4 000+ T20 matches
</div>
""", unsafe_allow_html=True)
