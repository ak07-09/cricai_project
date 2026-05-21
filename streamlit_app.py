
#  CRICMETRICS PRO


import streamlit as st
import pandas as pd
import numpy as np
import time

from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import get_logger

logger = get_logger(__name__)


# PAGE CONFIG


st.set_page_config(
    page_title="CricMetrics Pro",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# CSS


st.markdown("""
<style>

/* GOOGLE FONTS */
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=Rajdhani:wght@500;600;700&display=swap');

/* RESET */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

/* REMOVE HORIZONTAL SCROLL */
html {
    scroll-behavior: smooth;
    overflow-x: hidden;
}

body {
    overflow-x: hidden;
}

[data-testid="stAppViewContainer"] {
    overflow-x: hidden !important;
    background: #060c1a;
    min-height: 100vh;
    font-family: 'Inter', sans-serif;
}

/* REMOVE STREAMLIT DEFAULTS */
[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    display: none;
}

#MainMenu {
    display: none;
}

footer {
    display: none;
}

/* MAIN CONTAINER */
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}

/* HERO */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
}

.hero-badge {
    display: inline-block;
    background: rgba(0,200,80,0.12);
    border: 1px solid rgba(0,200,80,0.35);
    color: #00c850;
    font-family: 'Rajdhani', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 20px;
    margin-bottom: 22px;
}

.hero-title {
    font-family: 'Oswald', sans-serif;
    font-size: 82px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.hero-title span {
    background: linear-gradient(135deg, #00c850 0%, #00ff88 50%, #ffb700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.45);
    margin-bottom: 30px;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 40px;
    flex-wrap: wrap;
}

.hero-stat {
    text-align: center;
}

.hero-stat-val {
    font-family: 'Oswald', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: #00c850;
}

.hero-stat-lbl {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.divider-line {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,80,0.3), transparent);
    margin: 10px 0 40px;
}

/* SECTION */
.section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00c850;
    margin-bottom: 14px;
}

.section-title {
    font-family: 'Oswald', sans-serif;
    font-size: 22px;
    color: white;
    margin-bottom: 20px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
}

/* INPUTS */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: white !important;
}

label[data-testid="stWidgetLabel"] {
    color: rgba(255,255,255,0.6) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #00c850 0%, #00a040 100%) !important;
    color: black !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px 30px !important;
    width: 100% !important;
}

/* RESULT PANEL */
.result-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 28px;
}

.team-row {
    margin-bottom: 22px;
}

.team-name {
    font-family: 'Oswald', sans-serif;
    font-size: 24px;
    color: white;
    margin-bottom: 10px;
}

.team-percent {
    font-family: 'Oswald', sans-serif;
    font-size: 34px;
    font-weight: 700;
}

.track {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    overflow: hidden;
}

.fill {
    height: 100%;
    border-radius: 10px;
}

.footer {
    text-align: center;
    padding: 50px 20px;
    color: rgba(255,255,255,0.3);
    font-size: 13px;
}

.main-wrap {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 28px 40px;
}

</style>
""", unsafe_allow_html=True)


# DATA


TEAMS = [
    "India", "Australia", "England", "Pakistan",
    "South Africa", "New Zealand", "Sri Lanka",
    "West Indies", "Bangladesh", "Afghanistan",
    "CSK", "MI", "RCB", "KKR", "GT",
    "RR", "DC", "SRH", "PBKS", "LSG"
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "MCG, Melbourne",
    "Lord's, London",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad"
]

TEAM_COLORS = {
    "CSK": "#f5c518",
    "MI": "#004b8d",
    "RCB": "#d40000",
    "KKR": "#3a225d",
    "India": "#ff9933",
    "Australia": "#ffdd00",
    "Pakistan": "#01411c",
}

def get_team_color(name):
    return TEAM_COLORS.get(name, "#00c850")


# LOAD MODEL


@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()


# HERO


st.markdown("""
<div class="hero">

    <div class="hero-badge">
        ⚡ Real-Time Cricket Win Probability Engine
    </div>

    <div class="hero-title">
        Cric<span>Metrics</span> Pro
    </div>

    <div class="hero-sub">
        Probabilistic T20 Win Prediction · Calibrated XGBoost · Ball-by-Ball Match States
    </div>

    <div class="hero-stats">

        <div class="hero-stat">
            <div class="hero-stat-val">6,883</div>
            <div class="hero-stat-lbl">Matches Trained</div>
        </div>

        <div class="hero-stat">
            <div class="hero-stat-val">387K</div>
            <div class="hero-stat-lbl">Ball-by-Ball States</div>
        </div>

        <div class="hero-stat">
            <div class="hero-stat-val">0.9179</div>
            <div class="hero-stat-lbl">ROC-AUC</div>
        </div>

        <div class="hero-stat">
            <div class="hero-stat-val">83%</div>
            <div class="hero-stat-lbl">Prediction Accuracy</div>
        </div>

        <div class="hero-stat">
            <div class="hero-stat-val">16</div>
            <div class="hero-stat-lbl">Features</div>
        </div>

    </div>

</div>

<div class="divider-line"></div>
""", unsafe_allow_html=True)


# MAIN


st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="section-label">Step 01</div>
<div class="section-title">Match Setup</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    batting_team = st.selectbox("Batting Team", TEAMS)

with c2:
    bowling_team = st.selectbox("Bowling Team", TEAMS, index=1)

venue = st.selectbox("Venue", VENUES)

st.markdown("<br>", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)

with s1:
    target = st.number_input("Target", value=180)

with s2:
    current_score = st.number_input("Current Score", value=95)

with s3:
    overs_completed = st.number_input("Overs", value=10.0)

with s4:
    wickets = st.number_input("Wickets", value=3)

st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button("⚡ ANALYSE WIN PROBABILITY")


if predict_btn:

    result = pipeline.predict({
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "match_type": "T20",
        "target": target,
        "current_score": current_score,
        "overs_completed": overs_completed,
        "wickets": wickets
    })

    bat_pct = result["batting_win"]
    bowl_pct = result["bowling_win"]

    bat_color = get_team_color(batting_team)
    bowl_color = get_team_color(bowling_team)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-label">Live Analysis</div>
    <div class="section-title">Win Probability</div>

    <div class="result-panel">

        <div class="team-row">

            <div class="team-name">
                {batting_team}
            </div>

            <div class="team-percent" style="color:{bat_color}">
                {bat_pct:.1f}%
            </div>

            <div class="track">
                <div class="fill"
                     style="width:{bat_pct}%;
                            background:{bat_color};">
                </div>
            </div>

        </div>

        <div class="team-row">

            <div class="team-name">
                {bowling_team}
            </div>

            <div class="team-percent" style="color:{bowl_color}">
                {bowl_pct:.1f}%
            </div>

            <div class="track">
                <div class="fill"
                     style="width:{bowl_pct}%;
                            background:{bowl_color};">
                </div>
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">

    CRICMETRICS PRO · Calibrated XGBoost ·
    6,883 Matches · Ball-by-Ball Win Probability

</div>
""", unsafe_allow_html=True)