"""
streamlit_app.py — CricAI Sports Analytics Dashboard (v5)
==========================================================
Stadium dark + neon sports aesthetic.
All 10 IPL + 23 international teams. Full venue list.
Dynamic live stat pills, Plotly gauge + probability curve.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import get_logger

logger = get_logger(__name__)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricAI — Win Probability Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

:root {
    --bg:      #060b14; --bgc:  #0d1520; --bgc2:  #0a1220;
    --border:  rgba(0,210,255,0.12);
    --neon:    #00d2ff; --green: #00ff87; --red: #ff3d5a; --amber: #ffb800;
    --text:    #ddeeff; --muted: #4a6a88; --dim: #1a3050;
    --fh:'Barlow Condensed',sans-serif;
    --fb:'Barlow',sans-serif;
    --fm:'Share Tech Mono',monospace;
}
html,body,[data-testid="stApp"],[data-testid="stMainBlockContainer"],.main {
    background:var(--bg) !important; color:var(--text) !important; }
[data-testid="block-container"] {
    padding:0 2rem 5rem !important;
    max-width:1380px !important; margin:0 auto !important; }
#MainMenu,footer,header,
[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:rgba(0,210,255,.3);border-radius:3px}

/* HERO */
.hero{position:relative;width:100%;
  background:linear-gradient(180deg,#060b14 0%,#071525 55%,#060b14 100%);
  border-bottom:1px solid var(--border);overflow:hidden}
.hgrid{position:absolute;inset:0;
  background-image:linear-gradient(rgba(0,210,255,.04) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,210,255,.04) 1px,transparent 1px);
  background-size:55px 55px;animation:drift 26s linear infinite}
@keyframes drift{to{transform:translateY(55px)}}
.hi{position:relative;text-align:center;padding:54px 24px 42px}
.hchip{display:inline-block;font-family:var(--fm);font-size:10px;
  letter-spacing:3px;color:var(--neon);text-transform:uppercase;
  border:1px solid rgba(0,210,255,.28);padding:5px 18px;border-radius:2px;
  background:rgba(0,210,255,.05);margin-bottom:20px}
.htitle{font-family:var(--fh) !important;font-size:clamp(52px,8vw,98px) !important;
  font-weight:900 !important;line-height:.92 !important;letter-spacing:-1px !important;
  color:var(--text) !important;text-transform:uppercase;margin:0 0 6px !important}
.htitle em{font-style:normal;color:var(--neon);text-shadow:0 0 44px rgba(0,210,255,.55)}
.hsub{font-family:var(--fb);font-size:14px;font-weight:300;
  color:var(--muted);letter-spacing:.5px;margin-top:14px}
.hsub b{color:var(--neon);font-weight:600}
.krow{display:flex;justify-content:center;gap:44px;margin-top:28px;flex-wrap:wrap}
.kpi{text-align:center}
.kn{font-family:var(--fh);font-size:30px;font-weight:900;color:var(--neon);line-height:1}
.kl{font-family:var(--fm);font-size:9px;color:var(--muted);letter-spacing:2px;
  text-transform:uppercase;margin-top:3px}

/* TICKER */
.ticker{background:#030609;border-top:1px solid var(--border);
  border-bottom:1px solid var(--border);padding:9px 0;
  overflow:hidden;white-space:nowrap;margin:0 -2rem}
.ti{display:inline-block;animation:scroll 38s linear infinite;
  font-family:var(--fm);font-size:11px;color:var(--muted);letter-spacing:1px}
.ti s{text-decoration:none;color:var(--neon);margin:0 10px}
@keyframes scroll{from{transform:translateX(100vw)}to{transform:translateX(-100%)}}

/* LABELS */
.slabel{font-family:var(--fm);font-size:9px;letter-spacing:3px;color:var(--neon);
  text-transform:uppercase;opacity:.7;margin-bottom:5px}
.stitle{font-family:var(--fh);font-size:21px;font-weight:700;color:var(--text);
  text-transform:uppercase;letter-spacing:.5px;
  border-left:3px solid var(--neon);padding-left:12px;margin-bottom:18px}

/* CARDS */
.card{background:var(--bgc);border:1px solid var(--border);border-radius:7px;
  padding:22px;position:relative;overflow:hidden;margin-bottom:4px}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--neon),transparent);opacity:.35}
.ndiv{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:22px 0}

/* INPUTS */
[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label{
  font-family:var(--fm) !important;font-size:9px !important;
  letter-spacing:2.5px !important;color:var(--neon) !important;
  text-transform:uppercase !important;opacity:.8 !important}
[data-testid="stSelectbox"]>div>div,
[data-testid="stNumberInput"] input{
  background:#080f1c !important;border:1px solid rgba(0,210,255,.18) !important;
  border-radius:4px !important;color:var(--text) !important;
  font-family:var(--fb) !important;font-size:14px !important}
[data-testid="stSelectbox"]>div>div:hover,
[data-testid="stNumberInput"] input:focus{
  border-color:rgba(0,210,255,.45) !important;
  box-shadow:0 0 10px rgba(0,210,255,.08) !important}
[data-testid="stRadio"]>div{gap:8px !important}
[data-testid="stRadio"]>div>label{
  background:rgba(0,210,255,.05) !important;
  border:1px solid rgba(0,210,255,.2) !important;
  border-radius:4px !important;padding:8px 16px !important}

/* BUTTON */
[data-testid="stButton"]>button{
  width:100% !important;
  background:linear-gradient(135deg,#00c8f8 0%,#0080bb 100%) !important;
  color:#000 !important;font-family:var(--fh) !important;
  font-size:19px !important;font-weight:900 !important;
  letter-spacing:3px !important;text-transform:uppercase !important;
  padding:18px 32px !important;border:none !important;border-radius:4px !important;
  box-shadow:0 0 28px rgba(0,210,255,.22) !important;transition:all .18s !important}
[data-testid="stButton"]>button:hover{
  background:linear-gradient(135deg,#22d8ff 0%,#009ad4 100%) !important;
  box-shadow:0 0 48px rgba(0,210,255,.42) !important;transform:translateY(-2px) !important}

/* PROB BOX */
.pbox{background:var(--bgc2);border:1px solid var(--border);border-radius:7px;padding:26px;margin-top:20px}
.phead{font-family:var(--fm);font-size:9px;letter-spacing:3px;color:var(--muted);margin-bottom:20px}
.prow{margin-bottom:18px}
.pname{font-family:var(--fh);font-size:19px;font-weight:700;color:var(--text);
  text-transform:uppercase;letter-spacing:1px;
  display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}
.ppct{font-family:var(--fh);font-size:30px;font-weight:900;line-height:1}
.ptrack{height:13px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden}
.pfill{height:100%;border-radius:2px;position:relative}
.pfill::after{content:'';position:absolute;top:0;right:0;width:3px;height:100%;
  background:rgba(255,255,255,.55);border-radius:2px}
.wbanner{text-align:center;padding:18px;margin-top:18px;border-radius:5px;
  background:linear-gradient(135deg,rgba(0,255,135,.07),rgba(0,210,255,.07));
  border:1px solid rgba(0,255,135,.22)}
.wlabel{font-family:var(--fm);font-size:9px;letter-spacing:3px;
  color:var(--green);text-transform:uppercase;margin-bottom:7px}
.wname{font-family:var(--fh);font-size:34px;font-weight:900;color:var(--green);
  text-transform:uppercase;letter-spacing:2px;text-shadow:0 0 28px rgba(0,255,135,.4)}
.wconf{font-family:var(--fm);font-size:11px;color:rgba(0,255,135,.5);
  letter-spacing:2px;margin-top:5px}

/* STAT PILLS */
.pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.pill{background:rgba(0,210,255,.05);border:1px solid rgba(0,210,255,.14);
  border-radius:4px;padding:9px 14px;flex:1;min-width:80px;text-align:center}
.pl{font-family:var(--fm);font-size:8px;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;margin-bottom:3px}
.pv{font-family:var(--fh);font-size:21px;font-weight:800;color:var(--neon);line-height:1}

/* PLACEHOLDER */
.placeholder{background:var(--bgc);border:1px solid var(--border);
  border-radius:7px;padding:64px 32px;text-align:center}
.ph-icon{font-size:54px;margin-bottom:14px}
.ph-t{font-family:var(--fh);font-size:20px;font-weight:700;color:var(--dim);
  text-transform:uppercase;letter-spacing:2px;margin-bottom:6px}
.ph-s{font-family:var(--fm);font-size:11px;color:#152535;letter-spacing:1px}

/* EXPANDER */
[data-testid="stExpander"]{background:var(--bgc2) !important;
  border:1px solid var(--border) !important;border-radius:5px !important}
[data-testid="stExpander"] summary{font-family:var(--fm) !important;
  font-size:10px !important;color:var(--muted) !important;letter-spacing:2px !important}

/* FOOTER */
.foot{border-top:1px solid rgba(0,210,255,.07);padding:22px 0;
  display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:10px;margin-top:20px}
.flogo{font-family:var(--fh);font-size:22px;font-weight:900;
  color:rgba(0,210,255,.25);letter-spacing:2px}
.flogo em{font-style:normal;color:rgba(0,210,255,.5)}
.fmeta{font-family:var(--fm);font-size:9px;color:#0f2035;letter-spacing:1.5px}
</style>
""", unsafe_allow_html=True)

# ── MODEL LOAD ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

try:
    pipeline = load_pipeline()
    model_ok = True
except Exception as exc:
    model_ok  = False
    model_err = str(exc)

# ── TEAM & VENUE DATA ─────────────────────────────────────────────────────────
IPL_TEAMS = [
    "CSK — Chennai Super Kings",
    "MI — Mumbai Indians",
    "RCB — Royal Challengers Bengaluru",
    "KKR — Kolkata Knight Riders",
    "DC — Delhi Capitals",
    "PBKS — Punjab Kings",
    "RR — Rajasthan Royals",
    "SRH — Sunrisers Hyderabad",
    "GT — Gujarat Titans",
    "LSG — Lucknow Super Giants",
]
INTL_TEAMS = [
    "India","Australia","England","Pakistan",
    "South Africa","New Zealand","West Indies","Sri Lanka",
    "Bangladesh","Afghanistan","Zimbabwe","Ireland",
    "Scotland","Netherlands","Namibia","UAE",
    "Nepal","Oman","Papua New Guinea","USA",
    "Canada","Kenya","Uganda",
]
IPL_VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Rajiv Gandhi Int'l Stadium, Hyderabad",
    "Narendra Modi Stadium, Ahmedabad",
    "PCA Stadium, Mohali",
    "Sawai Mansingh Stadium, Jaipur",
    "Ekana Cricket Stadium, Lucknow",
    "YSRCA-VDCA Stadium, Visakhapatnam",
    "HPCA Stadium, Dharamsala",
    "MCA Stadium, Pune",
    "Brabourne Stadium, Mumbai",
]
INTL_VENUES = [
    "Melbourne Cricket Ground, Melbourne",
    "Sydney Cricket Ground, Sydney",
    "Perth Stadium, Perth",
    "Adelaide Oval, Adelaide",
    "Gabba, Brisbane",
    "Lord's Cricket Ground, London",
    "The Oval, London",
    "Headingley, Leeds",
    "Edgbaston, Birmingham",
    "Old Trafford, Manchester",
    "SuperSport Park, Centurion",
    "Newlands Stadium, Cape Town",
    "Wanderers Stadium, Johannesburg",
    "Kingsmead, Durban",
    "Gaddafi Stadium, Lahore",
    "National Stadium, Karachi",
    "Rawalpindi Cricket Stadium",
    "Dubai International Stadium",
    "Sheikh Zayed Stadium, Abu Dhabi",
    "Sharjah Cricket Stadium",
    "Eden Park, Auckland",
    "Hagley Oval, Christchurch",
    "Basin Reserve, Wellington",
    "Pallekele International Stadium",
    "R. Premadasa Stadium, Colombo",
    "Galle International Stadium",
    "Shere Bangla National Stadium, Dhaka",
    "Zahur Ahmed Chowdhury Stadium, Chittagong",
    "Kensington Oval, Barbados",
    "Sabina Park, Jamaica",
    "Queen's Park Oval, Trinidad",
    "National Cricket Stadium, Grenada",
]

def short(name: str) -> str:
    """Extract short name for model: 'CSK — Chennai...' → 'CSK', 'India' → 'India'"""
    return name.split("—")[0].strip() if "—" in name else name.split(",")[0].strip()

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hgrid"></div>
  <div class="hi">
    <div class="hchip">⚡ Real-Time Win Probability Engine</div>
    <h1 class="htitle">CRIC<em>AI</em></h1>
    <div class="hsub">
      <b>XGBoost + Platt Calibration</b> &nbsp;·&nbsp;
      Trained on <b>6,883 T20 Matches</b> &nbsp;·&nbsp;
      <b>387,033</b> Ball-by-Ball States
    </div>
    <div class="krow">
      <div class="kpi"><div class="kn">0.9179</div><div class="kl">ROC-AUC</div></div>
      <div class="kpi"><div class="kn">387K</div><div class="kl">Training Rows</div></div>
      <div class="kpi"><div class="kn">83%</div><div class="kl">Test Accuracy</div></div>
      <div class="kpi"><div class="kn">16</div><div class="kl">Features</div></div>
      <div class="kpi"><div class="kn">49s</div><div class="kl">Train Time</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TICKER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ticker">
  <div class="ti">
    <s>◆</s> IPL 2024 — CSK vs MI — Wankhede Stadium &nbsp;
    <s>◆</s> INT'L T20 — India vs Australia — MCG &nbsp;
    <s>◆</s> T20 WC — Pakistan vs England — Dubai Int'l Stadium &nbsp;
    <s>◆</s> IPL — RCB vs KKR — Chinnaswamy Stadium &nbsp;
    <s>◆</s> INT'L — New Zealand vs South Africa — Eden Park &nbsp;
    <s>◆</s> IPL — GT vs SRH — Narendra Modi Stadium &nbsp;
    <s>◆</s> CricAI · ML-Powered Live Win Probability · AUC 0.9179 &nbsp;
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

if not model_ok:
    st.error(f"⚠️ Model not found. Run `python src/pipeline/train_pipeline.py` first.\n\n{model_err}")
    st.stop()

# ── LAYOUT ─────────────────────────────────────────────────────────────────────
col_L, col_R = st.columns([1.05, 0.95], gap="large")

# ═══════════════════════ LEFT — INPUTS ════════════════════════════════════════
with col_L:
    st.markdown('<div class="slabel">Match Format</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">Teams & Venue</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    fmt    = st.radio("FORMAT", ["🏟️  IPL", "🌍  International"], horizontal=True)
    is_ipl = "IPL" in fmt
    t_pool = IPL_TEAMS  if is_ipl else INTL_TEAMS
    v_pool = IPL_VENUES if is_ipl else INTL_VENUES

    st.markdown('<div class="ndiv"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        bat_team = st.selectbox("🏏  BATTING TEAM  (chasing)", t_pool, index=0)
    with c2:
        bowl_opts = [t for t in t_pool if t != bat_team]
        bowl_team = st.selectbox("🎯  BOWLING TEAM  (defending)", bowl_opts, index=0)

    venue = st.selectbox("📍  VENUE", v_pool, index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="slabel">Live Scorecard</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">Current Match State</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        target = st.number_input("🎯  TARGET", min_value=50, max_value=350, value=180)
    with c4:
        score  = st.number_input("📈  CURRENT SCORE", min_value=0, max_value=349, value=95)

    c5, c6 = st.columns(2)
    with c5:
        overs = st.number_input("⏱️  OVERS (e.g. 10.3)", min_value=0.0,
                                 max_value=19.5, value=10.0, step=0.1, format="%.1f")
    with c6:
        wkts  = st.number_input("💀  WICKETS LOST", min_value=0, max_value=9, value=3)

    # Live derived stats
    lb   = int(overs) * 6 + min(round((overs % 1) * 10), 5)  # legal balls faced
    blft = max(1, 120 - lb)
    rreq = max(0, target - score)
    crr  = round((score * 6) / max(lb, 1), 2)
    rrr  = round((rreq * 6) / blft, 2)
    wih  = 10 - wkts
    rdel = round(crr - rrr, 2)

    rc = "var(--red)"   if rrr  > 12 else ("var(--amber)" if rrr  > 9  else "var(--green)")
    wc = "var(--red)"   if wih  <= 3  else ("var(--amber)" if wih  <= 5  else "var(--green)")
    dc = "var(--green)" if rdel >= 0  else "var(--red)"
    ds = f"+{rdel}" if rdel >= 0 else str(rdel)

    st.markdown(f"""
    <div class="pills">
      <div class="pill"><div class="pl">Runs Req</div><div class="pv">{rreq}</div></div>
      <div class="pill"><div class="pl">Balls Left</div><div class="pv">{blft}</div></div>
      <div class="pill"><div class="pl">CRR</div><div class="pv">{crr}</div></div>
      <div class="pill"><div class="pl">RRR</div><div class="pv" style="color:{rc}">{rrr}</div></div>
      <div class="pill"><div class="pl">Rate Gap</div><div class="pv" style="color:{dc}">{ds}</div></div>
      <div class="pill"><div class="pl">Wkts Left</div><div class="pv" style="color:{wc}">{wih}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  ANALYSE WIN PROBABILITY")


# ═══════════════════════ RIGHT — RESULTS ══════════════════════════════════════
with col_R:
    st.markdown('<div class="slabel">Analytics Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">Win Probability</div>', unsafe_allow_html=True)

    if predict_btn:
        if bat_team == bowl_team:
            st.error("Batting and bowling teams must be different.")
        else:
            with st.spinner("Running model…"):
                try:
                    res = pipeline.predict({
                        "batting_team":    short(bat_team),
                        "bowling_team":    short(bowl_team),
                        "venue":           venue.split(",")[0],
                        "match_type":      "T20",
                        "target":          target,
                        "current_score":   score,
                        "overs_completed": overs,
                        "wickets":         wkts,
                    })
                    bp  = float(res["batting_win"])
                    blp = float(res["bowling_win"])

                    bn   = short(bat_team)
                    bln  = short(bowl_team)
                    isb  = bp >= 50
                    win  = bn if isb else bln
                    conf = max(bp, blp)

                    bbar = ("linear-gradient(90deg,#00c96b,#00ff87)"
                            if isb else "linear-gradient(90deg,#cc2240,#ff3d5a)")
                    blbar= ("linear-gradient(90deg,#00c96b,#00ff87)"
                            if not isb else "linear-gradient(90deg,#cc2240,#ff3d5a)")
                    bcol = "var(--green)" if isb  else "var(--red)"
                    blcol= "var(--green)" if not isb else "var(--red)"

                    # ── PROBABILITY BARS ──────────────────────────────────────
                    st.markdown(f"""
                    <div class="pbox">
                      <div class="phead">◆ MATCH WIN PROBABILITY ANALYSIS</div>
                      <div class="prow">
                        <div class="pname">
                          <span>🏏 {bn}</span>
                          <span class="ppct" style="color:{bcol}">{bp:.1f}%</span>
                        </div>
                        <div class="ptrack">
                          <div class="pfill" style="width:{bp}%;background:{bbar}"></div>
                        </div>
                      </div>
                      <div class="prow">
                        <div class="pname">
                          <span>🎯 {bln}</span>
                          <span class="ppct" style="color:{blcol}">{blp:.1f}%</span>
                        </div>
                        <div class="ptrack">
                          <div class="pfill" style="width:{blp}%;background:{blbar}"></div>
                        </div>
                      </div>
                      <div class="wbanner">
                        <div class="wlabel">◆ Model Prediction</div>
                        <div class="wname">{win} FAVOURED</div>
                        <div class="wconf">CONFIDENCE: {conf:.1f}%</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── GAUGE ─────────────────────────────────────────────────
                    fg = go.Figure(go.Indicator(
                        mode="gauge+number", value=bp,
                        number={"suffix":"%","font":{"size":44,"color":"#00d2ff",
                                                      "family":"Barlow Condensed"}},
                        title={"text":f"{bn} Win %",
                               "font":{"size":12,"color":"#4a6a88",
                                       "family":"Share Tech Mono"}},
                        gauge={
                            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#0d1f35",
                                    "tickfont":{"color":"#1a3050","size":9}},
                            "bar":{"color":"#00d2ff","thickness":0.22},
                            "bgcolor":"#060b14","borderwidth":0,
                            "steps":[
                                {"range":[0,33],  "color":"rgba(255,61,90,.12)"},
                                {"range":[33,50], "color":"rgba(255,184,0,.08)"},
                                {"range":[50,67], "color":"rgba(0,210,255,.08)"},
                                {"range":[67,100],"color":"rgba(0,255,135,.12)"},
                            ],
                            "threshold":{"line":{"color":"rgba(255,184,0,.5)","width":2},
                                         "thickness":0.8,"value":50},
                        },
                    ))
                    fg.update_layout(paper_bgcolor="#0d1520",plot_bgcolor="#0d1520",
                                     height=210,margin=dict(t=36,b=0,l=16,r=16))
                    st.plotly_chart(fg,use_container_width=True,
                                   config={"displayModeBar":False})

                    # ── PROBABILITY CURVE ─────────────────────────────────────
                    st.markdown('<div class="slabel" style="margin-top:12px">Trajectory</div>',
                                unsafe_allow_html=True)
                    st.markdown('<div class="stitle">Win Probability Curve</div>',
                                unsafe_allow_html=True)

                    xs, ys = [], []
                    for b in range(6, lb + 1, 3):
                        ov_pt = b / 6
                        sc_pt = int(score * b / max(lb, 1))
                        try:
                            p = pipeline.predict({
                                "batting_team":    short(bat_team),
                                "bowling_team":    short(bowl_team),
                                "venue":           venue.split(",")[0],
                                "match_type":      "T20",
                                "target":          target,
                                "current_score":   max(0, sc_pt),
                                "overs_completed": round(ov_pt, 1),
                                "wickets":         wkts,
                            })
                            xs.append(round(ov_pt, 1))
                            ys.append(p["batting_win"])
                        except Exception:
                            pass

                    if len(xs) >= 2:
                        fl = go.Figure()
                        fl.add_trace(go.Scatter(
                            x=xs, y=ys, mode="lines",
                            fill="tozeroy",fillcolor="rgba(0,210,255,.06)",
                            line=dict(color="#00d2ff",width=2.5),
                            hovertemplate="Over %{x}<br><b>%{y:.1f}%</b><extra></extra>",
                        ))
                        fl.add_hline(y=50,line_dash="dot",
                                     line_color="rgba(255,184,0,.35)",line_width=1,
                                     annotation_text="50%",
                                     annotation_font={"color":"#ffb800","size":9},
                                     annotation_position="right")
                        fl.add_trace(go.Scatter(
                            x=[xs[-1]],y=[ys[-1]],mode="markers",
                            marker=dict(color="#00ff87",size=11,
                                        line=dict(color="#060b14",width=2)),
                            hovertemplate="Now: <b>%{y:.1f}%</b><extra></extra>",
                        ))
                        fl.update_layout(
                            paper_bgcolor="#0d1520",plot_bgcolor="#0d1520",
                            height=230,showlegend=False,
                            margin=dict(t=10,b=36,l=44,r=16),
                            xaxis=dict(title="Over",color="#1a3050",gridcolor="#0b1d30",
                                       tickfont={"color":"#1a3050","size":9},
                                       title_font={"color":"#4a6a88","size":10}),
                            yaxis=dict(title="Win %",range=[0,100],color="#1a3050",
                                       gridcolor="#0b1d30",tickformat=".0f",ticksuffix="%",
                                       tickfont={"color":"#1a3050","size":9},
                                       title_font={"color":"#4a6a88","size":10}),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fl,use_container_width=True,
                                       config={"displayModeBar":False})

                    # ── DEBUG ─────────────────────────────────────────────────
                    with st.expander("🔬  RAW MODEL FEATURES  (debug)"):
                        st.json({
                            "batting_team":     short(bat_team),
                            "bowling_team":     short(bowl_team),
                            "venue":            venue.split(",")[0],
                            "balls_bowled":     lb,
                            "balls_left":       blft,
                            "runs_required":    rreq,
                            "current_run_rate": crr,
                            "required_run_rate":rrr,
                            "rr_delta":         rdel,
                            "wickets_in_hand":  wih,
                            "resource_index":   round((blft/120)*(wih/10),3),
                            "match_progress":   round(lb/120,3),
                        })

                except Exception as exc:
                    st.error(f"Prediction error: {exc}")

    else:
        st.markdown("""
        <div class="placeholder">
          <div class="ph-icon">🏏</div>
          <div class="ph-t">Awaiting Match Data</div>
          <div class="ph-s">
            Enter match state on the left<br>and click ANALYSE WIN PROBABILITY
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="foot">
  <div class="flogo">CRIC<em>AI</em></div>
  <div class="fmeta">
    XGBOOST · PLATT CALIBRATION · 6,883 T20 MATCHES · 387K ROWS · AUC 0.9179
  </div>
  <div class="fmeta">FOR ANALYSIS PURPOSES ONLY</div>
</div>
""", unsafe_allow_html=True)
