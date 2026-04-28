import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as components

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Intelligence Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── MASTER CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Jost:wght@200;300;400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════
   ROOT TOKENS
═══════════════════════════════════════════ */
:root {
  --ink:       #03070f;
  --ink2:      #060d1c;
  --ink3:      #091426;
  --plate:     rgba(6,13,28,0.85);
  --gold:      #d4a843;
  --gold2:     #f0c96a;
  --gold3:     #8a6a1f;
  --ember:     #ff4d5a;
  --jade:      #00e5a0;
  --ice:       #5ce0ff;
  --fg:        #b8c8e8;
  --fg2:       #6a7fa8;
  --fg3:       #3a4f72;
  --border:    rgba(212,168,67,0.12);
  --border2:   rgba(212,168,67,0.25);
}

/* ═══════════════════════════════════════════
   GLOBAL
═══════════════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--ink) !important;
  color: var(--fg) !important;
  font-family: 'Jost', sans-serif !important;
}
[data-testid="stSidebar"] {
  background: var(--ink2) !important;
  border-right: 1px solid var(--border2) !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
.stDeployButton { display: none !important; }
[data-testid="stForm"] { background: transparent !important; border: none !important; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1320px !important; }

/* ═══════════════════════════════════════════
   ANIMATED CIRCUIT GRID BACKGROUND
═══════════════════════════════════════════ */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(212,168,67,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(212,168,67,.04) 1px, transparent 1px),
    radial-gradient(ellipse 70% 50% at 15% 15%, rgba(92,224,255,.05) 0%, transparent 55%),
    radial-gradient(ellipse 50% 60% at 85% 85%, rgba(212,168,67,.07) 0%, transparent 50%),
    radial-gradient(ellipse 60% 40% at 50% 50%, rgba(0,229,160,.03) 0%, transparent 60%);
  background-size: 48px 48px, 48px 48px, 100% 100%, 100% 100%, 100% 100%;
  pointer-events: none;
  z-index: 0;
  animation: gridShift 20s linear infinite;
}
@keyframes gridShift {
  0%   { background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }
  100% { background-position: 48px 48px, 48px 48px, 0 0, 0 0, 0 0; }
}

/* ═══════════════════════════════════════════
   SCANLINE OVERLAY
═══════════════════════════════════════════ */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,.06) 2px, rgba(0,0,0,.06) 4px
  );
  pointer-events: none;
  z-index: 1;
}

/* ═══════════════════════════════════════════
   MASTHEAD
═══════════════════════════════════════════ */
.masthead-wrap {
  position: relative;
  padding: 3rem 0 2rem;
  text-align: center;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.5rem;
  overflow: hidden;
}
.masthead-wrap::before {
  content: 'CHURN INTELLIGENCE';
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  font-family: 'Bebas Neue', sans-serif;
  font-size: 14rem;
  color: rgba(212,168,67,.025);
  white-space: nowrap;
  letter-spacing: .2em;
  pointer-events: none;
  z-index: 0;
}
.masthead-tag {
  font-family: 'Share Tech Mono', monospace;
  font-size: .65rem;
  letter-spacing: .45em;
  color: var(--gold);
  text-transform: uppercase;
  display: flex; align-items: center; justify-content: center; gap: .75rem;
  margin-bottom: 1rem;
}
.masthead-tag::before, .masthead-tag::after {
  content: ''; width: 40px; height: 1px;
}
.masthead-tag::before { background: linear-gradient(90deg, transparent, var(--gold)); }
.masthead-tag::after  { background: linear-gradient(90deg, var(--gold), transparent); }
.masthead-h1 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 5.5rem; line-height: .95;
  color: #fff; letter-spacing: .06em; margin: 0;
  position: relative; z-index: 1;
}
.masthead-h1 em {
  font-style: normal;
  -webkit-text-stroke: 2px var(--gold);
  color: transparent;
}
.masthead-sub {
  font-size: .8rem; color: var(--fg2);
  letter-spacing: .18em; text-transform: uppercase;
  margin-top: .8rem; font-weight: 300;
}
.live-dot {
  display: inline-block; width: 6px; height: 6px;
  border-radius: 50%; background: var(--jade);
  margin-right: 6px; box-shadow: 0 0 8px var(--jade);
  animation: blink 1.4s ease-in-out infinite;
  vertical-align: middle;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:.2; } }

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
.sb-logo {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem; letter-spacing: .12em; color: #fff;
  padding: 1.2rem 1rem .2rem;
  border-bottom: 1px solid var(--border2);
  margin-bottom: 1.5rem;
}
.sb-logo span { color: var(--gold); }
.sb-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: .55rem; letter-spacing: .35em; color: var(--fg2); margin-top: 0;
}
.sb-section {
  font-family: 'Share Tech Mono', monospace;
  font-size: .58rem; letter-spacing: .3em; color: var(--gold);
  text-transform: uppercase;
  border-top: 1px solid var(--border);
  padding-top: .9rem; margin: 1.2rem 0 .75rem;
}
.kv {
  display: flex; justify-content: space-between; align-items: center;
  padding: .55rem .85rem; margin-bottom: .35rem;
  border: 1px solid var(--border); border-radius: 6px;
  background: rgba(212,168,67,.04);
  font-size: .82rem; position: relative; overflow: hidden;
}
.kv::before {
  content: ''; position: absolute;
  left: 0; top: 0; bottom: 0; width: 2px;
  background: var(--gold); opacity: .5;
}
.kv-k { color: var(--fg2); }
.kv-v { font-family: 'Share Tech Mono', monospace; color: var(--gold2); font-size: .88rem; }
.ftag {
  display: inline-block;
  background: rgba(92,224,255,.05); border: 1px solid rgba(92,224,255,.2);
  color: var(--ice); font-family: 'Share Tech Mono', monospace;
  font-size: .6rem; padding: .2rem .55rem; border-radius: 3px;
  margin: .1rem .05rem; letter-spacing: .06em;
}
.sb-note {
  margin-top: 1rem; font-size: .7rem; color: var(--fg2);
  line-height: 1.75; padding: .75rem;
  border: 1px solid var(--border); border-radius: 6px;
  background: rgba(255,255,255,.015); font-weight: 300;
}

/* ═══════════════════════════════════════════
   SECTION LABELS
═══════════════════════════════════════════ */
.sec-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: .6rem; letter-spacing: .35em; color: var(--gold);
  text-transform: uppercase;
  display: flex; align-items: center; gap: .6rem; margin-bottom: 1.2rem;
}
.sec-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border2), transparent);
}

/* ═══════════════════════════════════════════
   FORM CONTROLS
═══════════════════════════════════════════ */
label, .stSelectbox label, .stSlider label {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: .64rem !important; letter-spacing: .25em !important;
  text-transform: uppercase !important; color: var(--fg2) !important;
  font-weight: 400 !important;
}
[data-baseweb="select"] {
  background: rgba(9,20,38,.9) !important;
  border: 1px solid var(--border2) !important; border-radius: 6px !important;
}
[data-baseweb="select"] div, [data-baseweb="select"] span {
  background: transparent !important; color: var(--fg) !important;
  font-family: 'Jost', sans-serif !important; font-size: .9rem !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
  background: rgba(212,168,67,.12) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--gold) !important; border-color: var(--gold2) !important;
  box-shadow: 0 0 12px rgba(212,168,67,.5) !important;
}
[data-testid="baseButton-primary"] {
  background: transparent !important;
  border: 1px solid var(--gold) !important; color: var(--gold) !important;
  border-radius: 6px !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: .78rem !important; letter-spacing: .25em !important;
  text-transform: uppercase !important; padding: .85rem !important;
  transition: all .3s ease !important;
}
[data-testid="baseButton-primary"]:hover {
  color: var(--gold2) !important; border-color: var(--gold2) !important;
  box-shadow: 0 0 24px rgba(212,168,67,.3) !important;
}

/* ═══════════════════════════════════════════
   RESULT CARDS
═══════════════════════════════════════════ */
.result-wrap {
  padding: 1.8rem; border-radius: 12px;
  text-align: center; margin-bottom: 1rem;
  position: relative; overflow: hidden;
}
.result-churn {
  background: linear-gradient(145deg, rgba(255,77,90,.1), rgba(255,77,90,.04));
  border: 1px solid rgba(255,77,90,.35);
}
.result-safe {
  background: linear-gradient(145deg, rgba(0,229,160,.1), rgba(0,229,160,.04));
  border: 1px solid rgba(0,229,160,.35);
}
.result-corner {
  position: absolute; width: 16px; height: 16px;
  border-color: currentColor; border-style: solid; opacity: .5;
}
.c-tl { top:8px; left:8px;   border-width: 1px 0 0 1px; }
.c-tr { top:8px; right:8px;  border-width: 1px 1px 0 0; }
.c-bl { bottom:8px; left:8px;  border-width: 0 0 1px 1px; }
.c-br { bottom:8px; right:8px; border-width: 0 1px 1px 0; }
.result-churn .result-corner { color: var(--ember); }
.result-safe  .result-corner { color: var(--jade); }
.result-eyebrow {
  font-family: 'Share Tech Mono', monospace;
  font-size: .6rem; letter-spacing: .45em; text-transform: uppercase; margin-bottom: .5rem;
}
.result-churn .result-eyebrow { color: var(--ember); }
.result-safe  .result-eyebrow { color: var(--jade); }
.result-verdict {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3.5rem; letter-spacing: .08em; line-height: 1; margin-bottom: .3rem;
}
.result-churn .result-verdict { color: #ff7a85; }
.result-safe  .result-verdict { color: #3dffc4; }
.result-pct {
  font-family: 'Share Tech Mono', monospace;
  font-size: .85rem; color: var(--fg2);
}
.result-pct b { font-size: 1.5rem; }
.result-churn .result-pct b { color: var(--ember); }
.result-safe  .result-pct b  { color: var(--jade); }
.badge {
  display: inline-block; padding: .25rem .85rem; border-radius: 3px;
  font-family: 'Share Tech Mono', monospace;
  font-size: .65rem; letter-spacing: .15em; text-transform: uppercase; margin-top: .8rem;
}
.badge-high { background:rgba(255,77,90,.12);  color:#ff7a85; border:1px solid rgba(255,77,90,.3); }
.badge-med  { background:rgba(255,166,0,.12);  color:#ffc95a; border:1px solid rgba(255,166,0,.3); }
.badge-low  { background:rgba(0,229,160,.12);  color:#3dffc4; border:1px solid rgba(0,229,160,.3); }

/* ═══════════════════════════════════════════
   BAR GAUGES
═══════════════════════════════════════════ */
.bar-row { display:flex; align-items:center; gap:1rem; margin-bottom:.85rem; }
.bar-lbl {
  font-family:'Share Tech Mono',monospace; font-size:.6rem;
  letter-spacing:.12em; text-transform:uppercase; color:var(--fg2);
  width:72px; flex-shrink:0;
}
.bar-track {
  flex:1; height:4px; background:rgba(255,255,255,.05);
  border-radius:99px; position:relative; overflow:visible;
}
.bar-fill { height:100%; border-radius:99px; position:relative; }
.bar-fill::after {
  content:''; position:absolute; right:-1px; top:50%;
  transform:translateY(-50%); width:8px; height:8px;
  border-radius:50%; box-shadow:0 0 10px currentColor; background:currentColor;
}
.bar-churn .bar-fill {
  background:linear-gradient(90deg,rgba(255,77,90,.4),var(--ember)); color:var(--ember);
}
.bar-safe .bar-fill {
  background:linear-gradient(90deg,rgba(0,229,160,.4),var(--jade)); color:var(--jade);
}
.bar-num { font-family:'Share Tech Mono',monospace; font-size:.82rem; width:44px; text-align:right; flex-shrink:0; }
.bar-churn .bar-num { color:var(--ember); }
.bar-safe  .bar-num { color:var(--jade); }

/* ═══════════════════════════════════════════
   SUMMARY TABLE
═══════════════════════════════════════════ */
.srow {
  display:flex; justify-content:space-between; align-items:center;
  padding:.5rem 0; border-bottom:1px solid rgba(255,255,255,.03); font-size:.84rem;
}
.srow:last-child { border-bottom:none; }
.srow-k { color:var(--fg2); font-weight:300; }
.srow-v {
  font-family:'Share Tech Mono',monospace; font-size:.78rem; color:var(--fg);
  background:rgba(212,168,67,.06); border:1px solid var(--border);
  padding:.12rem .6rem; border-radius:3px;
}

/* ═══════════════════════════════════════════
   INFO BOX
═══════════════════════════════════════════ */
.infobox {
  background:rgba(92,224,255,.04); border:1px solid rgba(92,224,255,.12);
  border-left:2px solid var(--ice); border-radius:6px;
  padding:.9rem 1.1rem; font-size:.82rem; color:var(--fg2);
  line-height:1.65; margin-bottom:1rem; font-weight:300;
}
.infobox strong { color:var(--ice); font-weight:500; }

/* ═══════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════ */
.ftr {
  display:flex; justify-content:space-between; align-items:center;
  padding:1rem 0 .5rem; border-top:1px solid var(--border);
  font-family:'Share Tech Mono',monospace;
  font-size:.58rem; letter-spacing:.18em; color:var(--fg3);
  text-transform:uppercase; margin-top:2rem;
}

/* ═══════════════════════════════════════════
   ANIMATIONS
═══════════════════════════════════════════ */
@keyframes fadeUp {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:translateY(0); }
}
.fade-up { animation: fadeUp .5s ease both; }
.d1 { animation-delay:.05s; } .d2 { animation-delay:.12s; }
.d3 { animation-delay:.2s;  } .d4 { animation-delay:.28s; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    pkg           = load_model()
    model         = pkg['model']
    le_dict       = pkg['label_encoders']
    feature_names = pkg['feature_names']
    accuracy      = pkg['accuracy']
    roc_auc       = pkg['roc_auc']
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">CHURN<span>IQ</span>
        <div class="sb-sub">INTELLIGENCE ENGINE · v3.0</div>
    </div>
    <div class="sb-section">◈ Model Vitals</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kv"><span class="kv-k">Algorithm</span><span class="kv-v">Random Forest</span></div>
    <div class="kv"><span class="kv-k">Estimators</span><span class="kv-v">200 trees</span></div>
    <div class="kv"><span class="kv-k">Max Depth</span><span class="kv-v">10 levels</span></div>
    <div class="kv"><span class="kv-k">Accuracy</span><span class="kv-v">{accuracy*100:.1f}%</span></div>
    <div class="kv"><span class="kv-k">ROC-AUC</span><span class="kv-v">{roc_auc:.4f}</span></div>
    <div class="kv"><span class="kv-k">Training Set</span><span class="kv-v">954 records</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">◈ Feature Importance</div>', unsafe_allow_html=True)
    fi_tags = [("Age·28.4%"), ("Services·22.9%"), ("Flyer·17.5%"),
               ("Income·15.7%"), ("Social·10.0%"), ("Hotel·5.4%")]
    st.markdown("".join(f'<span class="ftag">{t}</span>' for t in fi_tags), unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-note">
        🎓 <strong style="color:var(--fg)">B.Tech · Gen AI</strong><br>
        2nd Semester · Final Project<br><br>
        Ensemble decision trees on 954<br>
        airline customer records.<br>
        Base churn rate: <strong style="color:var(--gold2)">23.5%</strong>
    </div>""", unsafe_allow_html=True)


# ── MASTHEAD ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead-wrap">
    <div class="masthead-tag">◈ Airline Retention Analytics ◈</div>
    <h1 class="masthead-h1">PREDICT <em>CHURN</em></h1>
    <p class="masthead-sub">
        <span class="live-dot"></span>
        Random Forest · 200 Estimators · Real-time Inference
    </p>
</div>
""", unsafe_allow_html=True)


# ── TWO COLUMN LAYOUT ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════
#  LEFT — INPUT + FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="sec-label">◈ Customer Profile Input</div>', unsafe_allow_html=True)

    with st.form("churn_form"):
        age = st.slider("Age", min_value=18, max_value=80, value=32, step=1,
                        help="Training data range: 27–38")

        c1, c2 = st.columns(2)
        with c1:
            frequent_flyer = st.selectbox("Frequent Flyer", options=["No", "Yes", "No Record"])
        with c2:
            annual_income  = st.selectbox("Income Class",   options=["Low Income", "Middle Income", "High Income"])

        services_opted = st.slider("Services Opted", min_value=1, max_value=10, value=3, step=1,
                                   help="Training data range: 1–6")

        c3, c4 = st.columns(2)
        with c3:
            social_media  = st.selectbox("Social Media Sync", options=["No", "Yes"])
        with c4:
            hotel_booked  = st.selectbox("Hotel Booked",      options=["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "◈  EXECUTE CHURN ANALYSIS", type="primary", use_container_width=True
        )

    # Feature importance bars (static)
    st.markdown('<div class="sec-label" style="margin-top:1.5rem">◈ Feature Importance</div>', unsafe_allow_html=True)
    fi_items = [
        ("Age",           28.4, "var(--gold)"),
        ("Services",      22.9, "var(--gold)"),
        ("Frequent Flyer",17.5, "var(--ice)"),
        ("Income Class",  15.7, "var(--ice)"),
        ("Social Media",  10.0, "var(--fg2)"),
        ("Hotel Booked",   5.4, "var(--fg2)"),
    ]
    fi_html = ""
    for name, pct, color in fi_items:
        fi_html += f"""
        <div class="bar-row">
          <div class="bar-lbl">{name}</div>
          <div class="bar-track">
            <div style="width:{pct*3:.0f}%;height:4px;border-radius:99px;
                        background:{color};opacity:.8;box-shadow:0 0 6px {color}55"></div>
          </div>
          <div class="bar-num" style="color:{color}">{pct}%</div>
        </div>"""
    st.markdown(fi_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  RIGHT — RESULTS
# ══════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="sec-label">◈ Prediction Output</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div class="infobox fade-up d1">
            <strong>System Ready</strong><br>
            Configure the customer profile on the left and execute analysis.
            The Random Forest ensemble returns a churn probability and
            risk classification in real-time.
        </div>""", unsafe_allow_html=True)

        # Idle gauge placeholder
        components.html("""
        <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&display=swap" rel="stylesheet">
        <style>
          body{margin:0;background:transparent;}
          @keyframes idlePulse{0%,100%{opacity:.4}50%{opacity:.8}}
        </style>
        <svg width="100%" viewBox="0 0 260 145" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="g"><feGaussianBlur stdDeviation="2.5" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          <path d="M 25 120 A 105 105 0 0 1 235 120"
                fill="none" stroke="rgba(212,168,67,.08)" stroke-width="14" stroke-linecap="round"/>
          <g stroke="rgba(212,168,67,.18)" stroke-width="1" animation="idlePulse 2s infinite">
            <line x1="130" y1="17" x2="130" y2="30"/>
            <line x1="45"  y1="58" x2="55"  y2="67"/>
            <line x1="215" y1="58" x2="205" y2="67"/>
            <line x1="25"  y1="120" x2="38"  y2="120"/>
            <line x1="235" y1="120" x2="222" y2="120"/>
          </g>
          <text x="130" y="110" text-anchor="middle" font-size="11"
                fill="rgba(212,168,67,.25)" font-family="Share Tech Mono,monospace"
                letter-spacing="4" style="animation:idlePulse 2s infinite">STANDBY</text>
          <text x="25"  y="134" text-anchor="middle" font-size="9" fill="rgba(255,255,255,.15)" font-family="Share Tech Mono,monospace">0</text>
          <text x="235" y="134" text-anchor="middle" font-size="9" fill="rgba(255,255,255,.15)" font-family="Share Tech Mono,monospace">100</text>
          <text x="130" y="34"  text-anchor="middle" font-size="9" fill="rgba(255,255,255,.15)" font-family="Share Tech Mono,monospace">50</text>
        </svg>
        """, height=155)

        st.markdown('<div class="sec-label" style="margin-top:.5rem">◈ Sample Scenarios</div>', unsafe_allow_html=True)
        sample = pd.DataFrame({
            'Scenario': ['◉ Low Risk', '◎ Med Risk', '● High Risk'],
            'Age':      [35, 30, 28],
            'Flyer':    ['No', 'Yes', 'Yes'],
            'Income':   ['High', 'Low', 'Low'],
            'Services': [2, 4, 1],
            'Social':   ['No', 'Yes', 'No'],
            'Hotel':    ['Yes', 'No', 'No'],
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)

    else:
        # ── Encode & Predict ──────────────────────────────
        input_df = pd.DataFrame({
            'Age':                        [age],
            'FrequentFlyer':              [frequent_flyer],
            'AnnualIncomeClass':          [annual_income],
            'ServicesOpted':              [services_opted],
            'AccountSyncedToSocialMedia': [social_media],
            'BookedHotelOrNot':           [hotel_booked]
        })
        enc = input_df.copy()
        for col in ['FrequentFlyer', 'AnnualIncomeClass',
                    'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
            enc[col] = le_dict[col].transform(enc[col])
        enc = enc[feature_names]

        pred    = model.predict(enc)[0]
        proba   = model.predict_proba(enc)[0]
        p_churn = float(proba[1]) * 100
        p_safe  = float(proba[0]) * 100

        # Risk tier
        if p_churn >= 65:
            badge_html = '<span class="badge badge-high">◈ High Risk</span>'
        elif p_churn >= 35:
            badge_html = '<span class="badge badge-med">◈ Medium Risk</span>'
        else:
            badge_html  = '<span class="badge badge-low">◈ Low Risk</span>'

        # ── Animated SVG Arc Gauge ────────────────────────
        angle = p_churn / 100 * 180
        rad   = np.radians(180 - angle)
        cx, cy, R = 130, 120, 105
        rn = 82
        nx = cx + rn * float(np.cos(rad))
        ny = cy - rn * float(np.sin(rad))

        arc_color    = "#ff4d5a" if pred == 1 else "#00e5a0"
        needle_color = "#ff7a85" if pred == 1 else "#3dffc4"
        # stroke-dashoffset for animated arc (circumference of semi-circle ≈ π*R = 330)
        circ  = np.pi * R
        remain = circ * (1 - p_churn / 100)

        gauge_html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Share+Tech+Mono&display=swap" rel="stylesheet">
        <style>
          body{{margin:0;background:transparent;}}
          @keyframes arcIn{{
            from{{stroke-dashoffset:{circ:.1f}}}
            to{{stroke-dashoffset:{remain:.1f}}}
          }}
          @keyframes fadeIn{{from{{opacity:0;transform:scale(.85)}}to{{opacity:1;transform:scale(1)}}}}
          .arc-animated{{
            stroke-dasharray:{circ:.1f};
            stroke-dashoffset:{remain:.1f};
            animation:arcIn 1.3s cubic-bezier(.4,0,.2,1) both;
          }}
          .num-anim{{animation:fadeIn .5s .9s ease both;opacity:0;}}
        </style>
        <svg width="100%" viewBox="0 0 260 145" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <linearGradient id="ag" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="{arc_color}" stop-opacity=".4"/>
              <stop offset="100%" stop-color="{arc_color}" stop-opacity="1"/>
            </linearGradient>
          </defs>

          <!-- Track -->
          <path d="M 25 120 A 105 105 0 0 1 235 120"
                fill="none" stroke="rgba(255,255,255,.05)"
                stroke-width="14" stroke-linecap="round"/>

          <!-- Animated fill -->
          <path class="arc-animated"
                d="M 25 120 A 105 105 0 0 1 235 120"
                fill="none" stroke="url(#ag)" stroke-width="14"
                stroke-linecap="round" filter="url(#glow)"/>

          <!-- Tick marks -->
          <g stroke="rgba(255,255,255,.15)" stroke-width="1">
            <line x1="130" y1="17"  x2="130" y2="30"/>
            <line x1="45"  y1="58"  x2="55"  y2="67"/>
            <line x1="215" y1="58"  x2="205" y2="67"/>
            <line x1="25"  y1="120" x2="38"  y2="120"/>
            <line x1="235" y1="120" x2="222" y2="120"/>
          </g>
          <text x="25"  y="135" text-anchor="middle" font-size="9" fill="rgba(255,255,255,.2)" font-family="Share Tech Mono,monospace">0</text>
          <text x="235" y="135" text-anchor="middle" font-size="9" fill="rgba(255,255,255,.2)" font-family="Share Tech Mono,monospace">100</text>
          <text x="130" y="35"  text-anchor="middle" font-size="9" fill="rgba(255,255,255,.2)" font-family="Share Tech Mono,monospace">50</text>

          <!-- Needle -->
          <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
                stroke="{needle_color}" stroke-width="2.5" stroke-linecap="round"
                filter="url(#glow)" opacity=".95"/>
          <circle cx="{cx}" cy="{cy}" r="6"
                  fill="{arc_color}" filter="url(#glow)"/>
          <circle cx="{cx}" cy="{cy}" r="3" fill="#fff"/>

          <!-- Central readout -->
          <g class="num-anim">
            <text x="130" y="105" text-anchor="middle"
                  font-size="32" font-family="Bebas Neue,sans-serif"
                  fill="{arc_color}" filter="url(#glow)">{p_churn:.1f}%</text>
            <text x="130" y="120" text-anchor="middle"
                  font-size="8" font-family="Share Tech Mono,monospace"
                  fill="rgba(255,255,255,.3)" letter-spacing="3">CHURN PROBABILITY</text>
          </g>
        </svg>
        """
        components.html(gauge_html, height=160)

        # ── Verdict Card ──────────────────────────────────
        if pred == 1:
            st.markdown(f"""
            <div class="result-wrap result-churn fade-up d1">
              <div class="result-corner c-tl"></div><div class="result-corner c-tr"></div>
              <div class="result-corner c-bl"></div><div class="result-corner c-br"></div>
              <div class="result-eyebrow">⚠ Prediction Result</div>
              <div class="result-verdict">CHURN LIKELY</div>
              <div class="result-pct"><b>{p_churn:.1f}%</b> churn confidence</div>
              {badge_html}
            </div>""", unsafe_allow_html=True)
            rec = "Recommend targeted retention offer, loyalty tier upgrade, or priority outreach within 72 hours."
        else:
            st.markdown(f"""
            <div class="result-wrap result-safe fade-up d1">
              <div class="result-corner c-tl"></div><div class="result-corner c-tr"></div>
              <div class="result-corner c-bl"></div><div class="result-corner c-br"></div>
              <div class="result-eyebrow">✓ Prediction Result</div>
              <div class="result-verdict">RETAINED</div>
              <div class="result-pct"><b>{p_safe:.1f}%</b> retention confidence</div>
              {badge_html}
            </div>""", unsafe_allow_html=True)
            rec = "Customer profile is stable. Maintain engagement cadence and monitor service usage trends quarterly."

        # ── Confidence Bars ───────────────────────────────
        st.markdown('<div class="sec-label" style="margin-top:1rem">◈ Confidence Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="bar-row bar-safe fade-up d2">
          <div class="bar-lbl">Retained</div>
          <div class="bar-track"><div class="bar-fill" style="width:{p_safe:.1f}%"></div></div>
          <div class="bar-num">{p_safe:.1f}%</div>
        </div>
        <div class="bar-row bar-churn fade-up d3">
          <div class="bar-lbl">Churn</div>
          <div class="bar-track"><div class="bar-fill" style="width:{p_churn:.1f}%"></div></div>
          <div class="bar-num">{p_churn:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # ── Profile Summary ───────────────────────────────
        st.markdown('<div class="sec-label" style="margin-top:1rem">◈ Profile Summary</div>', unsafe_allow_html=True)
        fields = {
            "Age":             str(age),
            "Frequent Flyer":  frequent_flyer,
            "Income Class":    annual_income,
            "Services Opted":  str(services_opted),
            "Social Media":    social_media,
            "Hotel Booked":    hotel_booked,
        }
        rows_html = "".join(
            f'<div class="srow"><span class="srow-k">{k}</span>'
            f'<span class="srow-v">{v}</span></div>'
            for k, v in fields.items()
        )
        st.markdown(f'<div class="fade-up d4">{rows_html}</div>', unsafe_allow_html=True)

        # ── Action Recommendation ─────────────────────────
        st.markdown(f"""
        <div class="infobox fade-up d4" style="margin-top:1rem">
            <strong>Action Recommendation</strong><br>{rec}
        </div>""", unsafe_allow_html=True)


# ── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
    <span>🎓 B.Tech · Gen AI · 2nd Semester</span>
    <span>Random Forest · 200 Trees</span>
    <span>Built with Streamlit</span>
</div>
""", unsafe_allow_html=True)
