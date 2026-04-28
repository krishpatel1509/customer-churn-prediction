import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Prediction Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap" rel="stylesheet">

<style>
/* ── Root Variables ── */
:root {
    --navy:      #080f1e;
    --navy-2:    #0d1a30;
    --navy-3:    #112240;
    --gold:      #c9a84c;
    --gold-lt:   #e8c97a;
    --gold-dk:   #a07830;
    --cyan:      #64ffda;
    --red:       #ff5f6d;
    --green:     #43e97b;
    --text:      #ccd6f6;
    --text-dim:  #8892b0;
    --glass:     rgba(17, 34, 64, 0.72);
    --border:    rgba(201, 168, 76, 0.18);
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--navy-2) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Animated starfield background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(100,255,218,.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 80% 90%, rgba(201,168,76,.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(13,26,48,.9) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}

/* ── Section spacing ── */
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1280px !important;
}

/* ── Masthead ── */
.masthead {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
.masthead-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.35em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.masthead-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 600;
    line-height: 1;
    color: #fff;
    margin: 0;
    letter-spacing: -0.01em;
}
.masthead-title span {
    color: var(--gold);
}
.masthead-sub {
    font-size: 0.92rem;
    color: var(--text-dim);
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.masthead-rule {
    width: 80px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.5rem auto 0;
}

/* ── Glass Cards ── */
.glass-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
    margin-bottom: 1.25rem;
}
.glass-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.glass-card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--gold-lt) !important;
}

.sidebar-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
    padding: 1rem 1rem 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-logo span { color: var(--gold); }

.metric-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 1rem;
    background: rgba(201,168,76,.08);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 0.84rem;
}
.metric-pill-label { color: var(--text-dim); }
.metric-pill-value {
    font-family: 'DM Mono', monospace;
    color: var(--gold-lt);
    font-size: 0.9rem;
}

.feature-tag {
    display: inline-block;
    background: rgba(100,255,218,.06);
    border: 1px solid rgba(100,255,218,.18);
    color: var(--cyan);
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.22rem 0.6rem;
    border-radius: 20px;
    margin: 0.15rem 0.1rem;
    letter-spacing: 0.04em;
}

/* ── Form Controls ── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--gold-dk), var(--gold)) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background: rgba(201,168,76,0.15) !important;
}
[data-baseweb="select"] {
    background: rgba(13,26,48,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] div {
    background: transparent !important;
    color: var(--text) !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--gold-dk) 0%, var(--gold) 50%, var(--gold-lt) 100%) !important;
    color: var(--navy) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    padding: 0.75rem !important;
    box-shadow: 0 4px 20px rgba(201,168,76,.3) !important;
    transition: all .25s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 6px 28px rgba(201,168,76,.5) !important;
    transform: translateY(-1px) !important;
}
label, .stSelectbox label, .stSlider label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Prediction Result Cards ── */
.result-churn {
    background: linear-gradient(135deg, rgba(255,95,109,.12), rgba(255,95,109,.06));
    border: 1px solid rgba(255,95,109,.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-churn::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,95,109,.08) 0%, transparent 60%);
    animation: pulse 3s ease-in-out infinite;
}
.result-safe {
    background: linear-gradient(135deg, rgba(67,233,123,.12), rgba(67,233,123,.06));
    border: 1px solid rgba(67,233,123,.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-safe::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(67,233,123,.06) 0%, transparent 60%);
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.7; }
}
.result-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-label-churn { color: var(--red); }
.result-label-safe  { color: var(--green); }
.result-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-title-churn { color: #ff8f9a; }
.result-title-safe  { color: #7df5a5; }
.result-prob {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    color: var(--text-dim);
}
.result-prob span {
    font-size: 1.6rem;
    font-weight: 600;
}
.result-prob-churn span { color: var(--red); }
.result-prob-safe  span { color: var(--green); }

/* ── Gauge Bar ── */
.gauge-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.9rem;
}
.gauge-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    width: 90px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    flex-shrink: 0;
}
.gauge-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
    position: relative;
}
.gauge-fill-churn {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #ff5f6d, #ff8f9a);
    box-shadow: 0 0 8px rgba(255,95,109,.6);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.gauge-fill-safe {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #43e97b, #7df5a5);
    box-shadow: 0 0 8px rgba(67,233,123,.6);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.gauge-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    width: 46px;
    text-align: right;
    flex-shrink: 0;
}
.gauge-pct-churn { color: #ff8f9a; }
.gauge-pct-safe  { color: #7df5a5; }

/* ── Summary Table ── */
.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,.04);
    font-size: 0.86rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-key { color: var(--text-dim); font-weight: 400; }
.summary-val { color: var(--text); font-weight: 500; font-family: 'DM Mono', monospace; font-size: 0.82rem; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 0.75rem;
}
.risk-high   { background: rgba(255,95,109,.15); color: #ff8f9a; border: 1px solid rgba(255,95,109,.3); }
.risk-medium { background: rgba(255,166,0,.15);  color: #ffc966; border: 1px solid rgba(255,166,0,.3); }
.risk-low    { background: rgba(67,233,123,.15); color: #7df5a5; border: 1px solid rgba(67,233,123,.3); }

/* ── Info alerts ── */
.info-banner {
    background: rgba(100,255,218,.05);
    border: 1px solid rgba(100,255,218,.15);
    border-left: 3px solid var(--cyan);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.86rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin-bottom: 1.2rem;
}
.info-banner strong { color: var(--cyan); }

/* ── Streamlit overrides ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.stAlert { display: none; }
[data-testid="stForm"] { border: none !important; background: transparent !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    pkg          = load_model()
    model        = pkg['model']
    le_dict      = pkg['label_encoders']
    feature_names = pkg['feature_names']
    accuracy     = pkg['accuracy']
    roc_auc      = pkg['roc_auc']
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        Churn<span>IQ</span>
        <div style="font-size:.65rem;font-family:'DM Mono',monospace;letter-spacing:.25em;color:var(--text-dim);font-weight:400;margin-top:.15rem;">PREDICTION ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Model Vitals")
    st.markdown(f"""
    <div class="metric-pill">
        <span class="metric-pill-label">Algorithm</span>
        <span class="metric-pill-value">Random Forest</span>
    </div>
    <div class="metric-pill">
        <span class="metric-pill-label">Accuracy</span>
        <span class="metric-pill-value">{accuracy*100:.1f}%</span>
    </div>
    <div class="metric-pill">
        <span class="metric-pill-label">ROC-AUC</span>
        <span class="metric-pill-value">{roc_auc:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>#### Feature Set", unsafe_allow_html=True)
    features = ["Age", "FrequentFlyer", "AnnualIncomeClass", "ServicesOpted", "SocialMedia", "HotelBooking"]
    tags_html = "".join(f'<span class="feature-tag">{f}</span>' for f in features)
    st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="font-size:.72rem;color:var(--text-dim);line-height:1.7;padding:.8rem;border:1px solid var(--border);border-radius:8px;background:rgba(255,255,255,.02);">
        🎓 <strong style="color:var(--text)">B.Tech – Gen AI</strong><br>
        2nd Semester · Final Project<br><br>
        Predicts customer churn using<br>
        ensemble tree methodology.
    </div>
    """, unsafe_allow_html=True)


# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">✦ Airline Customer Analytics ✦</div>
    <h1 class="masthead-title">Predict <span>Churn</span></h1>
    <p class="masthead-sub">Machine-learning powered retention intelligence · Random Forest Classifier</p>
    <div class="masthead-rule"></div>
</div>
""", unsafe_allow_html=True)


# ── Two-column Layout ──────────────────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

# ────── LEFT: Input Form ───────────────────────────────────────────────────────
with col_form:
    st.markdown("""
    <div class="glass-card-title">⬡ Customer Profile</div>
    """, unsafe_allow_html=True)

    with st.form("churn_form"):
        age = st.slider("Age", min_value=18, max_value=80, value=32, step=1,
                        help="Customer age in years")

        frequent_flyer = st.selectbox(
            "Frequent Flyer Status",
            options=["No", "Yes", "No Record"],
            help="Whether the customer holds frequent flyer membership"
        )

        annual_income = st.selectbox(
            "Annual Income Class",
            options=["Low Income", "Middle Income", "High Income"],
            help="Customer's self-reported annual income tier"
        )

        services_opted = st.slider(
            "Services Opted", min_value=1, max_value=10, value=3, step=1,
            help="Number of ancillary services subscribed"
        )

        social_media = st.selectbox(
            "Account Synced to Social Media",
            options=["No", "Yes"],
            help="Whether the account is linked to social platforms"
        )

        hotel_booked = st.selectbox(
            "Booked Hotel or Not",
            options=["No", "Yes"],
            help="Whether the customer has booked associated hotel services"
        )

        submitted = st.form_submit_button(
            "⟶  Run Churn Analysis",
            type="primary",
            use_container_width=True
        )


# ────── RIGHT: Result ──────────────────────────────────────────────────────────
with col_result:
    if not submitted:
        st.markdown("""
        <div class="info-banner">
            <strong>How it works</strong><br>
            Enter the customer's profile details on the left panel, then click
            <em>Run Churn Analysis</em>. The Random Forest model will compute
            the churn probability and surface a risk classification in real-time.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card-title">⬡ Sample Risk Scenarios</div>', unsafe_allow_html=True)
        sample = pd.DataFrame({
            'Scenario':      ['◉ Low Risk',  '◎ Medium Risk', '● High Risk'],
            'Age':           [35,             30,              28],
            'Flyer':         ['No',           'Yes',           'Yes'],
            'Income':        ['High Income',  'Low Income',    'Low Income'],
            'Services':      [2,              4,               1],
            'Social Media':  ['No',           'Yes',           'No'],
            'Hotel':         ['Yes',          'No',            'No'],
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)

    else:
        # ── Encode & Predict ──
        input_df = pd.DataFrame({
            'Age':                       [age],
            'FrequentFlyer':             [frequent_flyer],
            'AnnualIncomeClass':         [annual_income],
            'ServicesOpted':             [services_opted],
            'AccountSyncedToSocialMedia':[social_media],
            'BookedHotelOrNot':          [hotel_booked]
        })

        encoded = input_df.copy()
        for col in ['FrequentFlyer', 'AnnualIncomeClass',
                    'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
            encoded[col] = le_dict[col].transform(encoded[col])
        encoded = encoded[feature_names]

        pred   = model.predict(encoded)[0]
        proba  = model.predict_proba(encoded)[0]
        p_churn = proba[1] * 100
        p_safe  = proba[0] * 100

        # Risk tier
        if p_churn >= 65:
            risk_badge = '<span class="risk-badge risk-high">High Risk</span>'
        elif p_churn >= 35:
            risk_badge = '<span class="risk-badge risk-medium">Medium Risk</span>'
        else:
            risk_badge = '<span class="risk-badge risk-low">Low Risk</span>'

        # ── Big Result Card ──
        if pred == 1:
            st.markdown(f"""
            <div class="result-churn">
                <div class="result-icon">⚠️</div>
                <div class="result-label result-label-churn">Prediction Result</div>
                <div class="result-title result-title-churn">Churn Likely</div>
                <div class="result-prob result-prob-churn">
                    <span>{p_churn:.1f}%</span> churn probability
                </div>
                {risk_badge}
            </div>
            """, unsafe_allow_html=True)
            rec = "⚡ Consider targeted retention offers, loyalty upgrades, or a personal outreach call."
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-label result-label-safe">Prediction Result</div>
                <div class="result-title result-title-safe">Retained</div>
                <div class="result-prob result-prob-safe">
                    <span>{p_safe:.1f}%</span> retention probability
                </div>
                {risk_badge}
            </div>
            """, unsafe_allow_html=True)
            rec = "✨ Customer is stable. Maintain engagement through periodic service quality checks."

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Gauges ──
        st.markdown('<div class="glass-card-title">⬡ Probability Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="gauge-row">
            <div class="gauge-label">Retained</div>
            <div class="gauge-track">
                <div class="gauge-fill-safe" style="width:{p_safe:.1f}%"></div>
            </div>
            <div class="gauge-pct gauge-pct-safe">{p_safe:.1f}%</div>
        </div>
        <div class="gauge-row">
            <div class="gauge-label">Churn</div>
            <div class="gauge-track">
                <div class="gauge-fill-churn" style="width:{p_churn:.1f}%"></div>
            </div>
            <div class="gauge-pct gauge-pct-churn">{p_churn:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Input Summary ──
        st.markdown('<div class="glass-card-title">⬡ Input Summary</div>', unsafe_allow_html=True)
        rows = {
            "Age":                         str(age),
            "Frequent Flyer":              frequent_flyer,
            "Annual Income Class":         annual_income,
            "Services Opted":              str(services_opted),
            "Social Media Synced":         social_media,
            "Hotel Booked":                hotel_booked,
        }
        rows_html = "".join(
            f'<div class="summary-row"><span class="summary-key">{k}</span>'
            f'<span class="summary-val">{v}</span></div>'
            for k, v in rows.items()
        )
        st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # ── Recommendation ──
        st.markdown(f"""
        <div class="info-banner">
            <strong>Recommendation</strong><br>{rec}
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;font-size:.72rem;color:var(--text-dim);">
    <span>🎓 B.Tech · Gen AI · 2nd Semester</span>
    <span style="font-family:'DM Mono',monospace;letter-spacing:.08em;">CHURNIQ v2.0 · RANDOM FOREST</span>
    <span>Built with Streamlit</span>
</div>
""", unsafe_allow_html=True)
