"""
EIS Analyzer — Streamlit Application
Interactive Electrochemical Impedance Spectroscopy simulator and ML-based parameter predictor.

Three pages:
  1. 🔬 EIS Simulator  — generate impedance spectra with interactive controls
  2. 🧠 Model Training — train a 1D-CNN regression model
  3. 📊 Model Evaluation — evaluate predictions vs ground truth

Author: Dulyawat Doonyapisut (charting9@gmail.com)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io
import io
import tempfile
import os

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EIS Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Keyframe Animations ── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 15px rgba(99,102,241,0.2); }
    50%      { box-shadow: 0 0 30px rgba(99,102,241,0.4), 0 0 60px rgba(139,92,246,0.15); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(99,102,241,0.2); }
    50%      { border-color: rgba(99,102,241,0.5); }
}
@keyframes orb1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%      { transform: translate(30px, -50px) scale(1.1); }
    66%      { transform: translate(-20px, 20px) scale(0.9); }
}
@keyframes orb2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%      { transform: translate(-40px, 30px) scale(0.9); }
    66%      { transform: translate(25px, -35px) scale(1.15); }
}

/* ── Root variables ── */
:root {
    --bg-primary: #06080f;
    --bg-secondary: #0d1117;
    --bg-card: rgba(13, 17, 23, 0.75);
    --bg-card-hover: rgba(22, 27, 38, 0.85);
    --border-glow: rgba(99, 102, 241, 0.35);
    --border-subtle: rgba(99, 102, 241, 0.12);
    --accent-1: #6366f1;
    --accent-2: #8b5cf6;
    --accent-3: #06b6d4;
    --accent-4: #10b981;
    --accent-5: #f59e0b;
    --accent-6: #ec4899;
    --text-primary: #e8edf5;
    --text-secondary: #8892a4;
    --text-muted: #4b5563;
    --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    --gradient-2: linear-gradient(135deg, #06b6d4, #10b981);
    --gradient-3: linear-gradient(135deg, #f59e0b, #ef4444);
    --gradient-accent: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #06b6d4 100%);
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 20px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 40px rgba(0,0,0,0.5);
    --shadow-glow: 0 0 20px rgba(99,102,241,0.15);
}

/* ── Global ── */
html, body, [data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Animated mesh background ── */
[data-testid="stApp"] {
    background: var(--bg-primary) !important;
    position: relative;
}
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(6,182,212,0.05) 0%, transparent 50%);
    animation: orb1 20s ease-in-out infinite;
    z-index: 0;
    pointer-events: none;
}
[data-testid="stApp"]::after {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse at 70% 60%, rgba(236,72,153,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 30% 30%, rgba(16,185,129,0.04) 0%, transparent 50%);
    animation: orb2 25s ease-in-out infinite;
    z-index: 0;
    pointer-events: none;
}
[data-testid="stApp"] > * { position: relative; z-index: 1; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13,17,23,0.95) 0%, rgba(17,24,39,0.92) 100%) !important;
    backdrop-filter: blur(24px) saturate(1.3) !important;
    border-right: 1px solid var(--border-subtle) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(99,102,241,0.15) !important;
    margin: 0.8rem 0 !important;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 0.5rem 0 0.2rem 0;
}
.sidebar-logo .logo-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 0.3rem;
    filter: drop-shadow(0 0 12px rgba(99,102,241,0.5));
}
            
.sidebar-logo .logo-text {
    font-size: 1.3rem;
    font-weight: 800;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.sidebar-logo .logo-sub {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}

/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.07), rgba(6,182,212,0.05));
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
    animation: slideUp 0.6s ease-out;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-accent);
    border-radius: 20px 20px 0 0;
}
.main-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.main-header h1 {
    background: var(--gradient-accent);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -0.03em;
    animation: gradientShift 6s ease infinite;
}
.main-header p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
    line-height: 1.5;
}

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideUp 0.5s ease-out;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--gradient-accent);
    opacity: 0;
    transition: opacity 0.35s ease;
}
.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-glow);
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow), var(--shadow-md);
}
.glass-card:hover::before {
    opacity: 1;
}
.glass-card h3 {
    color: var(--accent-3) !important;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
    animation: slideUp 0.5s ease-out;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(145deg, rgba(99,102,241,0.08), rgba(6,182,212,0.04));
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(99,102,241,0.05), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(99,102,241,0.35);
    box-shadow: 0 8px 25px rgba(99,102,241,0.15);
}
.metric-card:hover::after { opacity: 1; }
.metric-card .label {
    color: var(--text-secondary);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-card .value {
    color: var(--text-primary);
    font-size: 1.6rem;
    font-weight: 800;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 2rem 0;
    border: none;
}

/* ── Feature card (landing) ── */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    padding: 2rem 1.8rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    min-height: 190px;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    transition: opacity 0.3s ease;
    opacity: 0.6;
}
.feature-card.fc-purple::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.feature-card.fc-cyan::before   { background: linear-gradient(90deg, #06b6d4, #10b981); }
.feature-card.fc-amber::before  { background: linear-gradient(90deg, #f59e0b, #ef4444); }
.feature-card.fc-pink::before   { background: linear-gradient(90deg, #ec4899, #8b5cf6); }
.feature-card.fc-green::before  { background: linear-gradient(90deg, #10b981, #06b6d4); }
.feature-card:hover {
    transform: translateY(-6px);
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow), var(--shadow-lg);
}
.feature-card:hover::before { opacity: 1; }
.feature-card .fc-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 0.8rem;
    animation: float 4s ease-in-out infinite;
}
.feature-card .fc-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.feature-card .fc-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.feature-card .fc-tags {
    margin-top: 0.8rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    justify-content: center;
}
.feature-card .fc-tag {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    background: rgba(99,102,241,0.12);
    color: var(--accent-1);
    border: 1px solid rgba(99,102,241,0.15);
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gradient-1) !important;
    background-size: 200% auto !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.2rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    animation: pulseGlow 3s ease-in-out infinite !important;
}
.stButton > button:hover {
    background-position: right center !important;
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.45), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
.stButton > button:active {
    transform: translateY(-1px) scale(0.98) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,182,212,0.1)) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    color: #10b981 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    animation: none !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(6,182,212,0.18)) !important;
    border-color: rgba(16,185,129,0.5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.2) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] label { color: var(--text-secondary) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; }

/* ── Select / radio ── */
.stRadio > label, .stSelectbox > label { color: var(--text-secondary) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { transition: color 0.2s ease;
            color: white; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.6rem; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(13,17,23,0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.3s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    border-color: var(--border-glow) !important;
    color: var(--text-primary) !important;
    background: rgba(99,102,241,0.08) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient-1) !important;
    color: white !important;
    border-color: var(--accent-1) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.25) !important;
    border-radius: 16px !important;
    padding: 1.3rem !important;
    transition: all 0.3s ease !important;
    animation: borderGlow 3s ease-in-out infinite !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.5) !important;
    background: rgba(99,102,241,0.03) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--gradient-1) !important; border-radius: 8px !important; }

/* ── Tooltip + popover ── */
[data-testid="stTooltipIcon"] { color: var(--accent-1) !important; }

/* ── Success / warning / error ── */
.stAlert { border-radius: 12px !important; border: none !important; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.5rem 0;
    margin-top: 2rem;
    border-top: 1px solid rgba(99,102,241,0.1);
    color: var(--text-muted);
    font-size: 0.75rem;
}
.app-footer a { color: var(--accent-1); text-decoration: none; }

/* ── Hide default Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly dark template helper
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.5)",
    font=dict(family="Inter, sans-serif", color="#e8edf5", size=12),
    margin=dict(l=60, r=30, t=55, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
)

COLOR_PALETTE = [
    "#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b",
    "#ef4444", "#ec4899", "#14b8a6", "#3b82f6", "#a855f7",
]


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">⚡</span>
        <div class="logo-text">EIS Analyzer</div>
        <div class="logo-sub">Impedance Studio</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔬 EIS Simulator", "🧠 Model Training", "📊 Model Evaluation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div class='app-footer'>"
        "Built by <a href='mailto:charting9@gmail.com'>Dulyawat Doonyapisut</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: EIS Simulator
# ═══════════════════════════════════════════════════════════════════════════
if page == "🔬 EIS Simulator":
    from eis_simulation import (
        F_range, sim_circuit, CIRCUIT_INFO, export_data,
    )

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>EIS Simulator</h1>
        <p>Generate synthetic Electrochemical Impedance Spectroscopy data for various equivalent circuit models</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar controls ──
    with st.sidebar:
        st.markdown("### ⚙️ Simulation Parameters")

        circuit_id = st.selectbox(
            "Circuit Model",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: CIRCUIT_INFO[x]["name"],
        )
        st.caption(CIRCUIT_INFO[circuit_id]["description"])

        size_number = st.slider("Number of Spectra", 1, 512, 10, step=1)
        number_of_point = st.slider("Points per Spectrum", 20, 200, 100, step=10)

        st.markdown("#### 📡 Frequency Range (Hz)")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            freq_min = st.number_input("Min", value=0.01, format="%.4f", min_value=0.0001)
        with col_f2:
            freq_max = st.number_input("Max", value=1e6, format="%.0f", min_value=1.0)

        st.markdown("#### 🔧 Element Ranges")

        r_exp = st.slider("Resistance (10^x Ω)", -1.0, 5.0, (-1.0, 4.0), step=0.5)
        resistance_range = [10**r_exp[0], 10**r_exp[1]]

        alpha_range = st.slider("CPE α (ideality)", 0.5, 1.0, (0.8, 1.0), step=0.01)

        q_exp = st.slider("CPE Q (10^x s^α/Ω)", -7.0, -1.0, (-5.0, -3.0), step=0.5)
        q_range = [10**q_exp[0], 10**q_exp[1]]

        sigma_exp = st.slider("Warburg σ (10^x Ω·s^-½)", -1.0, 4.0, (0.0, 3.0), step=0.5)
        sigma_range = [10**sigma_exp[0], 10**sigma_exp[1]]

        generate_btn = st.button("⚡ Generate Spectra", use_container_width=True)

    # ── Main content ──
    if generate_btn:
        with st.spinner("Simulating impedance spectra…"):
            angular_frequency, frequency_Hz = F_range(freq_min, freq_max, number_of_point)
            Zsum, Zparam = sim_circuit(
                circuit_id, size_number, number_of_point,
                angular_frequency, resistance_range,
                list(alpha_range), q_range, sigma_range,
            )

        st.session_state["sim_result"] = {
            "Zsum": Zsum, "Zparam": Zparam,
            "frequency": frequency_Hz,
            "angular_frequency": angular_frequency,
            "circuit_id": circuit_id,
            "size_number": size_number,
            "number_of_point": number_of_point,
        }

    if "sim_result" in st.session_state:
        res = st.session_state["sim_result"]
        Zsum = res["Zsum"]
        Zparam = res["Zparam"]
        frequency = res["frequency"]
        cid = res["circuit_id"]
        sz = res["size_number"]
        npt = res["number_of_point"]

        # Metrics row
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">Circuit</div>
                <div class="value">#{cid}</div>
            </div>
            <div class="metric-card">
                <div class="label">Spectra</div>
                <div class="value">{sz}</div>
            </div>
            <div class="metric-card">
                <div class="label">Points</div>
                <div class="value">{npt}</div>
            </div>
            <div class="metric-card">
                <div class="label">Parameters</div>
                <div class="value">{Zparam.shape[1]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Circuit equations
        equations = {
            1: r"Z = R_1 + \frac{1}{\frac{1}{R_2} + \frac{1}{Z_{Q_1}}}",
            2: r"Z = R_1 + \frac{1}{\frac{1}{R_2} + \frac{1}{Z_{Q_1}}} + \frac{1}{\frac{1}{R_3} + \frac{1}{Z_{Q_2}}}",
            3: r"Z = R_1 + \frac{1}{\frac{1}{Z_{Q_1}} + \frac{1}{R_2 + Z_W}}",
            4: r"Z = R_1 + \frac{1}{\frac{1}{R_2} + \frac{1}{Z_{Q_1}}} + \frac{1}{\frac{1}{Z_{Q_2}} + \frac{1}{R_3 + Z_W}}",
            5: r"Z = R_1 + \frac{1}{\frac{1}{R_2 + \frac{1}{\frac{1}{R_3+Z_W}+\frac{1}{Z_{Q_2}}}} + \frac{1}{Z_{Q_1}}}",
        }
        with st.expander("📐 Circuit Equation", expanded=True):
            st.latex(equations[cid])

        # Plots
        spectrum_idx = st.slider(
            "Select spectrum to highlight",
            0, sz - 1, 0,
            help="Choose which spectrum to show in bold; all others shown semi-transparent",
        )

        tab_nyquist, tab_bode, tab_overlay = st.tabs(["Nyquist Plot", "Bode Plots", "All Spectra Overlay"])

        with tab_nyquist:
            fig = go.Figure()
            for i in range(sz):
                opacity = 1.0 if i == spectrum_idx else 0.15
                width = 2.5 if i == spectrum_idx else 1
                fig.add_trace(go.Scatter(
                    x=Zsum[i].real, y=-Zsum[i].imag,
                    mode="lines", name=f"Spectrum {i+1}",
                    line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=width),
                    opacity=opacity,
                    showlegend=(i == spectrum_idx),
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Nyquist Plot  (Z' vs −Z'')",
                xaxis_title="Z' (Ω)",
                yaxis_title="−Z'' (Ω)",
                height=500,
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)

        with tab_bode:
            phase = np.degrees(np.arctan2(Zsum[spectrum_idx].imag, Zsum[spectrum_idx].real))
            mag = np.absolute(Zsum[spectrum_idx])

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Phase vs Frequency", "|Z| vs Frequency"))
            fig.add_trace(go.Scatter(
                x=frequency, y=phase, mode="lines",
                line=dict(color="#8b5cf6", width=2),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=frequency, y=mag, mode="lines",
                line=dict(color="#06b6d4", width=2),
            ), row=1, col=2)
            fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=1, col=1)
            fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=1, col=2)
            fig.update_yaxes(title_text="Phase (°)", row=1, col=1)
            fig.update_yaxes(title_text="|Z| (Ω)", row=1, col=2)
            fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab_overlay:
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Nyquist", "Phase", "|Z|"))
            for i in range(min(sz, 50)):
                color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                fig.add_trace(go.Scatter(
                    x=Zsum[i].real, y=-Zsum[i].imag, mode="lines",
                    line=dict(color=color, width=1), opacity=0.6, showlegend=False,
                ), row=1, col=1)
                ph = np.degrees(np.arctan2(Zsum[i].imag, Zsum[i].real))
                mg = np.absolute(Zsum[i])
                fig.add_trace(go.Scatter(
                    x=frequency, y=ph, mode="lines",
                    line=dict(color=color, width=1), opacity=0.6, showlegend=False,
                ), row=1, col=2)
                fig.add_trace(go.Scatter(
                    x=frequency, y=mg, mode="lines",
                    line=dict(color=color, width=1), opacity=0.6, showlegend=False,
                ), row=1, col=3)
            fig.update_xaxes(title_text="Z' (Ω)", row=1, col=1)
            fig.update_yaxes(title_text="−Z'' (Ω)", row=1, col=1)
            fig.update_xaxes(type="log", title_text="Freq (Hz)", row=1, col=2)
            fig.update_yaxes(title_text="Phase (°)", row=1, col=2)
            fig.update_xaxes(type="log", title_text="Freq (Hz)", row=1, col=3)
            fig.update_yaxes(title_text="|Z| (Ω)", row=1, col=3)
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            if sz > 50:
                st.info(f"Showing first 50 of {sz} spectra for performance.")
            st.plotly_chart(fig, use_container_width=True)

        # Parameters table
        st.markdown('<div class="glass-card"><h3>📋 Generated Parameters</h3></div>', unsafe_allow_html=True)
        param_names = CIRCUIT_INFO[cid]["params"]
        df_params = pd.DataFrame(Zparam, columns=param_names)
        df_params.index = [f"Spectrum {i+1}" for i in range(len(df_params))]
        st.dataframe(df_params.style.format("{:.4g}"), use_container_width=True, height=300)

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            # .mat download
            buf = io.BytesIO()
            from eis_simulation import export_data
            # For single-circuit download, wrap Zsum in array
            Circuit_spec = np.expand_dims(Zsum, axis=0)
            x_data, y_data = export_data(Circuit_spec, sz, npt, numc=1)
            mdic = {"x_data": x_data, "y_data": y_data}
            scipy.io.savemat(buf, mdic)
            st.download_button(
                "📥 Download .mat",
                data=buf.getvalue(),
                file_name=f"eis_circuit{cid}_{sz}spectra.mat",
                mime="application/octet-stream",
                use_container_width=True,
            )
        with col_dl2:
            csv_buf = df_params.to_csv(index=True)
            st.download_button(
                "📥 Download Parameters CSV",
                data=csv_buf,
                file_name=f"eis_params_circuit{cid}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        # Placeholder
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3.5rem 2rem;">
            <h3 style="color:#8b5cf6; font-size:1.3rem;">Configure parameters in the sidebar and click ⚡ Generate Spectra</h3>
            <p style="color:#64748b; margin-top:0.6rem;">Choose a circuit model, set element ranges, and simulate EIS data interactively.</p>
        </div>
        """, unsafe_allow_html=True)

        # Show available circuits with feature cards
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 🏗️ Available Circuit Models")
        fc_colors = ["fc-purple", "fc-cyan", "fc-amber", "fc-pink", "fc-green"]
        fc_icons = ["🔵", "🟣", "🟠", "🔴", "🟢"]
        cols = st.columns(3)
        for i, (cid_val, info) in enumerate(CIRCUIT_INFO.items()):
            with cols[i % 3]:
                tags_html = ''.join(f'<span class="fc-tag">{p}</span>' for p in info['params'])
                st.markdown(f"""
                <div class="feature-card {fc_colors[i]}">
                    <span class="fc-icon">{fc_icons[i]}</span>
                    <div class="fc-title">{info['name']}</div>
                    <div class="fc-desc">{info['description']}</div>
                    <div class="fc-tags">{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Training
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Training":
    st.markdown("""
    <div class="main-header">
        <h1>Model Training</h1>
        <p>Train a 1D-CNN regression model to predict EIS circuit parameters from impedance spectra</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🎛️ Training Hyperparameters")
        epochs = st.slider("Epochs", 10, 500, 100, step=10)
        batch_size = st.selectbox("Batch Size", [128, 256, 512, 1024], index=3)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:.4f}",
        )
        test_size = st.slider("Validation Split", 0.1, 0.4, 0.2, step=0.05)

    # Upload training data
    uploaded_file = st.file_uploader(
        "📂 Upload training data (.mat file)",
        type=["mat"],
        help="Upload a .mat file containing 'x_data' and 'y_data' arrays",
    )

    if uploaded_file is not None:
        st.success(f"✅ Loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")

        # Architecture preview
        with st.expander("🏗️ Model Architecture", expanded=False):
            st.markdown("""
            ```
            Input → Conv1D(64, k=32) → Conv1D(128, k=16) → Conv1D(256, k=8)
                  → Conv1D(512, k=4) → Conv1D(768, k=2)
                  → Dense(512) → Dense(512) → BatchNorm → Flatten
                  → Dense(64) → Dense(64) → Output(6)
            ```
            """)
            st.markdown("""
            <div class="metric-row">
                <div class="metric-card"><div class="label">Conv Layers</div><div class="value">5</div></div>
                <div class="metric-card"><div class="label">Dense Layers</div><div class="value">4</div></div>
                <div class="metric-card"><div class="label">Optimizer</div><div class="value">Adam</div></div>
                <div class="metric-card"><div class="label">Loss</div><div class="value">MAE</div></div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🚀 Start Training", use_container_width=True):
    try:
        from ml_model import load_and_preprocess_data
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.metrics import mean_absolute_error
        import joblib

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("Loading and preprocessing data…"):
            x_train, x_test, y_train, y_test = load_and_preprocess_data(
                tmp_path, test_size=test_size
            )
            os.unlink(tmp_path)

        # Flatten input if needed (CNN كان بياخد 3D)
        if len(x_train.shape) > 2:
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="label">Train Samples</div><div class="value">{len(x_train):,}</div></div>
            <div class="metric-card"><div class="label">Val Samples</div><div class="value">{len(x_test):,}</div></div>
            <div class="metric-card"><div class="label">Features</div><div class="value">{x_train.shape[1]}</div></div>
            <div class="metric-card"><div class="label">Outputs</div><div class="value">{y_train.shape[1]}</div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Training Gradient Boosting model…"):

            base_model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )

            model = MultiOutputRegressor(base_model)
            model.fit(x_train, y_train)

        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_test, y_test_pred)

        st.success(
            f"✅ Training Complete! — Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}"
        )

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
            joblib.dump(model, tmp_model.name)
            with open(tmp_model.name, "rb") as f:
                model_bytes = f.read()
            os.unlink(tmp_model.name)

        st.download_button(
            "📥 Download Trained Model (.pkl)",
            data=model_bytes,
            file_name="eis_gradient_boost_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.exception(e)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:3.5rem 2rem;">
        <h3 style="color:#8b5cf6; font-size:1.3rem;">📤 Upload a .mat dataset to begin training</h3>
        <p style="color:#64748b; margin-top:0.6rem;">The file should contain <code>x_data</code> (spectra) and <code>y_data</code> (parameters) arrays.</p>
    </div>
    """, unsafe_allow_html=True)

        # Architecture info cards
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 🏗️ Model Architecture")
        arch_cols = st.columns(4)
        arch_items = [
            ("🧪", "Conv1D Layers", "5 layers (64→768)", "fc-purple"),
            ("🧠", "Dense Layers", "4 layers + output", "fc-cyan"),
            ("⚙️", "Optimizer", "Adam + ReduceLR", "fc-amber"),
            ("📊", "Loss Function", "Mean Absolute Error", "fc-green"),
        ]
        for col, (icon, title, desc, color) in zip(arch_cols, arch_items):
            with col:
                st.markdown(f"""
                <div class="feature-card {color}">
                    <span class="fc-icon">{icon}</span>
                    <div class="fc-title">{title}</div>
                    <div class="fc-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.markdown("""
    <div class="main-header">
        <h1>Model Evaluation</h1>
        <p>Evaluate trained model predictions against ground-truth circuit parameters</p>
    </div>
    """, unsafe_allow_html=True)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        model_file = st.file_uploader(
            "📂 Upload trained model (.h5)",
            type=["h5"],
            help="Upload the trained Keras model file",
        )
    with col_up2:
        test_file = st.file_uploader(
            "📂 Upload test data (.mat)",
            type=["mat"],
            help="Upload a .mat file with x_data and y_data for evaluation",
        )

    n_samples = st.slider("Number of samples to evaluate", 20, 500, 100, step=10,
                           help="First N samples used for per-parameter metrics and plots")

    if model_file is not None and test_file is not None:
        if st.button("🔍 Run Evaluation", use_container_width=True):
            try:
                import tensorflow as tf
                from ml_model import load_and_preprocess_data, evaluate_model, PARAM_NAMES

                with st.spinner("Loading model & data…"):
                    # Save files to temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_m:
                        tmp_m.write(model_file.getvalue())
                        tmp_m_path = tmp_m.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp_d:
                        tmp_d.write(test_file.getvalue())
                        tmp_d_path = tmp_d.name

                    predict_model = tf.keras.models.load_model(tmp_m_path)
                    _, x_test, _, y_test = load_and_preprocess_data(
                        tmp_d_path, test_size=0.2, is_test=True,
                    )
                    os.unlink(tmp_m_path)
                    os.unlink(tmp_d_path)

                with st.spinner("Running predictions…"):
                    y_pred, metrics = evaluate_model(predict_model, x_test, y_test, n_samples=n_samples)

                # Metrics summary table
                st.markdown('<div class="glass-card"><h3>📊 Performance Metrics</h3></div>', unsafe_allow_html=True)

                metrics_df = pd.DataFrame(metrics).T
                metrics_df.index.name = "Parameter"

                # Color-coded metric cards
                cols = st.columns(len(PARAM_NAMES))
                for i, name in enumerate(PARAM_NAMES):
                    with cols[i]:
                        r2_val = metrics[name]["R²"]
                        color = "#10b981" if r2_val > 0.9 else ("#f59e0b" if r2_val > 0.7 else "#ef4444")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="label">{name} R²</div>
                            <div class="value" style="color:{color};">{r2_val:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.dataframe(
                    metrics_df.style.format("{:.4f}"),
                    use_container_width=True,
                )

                # Per-parameter comparison plots
                st.markdown('<div class="glass-card"><h3>📈 Predicted vs Actual</h3></div>', unsafe_allow_html=True)

                n_params = len(PARAM_NAMES)
                n_cols = 3
                n_rows = (n_params + n_cols - 1) // n_cols

                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=PARAM_NAMES,
                    vertical_spacing=0.12,
                    horizontal_spacing=0.08,
                )

                for i, name in enumerate(PARAM_NAMES):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    a = y_test[:n_samples, i]
                    b = y_pred[:n_samples, i]

                    fig.add_trace(go.Scatter(
                        x=list(range(n_samples)), y=a,
                        mode="markers", name=f"{name} Actual",
                        marker=dict(color="#ef4444", size=5, symbol="star"),
                        showlegend=(i == 0),
                    ), row=row, col=col)
                    fig.add_trace(go.Scatter(
                        x=list(range(n_samples)), y=b,
                        mode="markers", name=f"{name} Predicted",
                        marker=dict(color="#6366f1", size=7, symbol="circle-open", line=dict(width=1.5)),
                        showlegend=(i == 0),
                    ), row=row, col=col)

                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=350 * n_rows,
                    title="",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5,
                        font=dict(size=12),
                    ),
                )
                # Rename legend entries generically
                fig.data[0].name = "Actual"
                fig.data[1].name = "Predicted"
                st.plotly_chart(fig, use_container_width=True)

                # Scatter: predicted vs actual (1:1 line)
                st.markdown('<div class="glass-card"><h3>🎯 Prediction Accuracy (1:1 Scatter)</h3></div>', unsafe_allow_html=True)
                fig2 = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=PARAM_NAMES,
                    vertical_spacing=0.12,
                    horizontal_spacing=0.08,
                )
                for i, name in enumerate(PARAM_NAMES):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    a = y_test[:n_samples, i]
                    b = y_pred[:n_samples, i]
                    min_val = min(a.min(), b.min())
                    max_val = max(a.max(), b.max())

                    fig2.add_trace(go.Scatter(
                        x=a, y=b, mode="markers",
                        marker=dict(color="#8b5cf6", size=5, opacity=0.7),
                        showlegend=False,
                    ), row=row, col=col)
                    # 1:1 line
                    fig2.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode="lines", line=dict(color="#ef4444", dash="dash", width=1),
                        showlegend=False,
                    ), row=row, col=col)
                    fig2.update_xaxes(title_text="Actual", row=row, col=col)
                    fig2.update_yaxes(title_text="Predicted", row=row, col=col)

                fig2.update_layout(**PLOTLY_LAYOUT, height=350 * n_rows)
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")
                st.exception(e)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <h3 style="color:#8b5cf6;">Upload a trained model (.h5) and test data (.mat) to begin evaluation</h3>
            <p style="color:#64748b;">The evaluation will compute R², MAE, MAPE, and MSE for each predicted parameter.</p>
        </div>
        """, unsafe_allow_html=True)

        # Info cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color:#10b981;">R² Score</h3>
                <p style="color:#94a3b8;">Coefficient of determination — measures how well predictions match actual values. Closer to 1.0 is better.</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color:#f59e0b;">MAE / MAPE</h3>
                <p style="color:#94a3b8;">Mean Absolute Error and Mean Absolute Percentage Error — average prediction deviation.</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color:#ef4444;">MSE</h3>
                <p style="color:#94a3b8;">Mean Squared Error — penalizes larger errors more heavily than MAE.</p>
            </div>
            """, unsafe_allow_html=True)
