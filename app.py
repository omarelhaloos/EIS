"""
EIS Analyzer — Streamlit Application
Interactive Electrochemical Impedance Spectroscopy simulator and ML-based parameter predictor.

Three pages:
  1. 🔬 EIS Simulator      — generate impedance spectra with interactive controls
  2. 🧠 Model Training     — train a 1D-CNN regression model
  3. 📉 Corrosion Predictor — industrial corrosion rate prediction from EIS data

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
        ["🔬 EIS Simulator", "🧠 Model Training", "📉 EIS Spectrum Prediction", "🌡️ Environmental Prediction"],
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
            # .mat download — save spectra features + circuit parameters
            buf = io.BytesIO()
            # Build x_data: (size_number, 3, number_of_point) from Zsum
            imge = Zsum.imag
            phase = np.degrees(np.arctan2(Zsum.imag, Zsum.real))
            mag = np.absolute(Zsum)
            x_data = np.stack([imge, phase, mag], axis=1)  # (sz, 3, npt)
            # y_data: actual circuit parameters (sz, n_params)
            y_data = Zparam
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

            # Flatten input if needed
            if len(x_train.shape) > 2:
                x_train = x_train.reshape(x_train.shape[0], -1)
                x_test = x_test.reshape(x_test.shape[0], -1)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card"><div class="label">Train Samples</div><div class="value">{len(x_train):,}</div></div>
                <div class="metric-card"><div class="label">Val Samples</div><div class="value">{len(x_test):,}</div></div>
                <div class="metric-card"><div class="label">Features</div><div class="value">{x_train.shape[1]}</div></div>
                <div class="metric-card"><div class="label">Outputs</div><div class="value">{y_train.shape[1] if y_train.ndim > 1 else 1}</div></div>
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
# PAGE 3: EIS Spectrum Prediction
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📉 EIS Spectrum Prediction":
    from corrosion_predictor import (
        load_model as cp_load_model,
        load_spectrum,
        load_mat_spectrum,
        build_feature_vector,
        predict_corrosion,
        classify_risk,
        create_gauge_chart,
    )

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>EIS Spectrum Prediction</h1>
        <p>Predict corrosion rate from EIS spectrum data using a trained model</p>
    </div>
    """, unsafe_allow_html=True)

    # Description card
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#8b5cf6;">🔬 EIS Spectrum-Based Prediction</h3>
        <p style="color:#94a3b8;">Upload a trained model (.pkl) and EIS spectrum data (.mat or .csv)
        to predict corrosion rate. Environmental conditions are ignored for this prediction mode.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploaders
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        model_file = st.file_uploader(
            "📂 Upload trained model (.pkl)",
            type=["pkl"],
            help="Upload a scikit-learn model saved with joblib",
            key="eis_model_upload",
        )
    with col_up2:
        spectrum_file = st.file_uploader(
            "📂 Upload EIS spectrum (.mat or .csv)",
            type=["mat", "csv"],
            help="Upload a .mat file (same format as training) or a CSV with EIS impedance data",
            key="eis_spectrum_upload",
        )

    # Predict button
    if st.button("⚡ Predict Corrosion Rate", use_container_width=True, key="eis_predict"):
        if model_file is None:
            st.error("❌ Please upload a trained model (.pkl) file.")
        elif spectrum_file is None:
            st.error("❌ Please upload an EIS spectrum (.mat or .csv) file.")
        else:
            try:
                with st.spinner("Loading model…"):
                    model = cp_load_model(model_file)

                with st.spinner("Processing spectrum…"):
                    file_name = spectrum_file.name.lower()
                    if file_name.endswith(".mat"):
                        spectrum = load_mat_spectrum(spectrum_file)
                    else:
                        spectrum = load_spectrum(spectrum_file)

                with st.spinner("Building features & predicting…"):
                    # Use default environmental values as placeholders
                    # (Models trained on spectrum+env might need them, but user requested removal from UI)
                    features = build_feature_vector(
                        spectrum=spectrum,
                        material="Carbon Steel",
                        temperature=25.0,
                        pressure=1.0,
                        ph=7.0,
                        sulfur=0.0,
                        flow_velocity=0.0,
                        service_years=0,
                    )
                    corrosion_rate = predict_corrosion(model, features)
                    risk_label, risk_color, risk_bg, risk_border = classify_risk(corrosion_rate)

                # Results
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card" style="flex:2;">
                        <div class="label">Predicted Corrosion Rate</div>
                        <div class="value" style="font-size:2rem; color:{risk_color};">
                            {corrosion_rate:.4f} <span style="font-size:0.9rem;">mm/yr</span>
                        </div>
                    </div>
                    <div class="metric-card" style="flex:1; border-color:{risk_border}; background:{risk_bg};">
                        <div class="label">Risk Level</div>
                        <div class="value" style="font-size:1.8rem; color:{risk_color};">
                            {"🟢" if risk_label == "Low" else "🟡" if risk_label == "Moderate" else "🔴"} {risk_label}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Features Used</div>
                        <div class="value">{features['full'].shape[1]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                st.markdown('<div class="glass-card"><h3>📊 Corrosion Gauge</h3></div>',
                            unsafe_allow_html=True)
                gauge_fig = create_gauge_chart(corrosion_rate, risk_label, risk_color)
                st.plotly_chart(gauge_fig, use_container_width=True)

                # Risk summary table (Simplified)
                st.markdown(f"""
                <div class="glass-card">
                    <h3>📋 Risk Assessment Summary</h3>
                    <table style="width:100%; border-collapse:collapse; margin-top:0.8rem;">
                        <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                            <td style="padding:0.6rem; color:#94a3b8;">Risk Level</td>
                            <td style="padding:0.6rem; font-weight:700; color:{risk_color};">{risk_label}</td>
                        </tr>
                        <tr>
                            <td style="padding:0.6rem; color:#94a3b8;">Corrosion Rate</td>
                            <td style="padding:0.6rem; color:#e8edf5;">{corrosion_rate:.4f} mm/yr</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            except ValueError as ve:
                st.error(f"❌ Validation Error: {ve}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)

    else:
        # Placeholder
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3.5rem 2rem;">
            <h3 style="color:#8b5cf6; font-size:1.3rem;">Upload a trained model (.pkl) and EIS spectrum (.mat) to predict corrosion rate</h3>
            <p style="color:#64748b; margin-top:0.6rem;">Click ⚡ Predict to start analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Classification")
        risk_cols = st.columns(3)
        risk_items = [
            ("🟢", "Low Risk", "< 0.1 mm/yr", "Minimal corrosion — safe for continued operation", "fc-green"),
            ("🟡", "Moderate Risk", "0.1 – 0.5 mm/yr", "Noticeable corrosion — schedule maintenance", "fc-amber"),
            ("🔴", "Severe Risk", "≥ 0.5 mm/yr", "Critical corrosion — immediate action required", "fc-purple"),
        ]
        for col, (icon, title, threshold, desc, color) in zip(risk_cols, risk_items):
            with col:
                st.markdown(f"""
                <div class="feature-card {color}">
                    <span class="fc-icon">{icon}</span>
                    <div class="fc-title">{title}</div>
                    <div class="fc-desc" style="font-weight:700; margin-bottom:0.3rem;">{threshold}</div>
                    <div class="fc-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)




    # Header
    st.markdown("""
    <div class="main-header">
        <h1>EIS Spectrum Prediction</h1>
        <p>Predict corrosion rate from EIS spectrum data using a trained model</p>
    </div>
    """, unsafe_allow_html=True)

    # Load environment ranges
    _csv_path = os.path.join(os.path.dirname(__file__), "corrosion_pipeline_data.csv")
    env_ranges = get_env_ranges(_csv_path)

    # Sidebar: Environmental Conditions
    with st.sidebar:
        st.markdown("### 🌡️ Environmental Conditions")
        st.caption("Set the operating environment for corrosion prediction")

        material = st.selectbox(
            "Material",
            options=env_ranges["materials"],
            index=1,
            help="Pipeline or component material type",
        )
        temperature = st.slider(
            "Temperature (°C)",
            min_value=env_ranges["temperature_c"][0],
            max_value=env_ranges["temperature_c"][1],
            value=round((env_ranges["temperature_c"][0] + env_ranges["temperature_c"][1]) / 2, 1),
            step=0.5,
        )
        pressure = st.slider(
            "Pressure (bar)",
            min_value=env_ranges["pressure_bar"][0],
            max_value=env_ranges["pressure_bar"][1],
            value=round((env_ranges["pressure_bar"][0] + env_ranges["pressure_bar"][1]) / 2, 1),
            step=0.5,
        )
        ph = st.slider(
            "pH",
            min_value=env_ranges["ph"][0],
            max_value=env_ranges["ph"][1],
            value=round((env_ranges["ph"][0] + env_ranges["ph"][1]) / 2, 2),
            step=0.01,
        )
        sulfur = st.slider(
            "Sulfur Content (ppm)",
            min_value=env_ranges["sulfur_ppm"][0],
            max_value=env_ranges["sulfur_ppm"][1],
            value=(env_ranges["sulfur_ppm"][0] + env_ranges["sulfur_ppm"][1]) // 2,
            step=1,
        )
        flow_velocity = st.slider(
            "Flow Velocity (m/s)",
            min_value=env_ranges["flow_velocity_ms"][0],
            max_value=env_ranges["flow_velocity_ms"][1],
            value=round((env_ranges["flow_velocity_ms"][0] + env_ranges["flow_velocity_ms"][1]) / 2, 2),
            step=0.01,
        )
        service_years = st.slider(
            "Service Years",
            min_value=env_ranges["service_years"][0],
            max_value=env_ranges["service_years"][1],
            value=(env_ranges["service_years"][0] + env_ranges["service_years"][1]) // 2,
            step=1,
        )

    # Description card
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#8b5cf6;">🔬 EIS Spectrum-Based Prediction</h3>
        <p style="color:#94a3b8;">Upload a trained model (.pkl) and EIS spectrum data (.mat or .csv)
        to predict corrosion rate using the full 600-feature pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploaders
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        model_file = st.file_uploader(
            "📂 Upload trained model (.pkl)",
            type=["pkl"],
            help="Upload a scikit-learn model saved with joblib",
            key="eis_model_upload",
        )
    with col_up2:
        spectrum_file = st.file_uploader(
            "📂 Upload EIS spectrum (.mat or .csv)",
            type=["mat", "csv"],
            help="Upload a .mat file (same format as training) or a CSV with EIS impedance data",
            key="eis_spectrum_upload",
        )

    # Environment summary
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Material</div>
            <div class="value" style="font-size:0.85rem;">{material}</div>
        </div>
        <div class="metric-card">
            <div class="label">Temp</div>
            <div class="value">{temperature}°C</div>
        </div>
        <div class="metric-card">
            <div class="label">Pressure</div>
            <div class="value">{pressure} bar</div>
        </div>
        <div class="metric-card">
            <div class="label">pH</div>
            <div class="value">{ph}</div>
        </div>
        <div class="metric-card">
            <div class="label">Sulfur</div>
            <div class="value">{sulfur} ppm</div>
        </div>
        <div class="metric-card">
            <div class="label">Flow</div>
            <div class="value">{flow_velocity} m/s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Predict button
    if st.button("⚡ Predict Corrosion Rate", use_container_width=True, key="eis_predict"):
        if model_file is None:
            st.error("❌ Please upload a trained model (.pkl) file.")
        elif spectrum_file is None:
            st.error("❌ Please upload an EIS spectrum (.mat or .csv) file.")
        else:
            try:
                with st.spinner("Loading model…"):
                    model = cp_load_model(model_file)

                with st.spinner("Processing spectrum…"):
                    file_name = spectrum_file.name.lower()
                    if file_name.endswith(".mat"):
                        spectrum = load_mat_spectrum(spectrum_file)
                    else:
                        spectrum = load_spectrum(spectrum_file)

                with st.spinner("Building features & predicting…"):
                    features = build_feature_vector(
                        spectrum=spectrum,
                        material=material,
                        temperature=temperature,
                        pressure=pressure,
                        ph=ph,
                        sulfur=float(sulfur),
                        flow_velocity=flow_velocity,
                        service_years=service_years,
                    )
                    corrosion_rate = predict_corrosion(model, features)
                    risk_label, risk_color, risk_bg, risk_border = classify_risk(corrosion_rate)

                # Results
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card" style="flex:2;">
                        <div class="label">Predicted Corrosion Rate</div>
                        <div class="value" style="font-size:2rem; color:{risk_color};">
                            {corrosion_rate:.4f} <span style="font-size:0.9rem;">mm/yr</span>
                        </div>
                    </div>
                    <div class="metric-card" style="flex:1; border-color:{risk_border}; background:{risk_bg};">
                        <div class="label">Risk Level</div>
                        <div class="value" style="font-size:1.8rem; color:{risk_color};">
                            {"🟢" if risk_label == "Low" else "🟡" if risk_label == "Moderate" else "🔴"} {risk_label}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Features Used</div>
                        <div class="value">{features['full'].shape[1]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                st.markdown('<div class="glass-card"><h3>📊 Corrosion Gauge</h3></div>',
                            unsafe_allow_html=True)
                gauge_fig = create_gauge_chart(corrosion_rate, risk_label, risk_color)
                st.plotly_chart(gauge_fig, use_container_width=True)

                # Risk summary table
                st.markdown(f"""
                <div class="glass-card">
                    <h3>📋 Risk Assessment Summary</h3>
                    <table style="width:100%; border-collapse:collapse; margin-top:0.8rem;">
                        <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                            <td style="padding:0.6rem; color:#94a3b8;">Risk Level</td>
                            <td style="padding:0.6rem; font-weight:700; color:{risk_color};">{risk_label}</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                            <td style="padding:0.6rem; color:#94a3b8;">Corrosion Rate</td>
                            <td style="padding:0.6rem; color:#e8edf5;">{corrosion_rate:.4f} mm/yr</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                            <td style="padding:0.6rem; color:#94a3b8;">Material</td>
                            <td style="padding:0.6rem; color:#e8edf5;">{material}</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                            <td style="padding:0.6rem; color:#94a3b8;">Temperature</td>
                            <td style="padding:0.6rem; color:#e8edf5;">{temperature}°C</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                            <td style="padding:0.6rem; color:#94a3b8;">pH</td>
                            <td style="padding:0.6rem; color:#e8edf5;">{ph}</td>
                        </tr>
                        <tr>
                            <td style="padding:0.6rem; color:#94a3b8;">Service Years</td>
                            <td style="padding:0.6rem; color:#e8edf5;">{service_years} years</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            except ValueError as ve:
                st.error(f"❌ Validation Error: {ve}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)

    else:
        # Placeholder + risk classification
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3.5rem 2rem;">
            <h3 style="color:#8b5cf6; font-size:1.3rem;">Upload a trained model (.pkl) and EIS spectrum (.mat) to predict corrosion rate</h3>
            <p style="color:#64748b; margin-top:0.6rem;">Configure environmental conditions in the sidebar, then click ⚡ Predict.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Classification")
        risk_cols = st.columns(3)
        risk_items = [
            ("🟢", "Low Risk", "< 0.1 mm/yr", "Minimal corrosion — safe for continued operation", "fc-green"),
            ("🟡", "Moderate Risk", "0.1 – 0.5 mm/yr", "Noticeable corrosion — schedule maintenance", "fc-amber"),
            ("🔴", "Severe Risk", "≥ 0.5 mm/yr", "Critical corrosion — immediate action required", "fc-purple"),
        ]
        for col, (icon, title, threshold, desc, color) in zip(risk_cols, risk_items):
            with col:
                st.markdown(f"""
                <div class="feature-card {color}">
                    <span class="fc-icon">{icon}</span>
                    <div class="fc-title">{title}</div>
                    <div class="fc-desc" style="font-weight:700; margin-bottom:0.3rem;">{threshold}</div>
                    <div class="fc-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Environmental Prediction
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌡️ Environmental Prediction":
    from corrosion_predictor import (
        classify_risk,
        create_gauge_chart,
        get_env_ranges,
        MATERIAL_TYPES,
        train_env_model,
        predict_from_env,
    )

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Environmental Prediction</h1>
        <p>Predict corrosion rate from environmental conditions using ML</p>
    </div>
    """, unsafe_allow_html=True)

    # Load environment ranges
    _csv_path = os.path.join(os.path.dirname(__file__), "corrosion_pipeline_data.csv")
    env_ranges = get_env_ranges(_csv_path)

    # Sidebar: Environmental Conditions
    with st.sidebar:
        st.markdown("### 🌡️ Environmental Conditions")
        st.caption("Adjust conditions to predict corrosion rate")

        material = st.selectbox(
            "Material",
            options=env_ranges["materials"],
            index=1,
            help="Pipeline or component material type",
            key="env_material",
        )
        temperature = st.slider(
            "Temperature (°C)",
            min_value=env_ranges["temperature_c"][0],
            max_value=env_ranges["temperature_c"][1],
            value=round((env_ranges["temperature_c"][0] + env_ranges["temperature_c"][1]) / 2, 1),
            step=0.5,
            key="env_temperature",
        )
        pressure = st.slider(
            "Pressure (bar)",
            min_value=env_ranges["pressure_bar"][0],
            max_value=env_ranges["pressure_bar"][1],
            value=round((env_ranges["pressure_bar"][0] + env_ranges["pressure_bar"][1]) / 2, 1),
            step=0.5,
            key="env_pressure",
        )
        ph = st.slider(
            "pH",
            min_value=env_ranges["ph"][0],
            max_value=env_ranges["ph"][1],
            value=round((env_ranges["ph"][0] + env_ranges["ph"][1]) / 2, 2),
            step=0.01,
            key="env_ph",
        )
        sulfur = st.slider(
            "Sulfur Content (ppm)",
            min_value=env_ranges["sulfur_ppm"][0],
            max_value=env_ranges["sulfur_ppm"][1],
            value=(env_ranges["sulfur_ppm"][0] + env_ranges["sulfur_ppm"][1]) // 2,
            step=1,
            key="env_sulfur",
        )
        flow_velocity = st.slider(
            "Flow Velocity (m/s)",
            min_value=env_ranges["flow_velocity_ms"][0],
            max_value=env_ranges["flow_velocity_ms"][1],
            value=round((env_ranges["flow_velocity_ms"][0] + env_ranges["flow_velocity_ms"][1]) / 2, 2),
            step=0.01,
            key="env_flow",
        )
        service_years = st.slider(
            "Service Years",
            min_value=env_ranges["service_years"][0],
            max_value=env_ranges["service_years"][1],
            value=(env_ranges["service_years"][0] + env_ranges["service_years"][1]) // 2,
            step=1,
            key="env_years",
        )

    # Description card
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#10b981;">🌡️ Environmental-Based Prediction</h3>
        <p style="color:#94a3b8;">Predict corrosion rate directly from environmental conditions
        using a model trained on 5,000 pipeline corrosion records.
        No EIS spectrum upload required.</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-train the model (cached)
    @st.cache_resource(show_spinner=False)
    def _get_env_model(path):
        return train_env_model(path)

    with st.spinner("Training environmental model on pipeline data…"):
        env_result = _get_env_model(_csv_path)

    env_model = env_result["model"]

    # Model performance metrics
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance")

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Train MAE</div>
            <div class="value" style="color:#10b981;">{env_result['train_mae']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Test MAE</div>
            <div class="value" style="color:#f59e0b;">{env_result['test_mae']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Train R²</div>
            <div class="value" style="color:#10b981;">{env_result['train_r2']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Test R²</div>
            <div class="value" style="color:#f59e0b;">{env_result['test_r2']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Training Data</div>
            <div class="value">5,000 records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance + Actual vs Predicted charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="glass-card"><h3>📈 Feature Importance</h3></div>',
                    unsafe_allow_html=True)
        feat_names = env_result["feature_names"]
        feat_imp = env_result["feature_importances"]
        sorted_idx = np.argsort(feat_imp)
        fig_imp = go.Figure(go.Bar(
            x=feat_imp[sorted_idx],
            y=[feat_names[i] for i in sorted_idx],
            orientation="h",
            marker=dict(
                color=feat_imp[sorted_idx],
                colorscale=[[0, "#6366f1"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
            ),
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e8edf5"),
            height=350,
            margin=dict(l=120, r=20, t=20, b=30),
            xaxis=dict(title="Importance", gridcolor="rgba(99,102,241,0.1)"),
            yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="glass-card"><h3>🎯 Actual vs Predicted</h3></div>',
                    unsafe_allow_html=True)
        y_actual = env_result["y_test"]
        y_pred = env_result["y_test_pred"]
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_actual, y=y_pred,
            mode="markers",
            marker=dict(size=5, color="#6366f1", opacity=0.4),
            name="Predictions",
        ))
        line_min = min(y_actual.min(), y_pred.min())
        line_max = max(y_actual.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(
            x=[line_min, line_max], y=[line_min, line_max],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Perfect",
        ))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e8edf5"),
            height=350,
            margin=dict(l=50, r=20, t=20, b=50),
            xaxis=dict(title="Actual (mm/yr)", gridcolor="rgba(99,102,241,0.1)"),
            yaxis=dict(title="Predicted (mm/yr)", gridcolor="rgba(99,102,241,0.1)"),
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Current conditions summary
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Material</div>
            <div class="value" style="font-size:0.85rem;">{material}</div>
        </div>
        <div class="metric-card">
            <div class="label">Temp</div>
            <div class="value">{temperature}°C</div>
        </div>
        <div class="metric-card">
            <div class="label">Pressure</div>
            <div class="value">{pressure} bar</div>
        </div>
        <div class="metric-card">
            <div class="label">pH</div>
            <div class="value">{ph}</div>
        </div>
        <div class="metric-card">
            <div class="label">Sulfur</div>
            <div class="value">{sulfur} ppm</div>
        </div>
        <div class="metric-card">
            <div class="label">Flow</div>
            <div class="value">{flow_velocity} m/s</div>
        </div>
        <div class="metric-card">
            <div class="label">Service</div>
            <div class="value">{service_years} yr</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Predict button
    if st.button("⚡ Predict Corrosion Rate", use_container_width=True, key="env_predict"):
        try:
            with st.spinner("Predicting…"):
                corrosion_rate = predict_from_env(
                    env_model,
                    material=material,
                    temperature=temperature,
                    pressure=pressure,
                    ph=ph,
                    sulfur=float(sulfur),
                    flow_velocity=flow_velocity,
                    service_years=service_years,
                )
                risk_label, risk_color, risk_bg, risk_border = classify_risk(corrosion_rate)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card" style="flex:2;">
                    <div class="label">Predicted Corrosion Rate</div>
                    <div class="value" style="font-size:2rem; color:{risk_color};">
                        {corrosion_rate:.4f} <span style="font-size:0.9rem;">mm/yr</span>
                    </div>
                </div>
                <div class="metric-card" style="flex:1; border-color:{risk_border}; background:{risk_bg};">
                    <div class="label">Risk Level</div>
                    <div class="value" style="font-size:1.8rem; color:{risk_color};">
                        {"🟢" if risk_label == "Low" else "🟡" if risk_label == "Moderate" else "🔴"} {risk_label}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Model Type</div>
                    <div class="value" style="font-size:0.8rem;">GradientBoosting</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            gauge_fig = create_gauge_chart(corrosion_rate, risk_label, risk_color)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Risk summary table
            st.markdown(f"""
            <div class="glass-card">
                <h3>📋 Risk Assessment Summary</h3>
                <table style="width:100%; border-collapse:collapse; margin-top:0.8rem;">
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">Risk Level</td>
                        <td style="padding:0.6rem; font-weight:700; color:{risk_color};">{risk_label}</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">Corrosion Rate</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{corrosion_rate:.4f} mm/yr</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">Material</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{material}</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">Temperature</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{temperature}°C</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">Pressure</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{pressure} bar</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">pH</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{ph}</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">Sulfur</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{sulfur} ppm</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(99,102,241,0.15);">
                        <td style="padding:0.6rem; color:#94a3b8;">Flow Velocity</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{flow_velocity} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding:0.6rem; color:#94a3b8;">Service Years</td>
                        <td style="padding:0.6rem; color:#e8edf5;">{service_years} years</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)

    # Risk Classification + Pipeline data preview
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Risk Classification")
    risk_cols = st.columns(3)
    risk_items = [
        ("🟢", "Low Risk", "< 0.1 mm/yr", "Minimal corrosion — safe for continued operation", "fc-green"),
        ("🟡", "Moderate Risk", "0.1 – 0.5 mm/yr", "Noticeable corrosion — schedule maintenance", "fc-amber"),
        ("🔴", "Severe Risk", "≥ 0.5 mm/yr", "Critical corrosion — immediate action required", "fc-purple"),
    ]
    for col, (icon, title, threshold, desc, color) in zip(risk_cols, risk_items):
        with col:
            st.markdown(f"""
            <div class="feature-card {color}">
                <span class="fc-icon">{icon}</span>
                <div class="fc-title">{title}</div>
                <div class="fc-desc" style="font-weight:700; margin-bottom:0.3rem;">{threshold}</div>
                <div class="fc-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Pipeline Corrosion Dataset")
    st.caption("Reference data used for environmental condition ranges and model training")
    try:
        _pipeline_df = pd.read_csv(_csv_path)
        st.dataframe(
            _pipeline_df.head(50).style.format({
                "temperature_c": "{:.1f}",
                "pressure_bar": "{:.1f}",
                "ph": "{:.2f}",
                "flow_velocity_ms": "{:.2f}",
                "corrosion_rate_mmpy": "{:.3f}",
            }),
            use_container_width=True,
            height=350,
        )
    except Exception:
        st.info("Pipeline dataset not available for preview.")


