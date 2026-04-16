# app.py
# Multi-Agent Data Intelligence System — Professional Streamlit UI
# Color palette: Dark Brown / Warm Caramel / Off-White / Cream

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from crew import run_crew

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Data Intelligence System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --brown-dark   : #3B2A1A;
    --brown-mid    : #6B4C35;
    --brown-light  : #A07855;
    --caramel      : #C49A6C;
    --cream        : #F5EFE6;
    --offwhite     : #FAF8F5;
    --white        : #FFFFFF;
    --border       : #DDD3C8;
    --text-dark    : #2C1A0E;
    --text-mid     : #5A4033;
    --text-light   : #8C7060;
}

.stApp {
    background-color: var(--offwhite);
    font-family: 'Inter', sans-serif;
    color: var(--text-dark);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--brown-dark) !important;
    border-right: 1px solid #2A1E12;
}
section[data-testid="stSidebar"] * {
    color: var(--cream) !important;
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--caramel) !important;
    font-family: 'Playfair Display', serif;
}
section[data-testid="stSidebar"] hr {
    border-color: #5A3E2B !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(160,120,85,0.3) !important;
    border-radius: 8px !important;
    padding: 10px 12px 8px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #F5EFE6 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #A07855 !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Headings ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    color: var(--brown-dark) !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem;
}
h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.7rem !important;
    color: var(--brown-dark) !important;
}
h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--brown-mid) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--brown-dark) !important;
}

/* ── Metric cards main ── */
[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px 14px;
    box-shadow: 0 1px 4px rgba(59,42,26,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-light) !important;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--brown-dark) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-light) !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--brown-dark) !important;
    border-bottom: 2px solid var(--caramel) !important;
}
div[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-radius: 6px;
    padding: 10px 16px;
    border: 1.5px solid var(--border);
    background: var(--white);
    color: var(--brown-dark);
    transition: all 0.18s ease;
    width: 100%;
    box-shadow: 0 1px 3px rgba(59,42,26,0.08);
}
.stButton > button:hover {
    background: var(--brown-dark);
    color: var(--cream);
    border-color: var(--brown-dark);
    box-shadow: 0 3px 10px rgba(59,42,26,0.18);
}
.stDownloadButton > button {
    background: var(--brown-dark) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 20px;
}

/* ── File uploader (main area — white bg is fine here) ── */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed var(--caramel) !important;
    background: var(--white) !important;
    border-radius: 10px !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(59,42,26,0.05);
}
[data-testid="stChatInput"] > div {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--white) !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--caramel) !important;
    box-shadow: 0 0 0 2px rgba(196,154,108,0.15) !important;
}

/* ── Expanders ── */
details {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--white) !important;
    margin-bottom: 8px;
}
summary {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--brown-mid) !important;
    padding: 12px 16px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
.stCode, code {
    background: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px;
    font-size: 12px;
    color: var(--text-dark) !important;
}
[data-testid="stAlert"] {
    border-radius: 8px;
    border: none;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
}
hr { border-color: var(--border) !important; }

/* ── Agent badges ── */
.agent-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
}
.badge-cleaning      { background: #EAF2EC; color: #2E6645; border: 1px solid #B8D9C2; }
.badge-preprocessing { background: #EAF0F8; color: #1E4575; border: 1px solid #B3CCE8; }
.badge-analysis      { background: #F5EFE0; color: #7A5A1E; border: 1px solid #E0CDA0; }
.badge-chatbot       { background: #F0ECF5; color: #4A2E70; border: 1px solid #CFC0E8; }
.badge-error         { background: #F5EAEA; color: #7A2020; border: 1px solid #E0B0B0; }

/* ── Stat cards ── */
.stat-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(59,42,26,0.06);
    margin-bottom: 12px;
}
.stat-card .label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-light);
    margin-bottom: 4px;
}
.stat-card .value {
    font-size: 26px;
    font-weight: 700;
    color: var(--brown-dark);
    line-height: 1.1;
}
.stat-card .sub {
    font-size: 11px;
    color: var(--text-light);
    margin-top: 2px;
}

.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--caramel);
    margin-bottom: 6px;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--brown-light); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────────────────────
CHART_COLORS = [
    "#6B4C35", "#C49A6C", "#A07855", "#3B2A1A",
    "#DDB98A", "#8C6040", "#E8D0B0", "#4A3020"
]
PLOTLY_LAYOUT = dict(
    font_family        = "Inter, sans-serif",
    font_color         = "#2C1A0E",
    paper_bgcolor      = "#FFFFFF",
    plot_bgcolor       = "#FAF8F5",
    title_font         = dict(family="Playfair Display, serif", size=16, color="#3B2A1A"),
    legend_bgcolor     = "#FFFFFF",
    legend_bordercolor = "#DDD3C8",
    legend_borderwidth = 1,
    margin             = dict(t=55, l=15, r=15, b=15),
    colorway           = CHART_COLORS,
)
HEATMAP_SCALE = [
    [0.0,  "#3B2A1A"], [0.25, "#7C5C4E"],
    [0.5,  "#FAF8F5"], [0.75, "#C49A6C"], [1.0, "#DDB98A"],
]


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "df_raw": None, "df_current": None,
        "df_cleaned": None, "df_preprocessed": None,
        "chat_history": [], "messages": [],
        "pipeline_steps": {
            "uploaded": False, "cleaned": False,
            "preprocessed": False, "analysed": False
        },
        "last_stats": None, "last_fig": None, "file_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_file(f):
    try:
        n = f.name.lower()
        if n.endswith(".csv"):
            return pd.read_csv(f)
        elif n.endswith((".xlsx", ".xls")):
            return pd.read_excel(f)
        else:
            st.error("Unsupported format.")
    except Exception as e:
        st.error(f"Could not read file: {e}")
    return None


def reset_pipeline(keep_raw=True):
    st.session_state.df_current      = st.session_state.df_raw.copy() if keep_raw else None
    st.session_state.df_cleaned      = None
    st.session_state.df_preprocessed = None
    st.session_state.chat_history    = []
    st.session_state.messages        = []
    st.session_state.last_stats      = None
    st.session_state.last_fig        = None
    st.session_state.pipeline_steps  = {
        "uploaded": keep_raw and st.session_state.df_raw is not None,
        "cleaned": False, "preprocessed": False, "analysed": False,
    }


def add_msg(role, content, agent="chatbot", fig=None, stats=None, df_result=None):
    st.session_state.messages.append({
        "role": role, "content": content, "agent": agent,
        "fig": fig, "stats": stats, "df_result": df_result,
    })


def badge(agent):
    MAP = {
        "cleaning"      : ("Cleaning Agent",      "badge-cleaning"),
        "preprocessing" : ("Preprocessing Agent", "badge-preprocessing"),
        "analysis"      : ("Analysis Agent",      "badge-analysis"),
        "chatbot"       : ("Assistant",           "badge-chatbot"),
        "error"         : ("Error",               "badge-error"),
    }
    lbl, cls = MAP.get(agent, ("Agent", "badge-chatbot"))
    return f'<span class="agent-badge {cls}">{lbl}</span>'


def hex_to_rgba(hex_color, opacity=0.33):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — pipeline tracker + overview only (no file uploader here)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h1 style="font-size:1.5rem;margin-bottom:2px;">Multi-Agent System</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:11px;color:#A07855;letter-spacing:0.08em;'
        'text-transform:uppercase;margin-top:0;">Data Intelligence Platform</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown('<p class="section-label" style="color:#A07855;">Pipeline Status</p>',
                unsafe_allow_html=True)
    STEPS = [
        ("📂  Data Uploaded",  "uploaded"),
        ("🧹  Cleaned",        "cleaned"),
        ("⚙️  Preprocessed",   "preprocessed"),
        ("📊  Analysed",       "analysed"),
    ]
    for label, key in STEPS:
        done = st.session_state.pipeline_steps[key]
        bg   = "rgba(196,154,108,0.15)" if done else "transparent"
        col  = "#C49A6C"                if done else "#5A4033"
        fw   = "600"                    if done else "400"
        st.markdown(
            f'<div style="font-size:13px;padding:7px 10px;margin-bottom:4px;'
            f'border-radius:6px;background:{bg};color:{col};font-weight:{fw};'
            f'border:1px solid {"rgba(196,154,108,0.3)" if done else "transparent"};">'
            f'{label}</div>',
            unsafe_allow_html=True
        )

    if st.session_state.df_current is not None:
        df = st.session_state.df_current
        st.markdown("---")
        st.markdown('<p class="section-label" style="color:#A07855;">Dataset Overview</p>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows",       f"{df.shape[0]:,}")
        c2.metric("Columns",    df.shape[1])
        c3, c4 = st.columns(2)
        c3.metric("Missing",    int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        st.markdown("---")
        st.markdown('<p class="section-label" style="color:#A07855;margin-bottom:8px;">Column Types</p>',
                    unsafe_allow_html=True)
        for dtype, count in df.dtypes.value_counts().items():
            st.markdown(
                f'<div style="font-size:12px;color:#D4C9C0;padding:3px 0;">'
                f'<code style="background:#5A3E2B;color:#DDB98A;'
                f'padding:1px 6px;border-radius:3px;font-size:11px;">{dtype}</code>'
                f'&nbsp; {count} column{"s" if count > 1 else ""}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:10px;color:#5A4033;text-align:center;'
        'letter-spacing:0.05em;">v2.0 · Multi-Agent System</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1>Multi-Agent Data Intelligence System</h1>'
    '<p style="color:#8C7060;font-size:15px;margin-top:-6px;margin-bottom:20px;">'
    'Intelligent pipeline for data cleaning, preprocessing, analysis, and chat.'
    '</p>',
    unsafe_allow_html=True
)

# ── FILE UPLOAD — placed in main area so default Streamlit styling works fine ─
st.markdown(
    '<p class="section-label">📂 Upload Your Dataset</p>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    label="Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported: CSV, XLSX, XLS — max 200MB"
)

if uploaded:
    if st.session_state.file_name != uploaded.name or st.session_state.df_raw is None:
        df_loaded = load_file(uploaded)
        if df_loaded is not None:
            st.session_state.df_raw    = df_loaded
            st.session_state.file_name = uploaded.name
            reset_pipeline(keep_raw=True)
            st.success(
                f"✅ **{uploaded.name}** loaded — "
                f"**{df_loaded.shape[0]:,} rows** × **{df_loaded.shape[1]} columns**"
            )
            add_msg(
                "assistant",
                f"**{uploaded.name}** loaded — "
                f"**{df_loaded.shape[0]:,} rows** and **{df_loaded.shape[1]} columns**. "
                "Use the action buttons to begin processing, or ask me anything.",
                agent="chatbot"
            )

st.markdown("---")

# ── Welcome screen ────────────────────────────────────────────────────────────
if st.session_state.df_current is None:
    st.markdown(
        '<h2 style="font-size:1.4rem;margin-bottom:16px;">What this system can do</h2>',
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, title, body in [
        (c1, "🧹 Clean",       "Remove duplicates, fill missing values, handle outliers"),
        (c2, "⚙️ Preprocess",  "Encode categories, normalise and scale for machine learning"),
        (c3, "📊 Analyse",     "Statistics, correlations, and interactive visualisations"),
        (c4, "💬 Chat",        "Ask questions about your data in plain English"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="label">{title}</div>'
                f'<div style="font-size:13px;color:#5A4033;margin-top:6px;line-height:1.5;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.stop()


# ── Action Buttons ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Quick Actions</p>', unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("🧹  Clean Data", use_container_width=True,
                 help="Remove duplicates, fill missing values, handle outliers"):
        with st.spinner("Running Cleaning Agent…"):
            result = run_crew("clean my data", st.session_state.df_current,
                              st.session_state.chat_history)
        if result["success"]:
            st.session_state.df_current = result["df"]
            st.session_state.df_cleaned = result["df"]
            st.session_state.pipeline_steps["cleaned"] = True
        add_msg("assistant", result["message"],
                agent=result["agent"] if result["success"] else "error",
                df_result=result.get("df"))
        st.rerun()

with b2:
    if st.button("⚙️  Preprocess", use_container_width=True,
                 help="Encode categories and scale numeric columns"):
        with st.spinner("Running Preprocessing Agent…"):
            result = run_crew("preprocess my data", st.session_state.df_current,
                              st.session_state.chat_history)
        if result["success"]:
            st.session_state.df_current      = result["df"]
            st.session_state.df_preprocessed = result["df"]
            st.session_state.pipeline_steps["preprocessed"] = True
        add_msg("assistant", result["message"],
                agent=result["agent"] if result["success"] else "error",
                df_result=result.get("df"))
        st.rerun()

with b3:
    if st.button("📊  Analyse", use_container_width=True,
                 help="Run statistical analysis and generate charts"):
        with st.spinner("Running Analysis Agent…"):
            result = run_crew(
                "analyse my data — show statistics and visualize correlations",
                st.session_state.df_current, st.session_state.chat_history
            )
        if result["success"]:
            st.session_state.pipeline_steps["analysed"] = True
            st.session_state.last_stats = result.get("stats")
            st.session_state.last_fig   = result.get("fig")
        add_msg("assistant", result["message"],
                agent=result["agent"] if result["success"] else "error",
                fig=result.get("fig"), stats=result.get("stats"))
        st.rerun()

with b4:
    if st.button("🔄  Reset", use_container_width=True,
                 help="Restore the original dataset and clear all results"):
        reset_pipeline(keep_raw=True)
        add_msg("assistant", "Pipeline reset. Original dataset restored.", agent="chatbot")
        st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_data, tab_analysis = st.tabs(
    ["💬  Conversation", "🗂  Data Preview", "📊  Analysis Dashboard"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONVERSATION
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONVERSATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    chat_box = st.container(height=480, border=True)
    with chat_box:
        if not st.session_state.messages:
            st.markdown(
                '<p style="color:#8C7060;font-size:13px;font-style:italic;">'
                'Use an action button above or type a question to begin.</p>',
                unsafe_allow_html=True
            )
        for msg_idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(badge(msg.get("agent", "chatbot")), unsafe_allow_html=True)
                    st.markdown(msg["content"])
                    if msg.get("stats"):
                        with st.expander("Descriptive Statistics", key=f"stats_{msg_idx}"):
                            st.code(msg["stats"])
                    if msg.get("fig") and msg.get("agent") == "analysis":
                       import plotly.graph_objects as go
                       fig = msg["fig"]
                       fig.update_layout(**PLOTLY_LAYOUT)
                       st.plotly_chart(
                       fig,
                       use_container_width=True,
                       key=f"chat_fig_{msg_idx}"
                       )
                    if msg.get("df_result") is not None:
                        with st.expander("Updated Dataset", key=f"df_{msg_idx}"):
                            st.dataframe(msg["df_result"], use_container_width=True)

    user_input = st.chat_input("Ask anything about your data…")
    if user_input:
        add_msg("user", user_input)
        with st.spinner("Thinking…"):
            result = run_crew(user_input, st.session_state.df_current,
                              st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user",      "content": user_input})
        st.session_state.chat_history.append({"role": "assistant",  "content": result["message"]})
        if result["success"]:
            if result["agent"] == "cleaning" and result.get("df") is not None:
                st.session_state.df_current = result["df"]
                st.session_state.df_cleaned = result["df"]
                st.session_state.pipeline_steps["cleaned"] = True
            elif result["agent"] == "preprocessing" and result.get("df") is not None:
                st.session_state.df_current      = result["df"]
                st.session_state.df_preprocessed = result["df"]
                st.session_state.pipeline_steps["preprocessed"] = True
            elif result["agent"] == "analysis":
                st.session_state.pipeline_steps["analysed"] = True
                st.session_state.last_stats = result.get("stats")
                st.session_state.last_fig   = result.get("fig")
        add_msg("assistant", result["message"],
                agent=result["agent"] if result["success"] else "error",
                fig=result.get("fig"), stats=result.get("stats"),
                df_result=result.get("df"))
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    df = st.session_state.df_current

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows",           f"{df.shape[0]:,}")
    m2.metric("Columns",        df.shape[1])
    m3.metric("Missing Values", int(df.isnull().sum().sum()))
    m4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.expander("Column Summary", expanded=False):
        col_info = pd.DataFrame({
            "Column"        : df.columns,
            "Type"          : [str(df[c].dtype)           for c in df.columns],
            "Non-Null Count": [int(df[c].notna().sum())    for c in df.columns],
            "Missing"       : [int(df[c].isnull().sum())   for c in df.columns],
            "Unique Values" : [int(df[c].nunique())        for c in df.columns],
            "Sample Value"  : [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
                               for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    st.markdown('<h4 style="margin-bottom:8px;">Full Dataset</h4>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=420)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download as CSV",
        data=csv_bytes,
        file_name="processed_data.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSIS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df           = st.session_state.df_current
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = [c for c in df.columns if c not in numeric_cols]

    if not st.session_state.pipeline_steps["analysed"]:
        st.info("Click **Analyse** above to generate AI insights. "
                "Charts below update automatically from your current data.")

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.markdown('<h2 style="font-size:1.3rem;margin-bottom:4px;">Summary</h2>',
                unsafe_allow_html=True)
    if numeric_cols:
        kpi_cols = st.columns(min(len(numeric_cols), 5))
        for i, col_name in enumerate(numeric_cols[:5]):
            with kpi_cols[i]:
                mean_val = df[col_name].mean()
                std_val  = df[col_name].std()
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="label">{col_name}</div>'
                    f'<div class="value">{mean_val:,.2f}</div>'
                    f'<div class="sub">mean · std {std_val:,.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<h2 style="font-size:1.3rem;margin-bottom:4px;">Correlation Analysis</h2>',
                unsafe_allow_html=True)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)
        heat_fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=HEATMAP_SCALE, zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=11, family="Inter, sans-serif"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        ))
        heat_fig.update_layout(
            **PLOTLY_LAYOUT, title="Correlation Matrix",
            xaxis=dict(tickfont=dict(size=11), tickangle=-30),
            yaxis=dict(tickfont=dict(size=11)), height=420,
        )
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append({
                    "Variable A": corr.columns[i],
                    "Variable B": corr.columns[j],
                    "Correlation": round(corr.iloc[i, j], 3)
                })
        corr_df = pd.DataFrame(corr_pairs).sort_values("Correlation", key=abs, ascending=False)

        h_col, t_col = st.columns([3, 2])
        with h_col:
            st.plotly_chart(heat_fig, use_container_width=True)
        with t_col:
            st.markdown('<h4 style="margin-bottom:8px;">Top Correlations</h4>',
                        unsafe_allow_html=True)
            st.dataframe(corr_df.reset_index(drop=True),
                         use_container_width=True, hide_index=True, height=380)
    else:
        st.info("At least two numeric columns are required for correlation analysis.")

    # ── Distribution Grid ─────────────────────────────────────────────────────
    if numeric_cols:
        st.markdown("---")
        st.markdown('<h2 style="font-size:1.3rem;margin-bottom:4px;">Distributions</h2>',
                    unsafe_allow_html=True)
        n_cols   = min(len(numeric_cols), 3)
        n_rows   = (len(numeric_cols) + n_cols - 1) // n_cols
        dist_fig = make_subplots(
            rows=n_rows, cols=n_cols, subplot_titles=numeric_cols,
            vertical_spacing=0.12, horizontal_spacing=0.08,
        )
        for idx, col_name in enumerate(numeric_cols):
            row = idx // n_cols + 1
            col = idx %  n_cols + 1
            dist_fig.add_trace(
                go.Histogram(
                    x=df[col_name].dropna(), nbinsx=25, name=col_name,
                    marker=dict(color=CHART_COLORS[idx % len(CHART_COLORS)],
                                line=dict(width=0.5, color="#FAF8F5")),
                    showlegend=False,
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                ),
                row=row, col=col
            )
        dist_fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Distribution of Numeric Columns",
            height=max(280, 240 * n_rows),
        )
        dist_fig.update_annotations(
            font=dict(size=11, family="Inter, sans-serif", color="#3B2A1A")
        )
        st.plotly_chart(dist_fig, use_container_width=True)

    # ── Box Plots ─────────────────────────────────────────────────────────────
    if numeric_cols:
        st.markdown("---")
        st.markdown('<h2 style="font-size:1.3rem;margin-bottom:4px;">Outlier Overview</h2>',
                    unsafe_allow_html=True)
        box_fig = go.Figure()
        for i, col_name in enumerate(numeric_cols):
            box_fig.add_trace(go.Box(
                y=df[col_name].dropna(), name=col_name,
                marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                line_color=CHART_COLORS[i % len(CHART_COLORS)],
                fillcolor=hex_to_rgba(CHART_COLORS[i % len(CHART_COLORS)]),
                boxmean="sd",
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
            ))
        box_fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Box Plots — Spread and Outliers by Column",
            height=400,
            xaxis=dict(tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(box_fig, use_container_width=True)

    # ── Scatter Explorer ──────────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.markdown("---")
        st.markdown('<h2 style="font-size:1.3rem;margin-bottom:4px;">Scatter Explorer</h2>',
                    unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns([2, 2, 1])
        with sc1:
            x_col = st.selectbox("X axis", numeric_cols, key="scatter_x")
        with sc2:
            y_col = st.selectbox(
                "Y axis",
                [c for c in numeric_cols if c != x_col] or numeric_cols,
                key="scatter_y"
            )
        with sc3:
            color_col = st.selectbox("Color by", ["None"] + cat_cols, key="scatter_color")

        scatter_fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_col if color_col != "None" else None,
            color_discrete_sequence=CHART_COLORS,
            trendline="ols", trendline_color_override="#3B2A1A",
            title=f"{x_col} vs {y_col}", template="none",
        )
        scatter_fig.update_layout(
            **PLOTLY_LAYOUT, height=420,
            xaxis=dict(tickfont=dict(size=11), gridcolor="#EDE8E3"),
            yaxis=dict(tickfont=dict(size=11), gridcolor="#EDE8E3"),
        )
        scatter_fig.update_traces(
            marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color="#FAF8F5"))
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # ── Category Breakdown ────────────────────────────────────────────────────
    if cat_cols:
        st.markdown("---")
        st.markdown('<h2 style="font-size:1.3rem;margin-bottom:4px;">Category Breakdown</h2>',
                    unsafe_allow_html=True)
        selected_cat = st.selectbox(
            "Select a categorical column", cat_cols,
            label_visibility="collapsed", key="cat_col"
        )
        value_counts = df[selected_cat].value_counts().head(20).reset_index()
        value_counts.columns = [selected_cat, "Count"]

        bar_col, pie_col = st.columns([3, 2])
        with bar_col:
            bar_fig = px.bar(
                value_counts, x=selected_cat, y="Count",
                title=f"Frequency — {selected_cat}",
                color_discrete_sequence=[CHART_COLORS[0]], template="none",
            )
            bar_fig.update_layout(
                **PLOTLY_LAYOUT, height=360,
                xaxis=dict(tickfont=dict(size=11), tickangle=-30, gridcolor="#EDE8E3"),
                yaxis=dict(tickfont=dict(size=11), gridcolor="#EDE8E3"),
                bargap=0.25,
            )
            bar_fig.update_traces(marker_line_width=0)
            st.plotly_chart(bar_fig, use_container_width=True)

        with pie_col:
            pie_fig = px.pie(
                value_counts, names=selected_cat, values="Count",
                title="Proportion", color_discrete_sequence=CHART_COLORS, hole=0.45,
            )
            pie_fig.update_layout(
                **PLOTLY_LAYOUT, height=360,
                showlegend=True, legend=dict(font=dict(size=11)),
            )
            pie_fig.update_traces(
                textfont_size=11,
                marker=dict(line=dict(color="#FAF8F5", width=1.5))
            )
            st.plotly_chart(pie_fig, use_container_width=True)

    # ── AI Insight ────────────────────────────────────────────────────────────
    if st.session_state.last_stats:
        st.markdown("---")
        st.markdown('<h2 style="font-size:1.3rem;margin-bottom:4px;">AI Analysis Summary</h2>',
                    unsafe_allow_html=True)
        with st.expander("Descriptive Statistics (raw)", expanded=False):
            st.code(st.session_state.last_stats)

    if not numeric_cols:
        st.info("No numeric columns detected. Upload a dataset with numeric values "
                "to unlock the full dashboard.")
