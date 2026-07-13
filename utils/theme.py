"""
Visual theme for the dashboard: a finance/analytics-product look
(dark navy + teal/gold accents, Bloomberg/TradingView inspired),
with a session-state driven light/dark toggle.
"""

import streamlit as st

PALETTE = {
    "dark": {
        "bg": "#0E1117",
        "bg_secondary": "#151922",
        "card": "#1A1F2B",
        "card_border": "#2A3040",
        "text": "#E6E9EF",
        "text_muted": "#9AA4B2",
        "accent": "#00C2A8",       # teal
        "accent_soft": "#00C2A822",
        "gold": "#D4A73C",
        "positive": "#22C55E",
        "negative": "#EF4444",
        "plot_template": "plotly_dark",
    },
    "light": {
        "bg": "#F7F9FC",
        "bg_secondary": "#FFFFFF",
        "card": "#FFFFFF",
        "card_border": "#E3E8EF",
        "text": "#1A1F2B",
        "text_muted": "#5B6472",
        "accent": "#0E8C7A",
        "accent_soft": "#0E8C7A18",
        "gold": "#B8860B",
        "positive": "#16A34A",
        "negative": "#DC2626",
        "plot_template": "plotly_white",
    },
}


def get_mode() -> str:
    return st.session_state.get("theme_mode", "dark")


def get_colors() -> dict:
    return PALETTE[get_mode()]


def theme_toggle_control():
    """Renders a compact light/dark toggle. Call once, in the sidebar."""
    current = get_mode()
    labels = {"dark": "🌙 Dark", "light": "☀️ Light"}
    choice = st.radio(
        "Appearance", ["dark", "light"], index=0 if current == "dark" else 1,
        format_func=lambda x: labels[x], horizontal=True, label_visibility="collapsed",
        key="theme_mode_widget",
    )
    st.session_state["theme_mode"] = choice


def inject_css():
    c = get_colors()

    import plotly.io as pio
    pio.templates.default = c["plot_template"]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background-color: {c['bg']};
        color: {c['text']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {c['bg_secondary']};
        border-right: 1px solid {c['card_border']};
    }}

    /* Headings */
    h1, h2, h3 {{
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: {c['text']} !important;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {c['card']};
        border: 1px solid {c['card_border']};
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {c['text_muted']} !important;
        font-weight: 600;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    div[data-testid="stMetricValue"] {{
        color: {c['text']} !important;
        font-weight: 800 !important;
    }}

    /* Custom card / box components */
    .dash-card {{
        background-color: {c['card']};
        border: 1px solid {c['card_border']};
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }}
    .dash-card.accent-left {{
        border-left: 4px solid {c['accent']};
    }}
    .badge {{
        display: inline-block;
        background: {c['accent_soft']};
        color: {c['accent']};
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .badge.gold {{ background: {c['gold']}22; color: {c['gold']}; }}
    .muted {{ color: {c['text_muted']}; font-size: 0.92rem; }}
    .big-title {{
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.1em;
    }}
    .subtitle {{
        color: {c['text_muted']};
        font-size: 1.05rem;
        margin-bottom: 1.4em;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        font-weight: 600;
    }}

    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {c['card_border']};
        border-radius: 10px;
    }}

    /* Buttons */
    .stButton>button, .stDownloadButton>button {{
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid {c['card_border']};
    }}
    .stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {{
        background-color: {c['accent']};
        border: none;
    }}

    /* Sidebar nav section labels */
    section[data-testid="stSidebar"] .stMarkdown p {{
        color: {c['text_muted']};
    }}
    </style>
    """, unsafe_allow_html=True)


def plot_template() -> str:
    return get_colors()["plot_template"]


def style_fig(fig):
    """Apply the dashboard's plot theme to a plotly figure, in place-ish (returns fig)."""
    c = get_colors()
    fig.update_layout(
        template=plot_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=c["text"]),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig
