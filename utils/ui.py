"""
Reusable presentation components: KPI rows, objective/methodology/interpretation/
conclusion boxes, section headers. Used across every page so each analysis reads
like a mini research brief, not a raw notebook dump.
"""

import streamlit as st


def page_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(f"""
    <div class="big-title">{icon} {title}</div>
    <div class="subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)


def section_badge(text: str, gold: bool = False):
    cls = "badge gold" if gold else "badge"
    st.markdown(f'<span class="{cls}">{text}</span>', unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, help_text_or_None)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        help_text = item[2] if len(item) > 2 else None
        with col:
            st.metric(label, value, help=help_text)


def objective_box(text: str):
    st.markdown(f"""
    <div class="dash-card accent-left">
    <span class="badge">🎯 Objective</span><br>
    {text}
    </div>
    """, unsafe_allow_html=True)


def methodology_box(text: str):
    with st.expander("🧪 Methodology — how this was computed", expanded=False):
        st.markdown(text)


def interpretation_box(text: str):
    st.markdown(f"""
    <div class="dash-card">
    <span class="badge">💡 Interpretation</span><br>
    {text}
    </div>
    """, unsafe_allow_html=True)


def conclusion_box(text: str):
    st.markdown(f"""
    <div class="dash-card accent-left">
    <span class="badge gold">✅ Conclusion</span><br>
    {text}
    </div>
    """, unsafe_allow_html=True)


def business_box(text: str):
    st.markdown(f"""
    <div class="dash-card">
    <span class="badge gold">📈 Business Implication</span><br>
    {text}
    </div>
    """, unsafe_allow_html=True)


def info_note(text: str):
    st.info(text, icon="ℹ️")
