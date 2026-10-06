"""Style commun : palette, gabarit Plotly, CSS."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PRIMARY = "#0F766E"
ACCENT = "#E4572E"
MUTED = "#64748B"
GRID = "#E6ECEF"

pio.templates["pro"] = go.layout.Template(
    layout=dict(
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#12202B"),
        colorway=[PRIMARY, ACCENT, "#2563EB", "#CA8A04", MUTED],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#CBD5DB"),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#CBD5DB"),
    )
)
pio.templates.default = "pro"

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
h2, h3 {font-weight: 650; letter-spacing: -0.01em;}
[data-testid="stMetric"] {
    background: #F3F6F8; border: 1px solid #E1E8ED;
    border-radius: 12px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] {color: #64748B;}
[data-testid="stSidebar"] {border-right: 1px solid #E1E8ED;}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)
