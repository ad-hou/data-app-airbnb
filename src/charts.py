"""Graphiques Plotly de l'application."""
import plotly.express as px

import src.style  # noqa: F401  (enregistre le gabarit Plotly)
from src.style import PRIMARY

PARIS_CENTER = {"lat": 48.8566, "lon": 2.3522}


def price_histogram(df):
    fig = px.histogram(df, x="price", nbins=60, color_discrete_sequence=[PRIMARY])
    fig.update_layout(
        xaxis_title="Prix par nuit (\u20ac)",
        yaxis_title="Nombre de logements",
        bargap=0.05,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def price_by_neighbourhood(df):
    """Prix median par quartier, barres horizontales triees."""
    g = (
        df.groupby("neighbourhood_cleansed")["price"]
        .agg(median="median", n="size")
        .reset_index()
        .sort_values("median")
    )
    fig = px.bar(
        g, x="median", y="neighbourhood_cleansed", orientation="h",
        color_discrete_sequence=[PRIMARY],
        hover_data={"n": True, "median": ":.0f"},
        height=560,
    )
    fig.update_layout(
        xaxis_title="Prix m\u00e9dian par nuit (\u20ac)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def price_map(df, max_points=5000):
    """Vraie carte : un point par logement, colore selon le prix."""
    sample = df.sample(min(len(df), max_points), random_state=42)
    fig = px.scatter_map(
        sample,
        lat="latitude",
        lon="longitude",
        color="price",
        color_continuous_scale="Viridis",
        range_color=(50, 400),
        hover_name="neighbourhood_cleansed",
        hover_data={"price": ":.0f", "room_type": True,
                    "latitude": False, "longitude": False},
        center=PARIS_CENTER,
        zoom=11,
        opacity=0.7,
        height=600,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        map_style="carto-positron",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar_title="\u20ac",
    )
    return fig


def correlation_heatmap(df, columns):
    corr = df[columns].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        height=560,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig
