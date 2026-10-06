"""Graphiques Plotly de l'application."""
import math

import plotly.express as px


def price_histogram(df):
    fig = px.histogram(df, x="price", nbins=60)
    fig.update_layout(
        xaxis_title="Prix par nuit (EUR)",
        yaxis_title="Nombre de logements",
        bargap=0.05,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def price_map(df, max_points=5000):
    """Nuage de points longitude/latitude (rendu SVG, sans fond de carte)."""
    sample = df.sample(min(len(df), max_points), random_state=42)
    fig = px.scatter(
        sample,
        x="longitude",
        y="latitude",
        color="price",
        color_continuous_scale="Viridis",
        range_color=(0, 500),
        hover_name="neighbourhood_cleansed",
        hover_data={"price": ":.0f", "room_type": True,
                    "latitude": False, "longitude": False},
        opacity=0.6,
        height=560,
    )
    # A la latitude de Paris, 1 degre de longitude est plus court que 1 degre de latitude
    ratio = 1 / math.cos(math.radians(48.86))
    fig.update_yaxes(scaleanchor="x", scaleratio=ratio, title="Latitude")
    fig.update_xaxes(title="Longitude")
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      coloraxis_colorbar_title="EUR")
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
