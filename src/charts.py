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
        color="median", color_continuous_scale="Viridis",
        hover_data={"n": True, "median": ":.0f"},
        height=560,
    )
    fig.update_layout(
        xaxis_title="Prix median par nuit (EUR)",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def price_map(df):
    """Grille geographique : prix moyen par case (sans fond de carte)."""
    fig = px.density_heatmap(
        df, x="longitude", y="latitude", z="price", histfunc="avg",
        nbinsx=45, nbinsy=35,
        color_continuous_scale="Viridis", range_color=(100, 400),
        range_x=(2.22, 2.47), range_y=(48.81, 48.91),
        height=560,
    )
    ratio = 1 / math.cos(math.radians(48.86))
    fig.update_yaxes(scaleanchor="x", scaleratio=ratio, title="Latitude")
    fig.update_xaxes(title="Longitude")
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
