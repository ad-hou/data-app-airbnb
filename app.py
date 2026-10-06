"""Application Streamlit - page Exploration (accueil)."""
import pandas as pd
import streamlit as st

from src.charts import (correlation_heatmap, price_by_neighbourhood,
                        price_histogram, price_map)
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PRICE_MAX, PRICE_MIN, TARGET

DATASET_PATH = "data/listings_clean.parquet"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(page_title="Airbnb Paris", layout="wide")


@st.cache_data
def load_data():
    return pd.read_parquet(DATASET_PATH)


df = load_data()

st.title("Airbnb Paris - Exploration des donnees")
st.caption(
    "Source : Inside Airbnb (donnees publiques). Logements entre "
    f"{PRICE_MIN} et {PRICE_MAX} EUR la nuit, sejour minimum de 30 nuits maximum."
)

# ---- Filtres ----
st.sidebar.header("Filtres")
quartiers = sorted(df["neighbourhood_cleansed"].unique())
types = sorted(df["room_type"].unique())
sel_q = st.sidebar.multiselect("Quartier", quartiers, default=quartiers)
sel_t = st.sidebar.multiselect("Type de logement", types, default=types)
pmin, pmax = st.sidebar.slider(
    "Prix par nuit (EUR)", PRICE_MIN, PRICE_MAX, (PRICE_MIN, PRICE_MAX), step=10
)

mask = (
    df["neighbourhood_cleansed"].isin(sel_q)
    & df["room_type"].isin(sel_t)
    & df[TARGET].between(pmin, pmax)
)
view = df[mask]

if view.empty:
    st.warning("Aucun logement ne correspond a ces filtres.")
    st.stop()

# ---- Indicateurs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Logements", f"{len(view):,}".replace(",", " "))
c2.metric("Variables explicatives", len(FEATURES))
c3.metric("Valeurs manquantes", f"{view[FEATURES].isna().mean().mean():.1%}")
c4.metric("Prix median", f"{view[TARGET].median():.0f} EUR")

# ---- Graphiques ----
tab1, tab2, tab3, tab4 = st.tabs(
    ["Distribution des prix", "Quartiers", "Carte", "Correlations"]
)

with tab1:
    st.plotly_chart(price_histogram(view), width="stretch")
    st.caption("Distribution tres asymetrique : beaucoup de logements autour de 100-200 EUR, "
               "une longue queue vers les prix eleves.")

with tab2:
    st.plotly_chart(price_by_neighbourhood(view), width="stretch")
    st.caption("Prix median par nuit et par quartier (donnees filtrees).")

with tab3:
    st.plotly_chart(price_map(view), width="stretch")
    st.caption("Prix moyen par case geographique (cases vides = aucun logement). "
               "Echelle de couleur limitee a 100-400 EUR.")

with tab4:
    st.plotly_chart(correlation_heatmap(view, NUMERIC_FEATURES + [TARGET]), width="stretch")
    st.caption("Correlation lineaire entre variables numeriques. "
               "Une correlation n'implique pas une relation de cause a effet.")

with st.expander("Apercu des donnees filtrees"):
    st.dataframe(view.head(100), width="stretch")
