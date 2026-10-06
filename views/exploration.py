"""Page Exploration."""
import pandas as pd
import streamlit as st

from src.charts import (correlation_heatmap, price_by_neighbourhood,
                        price_histogram, price_map)
from src.labels import room_label
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PRICE_MAX, PRICE_MIN, TARGET

DATASET_PATH = "data/listings_clean.parquet"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_data
def load_data():
    return pd.read_parquet(DATASET_PATH)


df = load_data()

st.title("Exploration des donn\u00e9es")
st.caption(
    "Source : Inside Airbnb (donn\u00e9es publiques). Logements entre "
    f"{PRICE_MIN} et {PRICE_MAX} \u20ac la nuit, s\u00e9jour minimum de 30 nuits maximum."
)

# ---- Filtres ----
st.sidebar.header("Filtres")
quartiers = sorted(df["neighbourhood_cleansed"].unique())
types = sorted(df["room_type"].unique())
sel_q = st.sidebar.selectbox("Quartier", ["Tous les quartiers"] + quartiers)
sel_t = st.sidebar.multiselect("Type de logement", types, default=types, format_func=room_label)
pmin, pmax = st.sidebar.slider(
    "Prix par nuit (\u20ac)", PRICE_MIN, PRICE_MAX, (PRICE_MIN, PRICE_MAX), step=10
)

mask = df["room_type"].isin(sel_t) & df[TARGET].between(pmin, pmax)
if sel_q != "Tous les quartiers":
    mask &= df["neighbourhood_cleansed"] == sel_q
view = df[mask]

if view.empty:
    st.warning("Aucun logement ne correspond \u00e0 ces filtres.")
    st.stop()

# ---- Indicateurs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Logements", f"{len(view):,}".replace(",", " "))
c2.metric("Variables explicatives", len(FEATURES))
c3.metric("Valeurs manquantes", f"{view[FEATURES].isna().mean().mean():.1%}")
c4.metric("Prix m\u00e9dian", f"{view[TARGET].median():.0f} \u20ac")

# ---- Graphiques (un seul \u00e0 la fois : la carte WebGL s'initialise bien) ----
VUES = ["Distribution des prix", "Quartiers", "Carte", "Corr\u00e9lations"]
vue = st.segmented_control("Vue", VUES, default=VUES[0], label_visibility="collapsed")
vue = vue or VUES[0]

if vue == VUES[0]:
    st.plotly_chart(price_histogram(view), width="stretch")
    st.caption("Distribution tr\u00e8s asym\u00e9trique : beaucoup de logements autour de "
               "100-200 \u20ac, avec une longue queue vers les prix \u00e9lev\u00e9s.")
elif vue == VUES[1]:
    st.plotly_chart(price_by_neighbourhood(view), width="stretch")
    st.caption("Prix m\u00e9dian par nuit et par quartier (donn\u00e9es filtr\u00e9es).")
elif vue == VUES[2]:
    st.plotly_chart(price_map(view), width="stretch")
    st.caption("5 000 logements maximum affich\u00e9s (tirage al\u00e9atoire). "
               "Couleurs limit\u00e9es \u00e0 50-400 \u20ac pour rester lisibles.")
else:
    st.plotly_chart(correlation_heatmap(view, NUMERIC_FEATURES + [TARGET]), width="stretch")
    st.caption("Corr\u00e9lation lin\u00e9aire entre variables num\u00e9riques. "
               "Une corr\u00e9lation n'implique pas une relation de cause \u00e0 effet.")

with st.expander("Aper\u00e7u des donn\u00e9es filtr\u00e9es"):
    st.dataframe(view.head(100), width="stretch")
