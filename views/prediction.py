"""Page Prediction : estimation du prix d'un logement."""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.model import FEATURES



@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


@st.cache_data
def load_metrics():
    with open("models/metrics.json", encoding="ascii") as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_parquet("data/listings_clean.parquet")


df = load_data()
m = load_metrics()
model = load_model()

quartiers = sorted(df["neighbourhood_cleansed"].unique())
room_types = df["room_type"].value_counts().index.tolist()
coords = df.groupby("neighbourhood_cleansed")[["latitude", "longitude"]].median()

st.title("Estimer le prix d'un logement")
st.caption("Renseignez les caracteristiques puis cliquez sur le bouton. "
           "Estimation indicative, a ne pas utiliser pour fixer un prix reel.")

# Hors formulaire : la liste des types de bien depend du type de logement choisi
s1, s2, s3 = st.columns(3)
quartier = s1.selectbox("Quartier", quartiers,
                        index=quartiers.index("Buttes-Montmartre")
                        if "Buttes-Montmartre" in quartiers else 0)
room_type = s2.selectbox("Type de logement", room_types)
sub = df.loc[df["room_type"] == room_type, "property_type"].value_counts()
prop_types = sub[sub >= 20].index.tolist() or sub.index.tolist()
property_type = s3.selectbox("Type de bien", prop_types)

with st.form("prediction"):
    c1, c2 = st.columns(2)
    accommodates = c1.number_input("Capacite (personnes)", 1, 16, 2)
    bedrooms = c2.number_input("Chambres", 0, 10, 1)
    beds = c1.number_input("Lits", 1, 16, 1)
    bathrooms = c2.number_input("Salles de bain", 0.0, 8.0, 1.0, step=0.5)

    with st.expander("Options avancees"):
        a1, a2 = st.columns(2)
        n_reviews = a1.number_input("Nombre d'avis", 0, 1000, 10)
        rating = a2.slider("Note moyenne (ignoree si 0 avis)", 1.0, 5.0, 4.7, step=0.05)
        availability = a1.number_input("Disponibilite (jours/an)", 0, 365, 150)
        min_nights = a2.number_input("Nuits minimum", 1, 30, 2)

    submitted = st.form_submit_button("Estimer le prix", type="primary")

if submitted:
    warnings = []
    if beds * 2 < accommodates:
        warnings.append("Peu de lits pour la capacite indiquee.")
    if bedrooms > accommodates:
        warnings.append("Plus de chambres que de personnes.")
    if bathrooms > bedrooms + 2:
        warnings.append("Beaucoup de salles de bain par rapport aux chambres.")
    if room_type in ("Private room", "Shared room", "Hotel room") and bedrooms > 2:
        warnings.append("Un logement de ce type a rarement plus de 2 chambres.")
    if accommodates > 8 or bedrooms > 4:
        warnings.append("Profil rare dans les donnees : l'estimation est peu fiable.")
    for w in warnings:
        st.warning(w)

    row = {
        "neighbourhood_cleansed": quartier,
        "room_type": room_type,
        "property_type": property_type,
        "accommodates": accommodates,
        "bedrooms": float(bedrooms),
        "beds": float(beds),
        "bathrooms": float(bathrooms),
        "number_of_reviews": n_reviews,
        "review_scores_rating": rating if n_reviews > 0 else np.nan,
        "availability_365": availability,
        "minimum_nights": min_nights,
        "latitude": coords.loc[quartier, "latitude"],
        "longitude": coords.loc[quartier, "longitude"],
    }
    X = pd.DataFrame([row])[FEATURES]
    price = float(model.predict(X)[0])
    low, high = price * m["ratio_q10"], price * m["ratio_q90"]

    r1, r2 = st.columns([1, 2])
    r1.metric("Prix estime par nuit", f"{price:.0f} EUR")
    r2.metric("Fourchette (80 % des cas)", f"{low:.0f} a {high:.0f} EUR")
    st.caption(
        "Fourchette mesuree sur les logements de test : dans 80 % des cas, le prix reel "
        f"se situe entre x{m['ratio_q10']:.2f} et x{m['ratio_q90']:.2f} la prediction. "
        "Elle est large car le prix depend d'elements absents des donnees "
        "(photos, vue, decoration, etage)."
    )

st.info(
    "Limites : modele entraine sur des logements entre 20 et 1000 EUR la nuit, "
    "sejour minimum de 30 nuits maximum. Il sous-estime les logements tres chers."
)
