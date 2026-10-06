"""Page Modele : performances, comparaison, predit vs reel, importance."""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data import TARGET
from src.model import FEATURES, RANDOM_STATE, TEST_SIZE

LABELS = {
    "accommodates": "Capacit\u00e9 (personnes)",
    "bedrooms": "Chambres",
    "beds": "Lits",
    "bathrooms": "Salles de bain",
    "number_of_reviews": "Nombre d'avis",
    "review_scores_rating": "Note moyenne",
    "availability_365": "Disponibilit\u00e9 (jours/an)",
    "minimum_nights": "Nuits minimum",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "neighbourhood_cleansed": "Quartier",
    "room_type": "Type de logement",
    "property_type": "Type de bien",
}


@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


@st.cache_data
def load_metrics():
    with open("models/metrics.json", encoding="ascii") as f:
        return json.load(f)


@st.cache_data
def load_test_set():
    df = pd.read_parquet("data/listings_clean.parquet")
    _, X_test, _, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_test, y_test


@st.cache_data
def predict_test():
    X_test, y_test = load_test_set()
    return load_model().predict(X_test), y_test.to_numpy()


@st.cache_data
def compute_importance(n=2000):
    X_test, y_test = load_test_set()
    Xs = X_test.sample(min(n, len(X_test)), random_state=RANDOM_STATE)
    ys = y_test.loc[Xs.index]
    res = permutation_importance(
        load_model(), Xs, ys, scoring="neg_mean_absolute_error",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=1,
    )
    out = pd.DataFrame({
        "variable": [LABELS[f] for f in Xs.columns],
        "hausse_mae": res.importances_mean,
    })
    return out.sort_values("hausse_mae")


m = load_metrics()
t, b = m["test"], m["baseline_test"]

st.title("Mod\u00e8le de pr\u00e9diction du prix")
st.caption(
    f"Mod\u00e8le retenu : {m['best_model']}. \u00c9valu\u00e9 sur {m['n_test']:,} logements "
    "mis de c\u00f4t\u00e9 avant l'entra\u00eenement (jamais vus par le mod\u00e8le).".replace(",", " ")
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE (erreur moyenne)", f"{t['mae']:.1f} \u20ac",
          f"{t['mae'] - b['mae']:.1f} \u20ac vs baseline", delta_color="inverse")
c2.metric("RMSE", f"{t['rmse']:.1f} \u20ac")
c3.metric("R\u00b2", f"{t['r2']:.3f}")
c4.metric("MAE de la baseline", f"{b['mae']:.1f} \u20ac")
st.caption("La baseline pr\u00e9dit toujours le prix m\u00e9dian : c'est le point de comparaison "
           "minimal qu'un mod\u00e8le utile doit battre.")

st.subheader("Comparaison des mod\u00e8les")
cmp = pd.DataFrame({"modele": list(m["cv_mae"]), "mae": list(m["cv_mae"].values())})
cmp = cmp.sort_values("mae", ascending=False)
fig = px.bar(cmp, x="mae", y="modele", orientation="h", text_auto=".1f")
fig.update_layout(xaxis_title="MAE en validation crois\u00e9e (\u20ac) \u2013 plus bas = mieux",
                  yaxis_title="", margin=dict(l=10, r=10, t=10, b=10), height=300)
st.plotly_chart(fig, width="stretch")
st.caption("Compar\u00e9s uniquement sur les donn\u00e9es d'entra\u00eenement (validation crois\u00e9e "
           "\u00e0 3 blocs) : le jeu de test n'a servi qu'\u00e0 la mesure finale.")

st.subheader("Prix pr\u00e9dit vs prix r\u00e9el")
pred, real = predict_test()
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(pred), size=min(3000, len(pred)), replace=False)
pv = pd.DataFrame({"reel": real[idx], "predit": pred[idx]})
fig2 = px.scatter(pv, x="reel", y="predit", opacity=0.35, height=520)
fig2.add_shape(type="line", x0=0, y0=0, x1=1000, y1=1000,
               line=dict(color="red", dash="dash"))
fig2.update_layout(xaxis_title="Prix r\u00e9el (\u20ac)", yaxis_title="Prix pr\u00e9dit (\u20ac)",
                   margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, width="stretch")
st.caption("Un mod\u00e8le parfait aligne tous les points sur la diagonale rouge. "
           "3 000 logements de test affich\u00e9s. Le mod\u00e8le sous-estime les prix \u00e9lev\u00e9s.")

st.subheader("Importance des variables")
with st.spinner("Calcul de l'importance (une seule fois)..."):
    imp = compute_importance()
fig3 = px.bar(imp, x="hausse_mae", y="variable", orientation="h", height=480)
fig3.update_layout(xaxis_title="Hausse de l'erreur (\u20ac) quand la variable est m\u00e9lang\u00e9e",
                   yaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig3, width="stretch")
st.caption("M\u00e9thode par permutation, sur 2 000 logements de test : plus la barre est "
           "longue, plus la variable compte pour les pr\u00e9dictions. "
           "Ce n'est pas une relation de cause \u00e0 effet.")
