"""Page A propos : donnees, methode, resultats, limites."""
import json

import streamlit as st

from src.data import MAX_MIN_NIGHTS, PRICE_MAX, PRICE_MIN
from src.model import CV_FOLDS, RANDOM_STATE, TEST_SIZE


@st.cache_data
def load_metrics():
    with open("models/metrics.json", encoding="ascii") as f:
        return json.load(f)


m = load_metrics()
t, b = m["test"], m["baseline_test"]


def fr(n):
    return f"{n:,}".replace(",", " ")


st.title("\u00c0 propos")

st.header("Donn\u00e9es")
st.markdown(
    f"""
- **Source** : Inside Airbnb, annonces de Paris (donn\u00e9es publiques).
- **Cible** : le prix par nuit (r\u00e9gression).
- **Nettoyage** : annonces sans prix supprim\u00e9es ; prix conserv\u00e9s entre
  **{PRICE_MIN} et {PRICE_MAX} \u20ac** ; s\u00e9jour minimum de **{MAX_MIN_NIGHTS} nuits maximum**
  (au-del\u00e0, ce sont des locations longue dur\u00e9e dont le prix n'est pas comparable).
- **Apr\u00e8s nettoyage** : {fr(m['n_rows_clean'])} logements, {m['n_features']} variables explicatives.
- Les valeurs manquantes des variables sont conserv\u00e9es puis imput\u00e9es dans le mod\u00e8le
  (m\u00e9diane + indicateur de valeur manquante).
"""
)

st.header("M\u00e9thode")
st.markdown(
    f"""
1. **D\u00e9coupe** : {int((1 - TEST_SIZE) * 100)} % entra\u00eenement ({fr(m['n_train'])} logements) /
   {int(TEST_SIZE * 100)} % test ({fr(m['n_test'])} logements), graine fixe ({RANDOM_STATE}).
   Le jeu de test n'est utilis\u00e9 qu'une seule fois, pour la mesure finale.
2. **Comparaison** de {len(m['models_compared'])} mod\u00e8les
   ({', '.join(m['models_compared'])}) par validation crois\u00e9e \u00e0 {CV_FOLDS} blocs,
   **sur l'entra\u00eenement uniquement**, avec une baseline (toujours pr\u00e9dire la m\u00e9diane).
3. **Crit\u00e8re de choix** : l'erreur absolue moyenne (MAE) en euros.
4. **Prix en logarithme** pour l'apprentissage (distribution tr\u00e8s asym\u00e9trique) ;
   les pr\u00e9dictions et les m\u00e9triques sont reconverties en euros.
5. **Pipeline unique** : imputation, mise \u00e0 l'\u00e9chelle et encodage sont appris sur
   l'entra\u00eenement seul, ce qui \u00e9vite toute fuite de donn\u00e9es vers le test.
"""
)

st.header("R\u00e9sultats sur le jeu de test")
st.markdown(f"**Mod\u00e8le retenu : {m['best_model']}**")
c2, c3, c4 = st.columns(3)
c2.metric("MAE", f"{t['mae']:.1f} \u20ac", f"{t['mae'] - b['mae']:.1f} \u20ac vs baseline",
          delta_color="inverse")
c3.metric("RMSE", f"{t['rmse']:.1f} \u20ac")
c4.metric("R\u00b2", f"{t['r2']:.3f}")
st.caption(f"Baseline (m\u00e9diane) : MAE {b['mae']:.1f} \u20ac, RMSE {b['rmse']:.1f} \u20ac, "
           f"R\u00b2 {b['r2']:.3f}.")

st.header("Limites")
st.markdown(
    f"""
- **Les prix \u00e9lev\u00e9s sont sous-estim\u00e9s** : au-del\u00e0 d'environ 500 \u20ac, les pr\u00e9dictions
  se plafonnent (voir le graphique pr\u00e9dit vs r\u00e9el). Ce qui fait la valeur d'un logement
  de luxe (vue, d\u00e9coration, \u00e9tage, photos) n'est pas dans les donn\u00e9es.
- **Fourchette large** : dans 80 % des cas, le prix r\u00e9el se situe entre
  \u00d7{m['ratio_q10']:.2f} et \u00d7{m['ratio_q90']:.2f} la pr\u00e9diction.
- **P\u00e9rim\u00e8tre** : le mod\u00e8le n'est valable que pour des logements entre {PRICE_MIN} et
  {PRICE_MAX} \u20ac la nuit et des s\u00e9jours de {MAX_MIN_NIGHTS} nuits maximum. Les profils rares
  (grande capacit\u00e9, nombreuses chambres) sont mal couverts.
- **Importance des variables** : elle mesure de combien l'erreur augmente quand une variable
  est m\u00e9lang\u00e9e. Ce n'est pas une relation de cause \u00e0 effet, et elle peut \u00eatre biais\u00e9e
  quand des variables sont corr\u00e9l\u00e9es (latitude/longitude, chambres/capacit\u00e9).
- **Un seul instantan\u00e9** : les prix viennent d'une extraction dat\u00e9e d'Inside Airbnb,
  ils ne tiennent compte ni de la saison ni de l'\u00e9volution du march\u00e9.
- **Estimation indicative** : l'outil ne doit pas servir \u00e0 fixer un prix r\u00e9el.
"""
)

st.header("Reproductibilit\u00e9")
st.markdown(
    """
Code, tests et notebook d'exploration sur GitHub. Le nettoyage est dans `src/data.py`,
l'entra\u00eenement dans `src/model.py` (`python -m src.model`), et les chiffres de cette page
proviennent de `models/metrics.json`.
"""
)
