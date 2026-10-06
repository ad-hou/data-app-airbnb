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
fr = lambda n: f"{n:,}".replace(",", " ")

st.title("A propos")

st.header("Donnees")
st.markdown(
    f"""
- **Source** : Inside Airbnb, annonces de Paris (donnees publiques).
- **Cible** : le prix par nuit (regression).
- **Nettoyage** : annonces sans prix supprimees ; prix conserves entre
  **{PRICE_MIN} et {PRICE_MAX} EUR** ; sejour minimum de **{MAX_MIN_NIGHTS} nuits maximum**
  (au-dela, ce sont des locations longue duree dont le prix n'est pas comparable).
- **Apres nettoyage** : {fr(m['n_rows_clean'])} logements, {m['n_features']} variables explicatives.
- Les valeurs manquantes des variables sont conservees puis imputees dans le modele
  (mediane + indicateur de valeur manquante).
"""
)

st.header("Methode")
st.markdown(
    f"""
1. **Decoupe** : {int((1 - TEST_SIZE) * 100)} % entrainement ({fr(m['n_train'])} logements) /
   {int(TEST_SIZE * 100)} % test ({fr(m['n_test'])} logements), graine fixe ({RANDOM_STATE}).
   Le jeu de test n'est utilise qu'une seule fois, pour la mesure finale.
2. **Comparaison** de {len(m['models_compared'])} modeles
   ({', '.join(m['models_compared'])}) par validation croisee a {CV_FOLDS} blocs,
   **sur l'entrainement uniquement**, avec une baseline (toujours predire la mediane).
3. **Critere de choix** : l'erreur absolue moyenne (MAE) en euros.
4. **Prix en logarithme** pour l'apprentissage (distribution tres asymetrique) ;
   les predictions et les metriques sont reconverties en euros.
5. **Pipeline unique** : imputation, mise a l'echelle et encodage sont appris sur
   l'entrainement seul, ce qui evite toute fuite de donnees vers le test.
"""
)

st.header("Resultats sur le jeu de test")
st.markdown(f"**Modele retenu : {m['best_model']}**")
c2, c3, c4 = st.columns(3)
c2.metric("MAE", f"{t['mae']:.1f} EUR", f"{t['mae'] - b['mae']:.1f} EUR vs baseline",
          delta_color="inverse")
c3.metric("RMSE", f"{t['rmse']:.1f} EUR")
c4.metric("R2", f"{t['r2']:.3f}")
st.caption(f"Baseline (mediane) : MAE {b['mae']:.1f} EUR, RMSE {b['rmse']:.1f} EUR, "
           f"R2 {b['r2']:.3f}.")

st.header("Limites")
st.markdown(
    f"""
- **Les prix eleves sont sous-estimes** : au-dela d'environ 500 EUR, les predictions
  se plafonnent (voir le graphique predit vs reel). Ce qui fait la valeur d'un logement
  de luxe (vue, decoration, etage, photos) n'est pas dans les donnees.
- **Fourchette large** : dans 80 % des cas, le prix reel se situe entre
  x{m['ratio_q10']:.2f} et x{m['ratio_q90']:.2f} la prediction.
- **Perimetre** : le modele n'est valable que pour des logements entre {PRICE_MIN} et
  {PRICE_MAX} EUR la nuit et des sejours de {MAX_MIN_NIGHTS} nuits maximum. Les profils rares
  (grande capacite, nombreuses chambres) sont mal couverts.
- **Importance des variables** : elle mesure de combien l'erreur augmente quand une variable
  est melangee. Ce n'est pas une relation de cause a effet, et elle peut etre biaisee
  quand des variables sont correlees (latitude/longitude, chambres/capacite).
- **Un seul instantane** : les prix viennent d'une extraction datee d'Inside Airbnb,
  ils ne tiennent compte ni de la saison ni de l'evolution du marche.
- **Estimation indicative** : l'outil ne doit pas servir a fixer un prix reel.
"""
)

st.header("Reproductibilite")
st.markdown(
    """
Code, tests et notebook d'exploration sur GitHub. Le nettoyage est dans `src/data.py`,
l'entrainement dans `src/model.py` (`python -m src.model`), et les chiffres de cette page
proviennent de `models/metrics.json`.
"""
)
