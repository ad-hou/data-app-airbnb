# Airbnb Paris – exploration et prédiction de prix

Application Streamlit pour explorer les annonces Airbnb de Paris et estimer
le prix par nuit d'un logement à partir de ses caractéristiques.

**Démo en ligne** : _lien à ajouter après déploiement_

## Ce que fait l'application

- **Exploration** : filtres (quartier, type, prix), distribution des prix, prix par quartier, carte, corrélations.
- **Modèle** : métriques mesurées sur un jeu de test, comparaison des modèles, prix prédit vs réel, importance des variables.
- **Prédiction** : formulaire, prix estimé et fourchette à 80 %.
- **À propos** : données, méthode, limites.

## Données

Inside Airbnb, annonces de Paris (données publiques) : http://insideairbnb.com/get-the-data/

Nettoyage : prix entre 20 et 1000 € la nuit, séjour minimum de 30 nuits maximum.
45 973 logements retenus, 13 variables explicatives.

## Méthode

- Découpe 80 % entraînement / 20 % test (graine 42) ; le test n'est utilisé qu'une fois.
- Comparaison de 3 modèles + une baseline par validation croisée à 3 blocs, sur l'entraînement.
- Prix en logarithme pour l'apprentissage ; métriques reconverties en euros.
- Pipeline scikit-learn unique (imputation, échelle, encodage) : pas de fuite de données.

## Résultats (jeu de test, 9 195 logements)

| Indicateur | Gradient Boosting | Baseline (médiane) |
|---|---|---|
| MAE | 65,6 € | 126,8 € |
| RMSE | 104,6 € | 189,7 € |
| R² | 0,667 | -0,096 |

Valeurs lues dans `models/metrics.json`.

## Limites

- Les prix élevés (au-delà de ~500 €) sont sous-estimés.
- La fourchette à 80 % est large (×0,68 à ×1,49 la prédiction).
- Valable uniquement pour 20-1000 € et des séjours de 30 nuits maximum.
- L'importance par permutation n'est pas une relation de cause à effet.
- Un seul instantané des données, sans saisonnalité.

## Lancer en local

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
pip install -r requirements-dev.txt

# Télécharger listings.csv.gz (Paris) depuis Inside Airbnb dans data/raw/
python -m src.make_dataset         # jeu de données nettoyé
python -m src.model                # entraînement + models/model.joblib
python -m pytest -q                # tests
streamlit run app.py
```

## Structure

```
app.py                  point d'entrée : navigation et style
views/                  pages : exploration, modèle, prédiction, à propos
src/data.py             chargement et nettoyage
src/model.py            entraînement et sauvegarde
src/charts.py           graphiques Plotly
src/labels.py           libellés français
src/style.py            thème et gabarit des graphiques
notebooks/              exploration
tests/                  tests pytest
models/                 modèle et métriques
```
