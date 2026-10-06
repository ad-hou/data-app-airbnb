# Airbnb Paris - exploration et prediction de prix

Application Streamlit qui permet d'explorer les annonces Airbnb de Paris
et d'estimer le prix par nuit d'un logement a partir de ses caracteristiques.

**Demo en ligne** : _lien a ajouter apres deploiement_

## Ce que fait l'application

- **Exploration** : filtres (quartier, type, prix), distribution des prix, prix par quartier, carte, correlations.
- **Modele** : metriques mesurees sur un jeu de test, comparaison des modeles, prix predit vs reel, importance des variables.
- **Prediction** : formulaire, prix estime et fourchette a 80 %.
- **A propos** : donnees, methode, limites.

## Donnees

Inside Airbnb, annonces de Paris (donnees publiques) : http://insideairbnb.com/get-the-data/

Nettoyage : prix entre 20 et 1000 EUR la nuit, sejour minimum de 30 nuits maximum.
45 973 logements retenus, 13 variables explicatives.

## Methode

- Decoupe 80 % entrainement / 20 % test (graine 42), le test n'est utilise qu'une fois.
- Comparaison de 3 modeles + baseline par validation croisee a 3 blocs, sur l'entrainement.
- Prix en logarithme pour l'apprentissage ; metriques reconverties en euros.
- Pipeline scikit-learn unique (imputation, echelle, encodage) : pas de fuite de donnees.

## Resultats (jeu de test, 9 195 logements)

| Indicateur | Gradient Boosting | Baseline (mediane) |
|---|---|---|
| MAE | 65.6 EUR | 126.8 EUR |
| RMSE | 104.6 EUR | 189.7 EUR |
| R2 | 0.667 | -0.096 |

Valeurs lues dans `models/metrics.json`.

## Limites

- Les prix eleves (au-dela de ~500 EUR) sont sous-estimes.
- La fourchette a 80 % est large (x0.68 a x1.49 la prediction).
- Valable uniquement pour 20-1000 EUR et sejours de 30 nuits maximum.
- L'importance par permutation n'est pas une relation de cause a effet.
- Un seul instantane des donnees, sans saisonnalite.

## Lancer en local

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
pip install -r requirements-dev.txt

# Telecharger listings.csv.gz (Paris) depuis Inside Airbnb dans data/raw/
python -m src.make_dataset         # jeu de donnees nettoye
python -m src.model                # entrainement + models/model.joblib
pytest -q                          # tests
streamlit run app.py
```

## Structure
app.py page Exploration
pages/ Modele, Prediction, A propos
src/data.py chargement et nettoyage
src/model.py entrainement et sauvegarde
src/charts.py graphiques Plotly
notebooks/ exploration
tests/ tests pytest
models/ modele et metriques
