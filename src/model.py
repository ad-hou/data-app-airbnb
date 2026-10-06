"""Entrainement, comparaison et sauvegarde du modele de prix."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_listings

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
DATA_PATH = "data/raw/listings.csv.gz"
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BASELINE = "Baseline (mediane)"


def build_model(estimator):
    """Pipeline complet : pretraitement + modele, cible en log."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=30,
        sparse_output=False,
    )
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("prep", prep), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)


def candidates():
    return {
        BASELINE: DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = load_listings(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Entrainement : {len(X_train)} lignes | Test : {len(X_test)} lignes\n")

    # 1) Comparaison par validation croisee, sur l'entrainement uniquement
    cv_mae = {}
    for name, est in candidates().items():
        scores = cross_validate(
            build_model(est), X_train, y_train,
            cv=CV_FOLDS, scoring="neg_mean_absolute_error",
        )
        cv_mae[name] = float(-scores["test_score"].mean())
        print(f"{name:<22} MAE (validation croisee) : {cv_mae[name]:7.2f} EUR")

    best = min((n for n in cv_mae if n != BASELINE), key=cv_mae.get)
    print(f"\nMeilleur modele : {best}\n")

    # 2) Une seule evaluation sur le jeu de test
    baseline_model = build_model(candidates()[BASELINE]).fit(X_train, y_train)
    baseline_test = evaluate(y_test, baseline_model.predict(X_test))

    model = build_model(candidates()[best]).fit(X_train, y_train)
    pred = model.predict(X_test)
    test = evaluate(y_test, pred)
    ratio = (y_test.to_numpy() / pred)
    q10, q90 = np.percentile(ratio, [10, 90])

    print(f"TEST - {best}")
    print(f"  MAE  : {test['mae']:.2f} EUR")
    print(f"  RMSE : {test['rmse']:.2f} EUR")
    print(f"  R2   : {test['r2']:.3f}")
    print(f"TEST - baseline (MAE) : {baseline_test['mae']:.2f} EUR")
    print(f"Intervalle 80 % : prediction x {q10:.2f} a x {q90:.2f}")

    # 3) Sauvegarde du modele et des mesures
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    metrics = {
        "best_model": best,
        "n_rows_clean": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(FEATURES),
        "models_compared": [n for n in cv_mae if n != BASELINE],
        "cv_mae": cv_mae,
        "test": test,
        "baseline_test": baseline_test,
        "ratio_q10": float(q10),
        "ratio_q90": float(q90),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="ascii")
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nModele sauvegarde : {MODEL_PATH} ({size_mb:.1f} Mo)")


if __name__ == "__main__":
    main()
