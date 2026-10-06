import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.model import FEATURES

MODEL = Path("models/model.joblib")
METRICS = Path("models/metrics.json")

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="modele non entraine (python -m src.model)"
)


def sample_row(**over):
    row = {
        "neighbourhood_cleansed": "Buttes-Montmartre",
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "accommodates": 2, "bedrooms": 1.0, "beds": 1.0, "bathrooms": 1.0,
        "number_of_reviews": 10, "review_scores_rating": 4.7,
        "availability_365": 150, "minimum_nights": 2,
        "latitude": 48.886, "longitude": 2.343,
    }
    row.update(over)
    return pd.DataFrame([row])[FEATURES]


def test_prediction_is_plausible():
    price = float(joblib.load(MODEL).predict(sample_row())[0])
    assert 20 < price < 1000


def test_prediction_handles_missing_rating():
    X = sample_row(number_of_reviews=0, review_scores_rating=np.nan)
    assert np.isfinite(joblib.load(MODEL).predict(X)[0])


def test_larger_home_costs_more():
    model = joblib.load(MODEL)
    small = model.predict(sample_row())[0]
    big = model.predict(sample_row(accommodates=6, bedrooms=3.0, beds=4.0))[0]
    assert big > small


def test_metrics_file_is_consistent():
    m = json.loads(METRICS.read_text(encoding="ascii"))
    assert m["test"]["mae"] < m["baseline_test"]["mae"]
    assert m["ratio_q10"] < 1 < m["ratio_q90"]
