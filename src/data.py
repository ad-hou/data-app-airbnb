"""Chargement et nettoyage des donnees Inside Airbnb (Paris)."""
import pandas as pd

TARGET = "price"
PRICE_MIN = 20
PRICE_MAX = 1000
MAX_MIN_NIGHTS = 30

CATEGORICAL_FEATURES = ["neighbourhood_cleansed", "room_type", "property_type"]
NUMERIC_FEATURES = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "number_of_reviews", "review_scores_rating",
    "availability_365", "minimum_nights",
    "latitude", "longitude",
]
ALL_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]


def parse_price(s):
    """Convertit '$1,234.50' en 1234.5. Valeur illisible ou absente -> NaN."""
    cleaned = (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_listings(df):
    """Applique les regles de nettoyage. Les NaN des variables sont gardes
    (ils sont imputes plus tard dans le pipeline, sur l'entrainement seul)."""
    out = df.copy()
    out[TARGET] = parse_price(out[TARGET])
    out = out.dropna(subset=[TARGET])
    out = out[out[TARGET].between(PRICE_MIN, PRICE_MAX)]
    out = out[out["minimum_nights"] <= MAX_MIN_NIGHTS]
    return out[ALL_COLUMNS].reset_index(drop=True)


def load_listings(path):
    """Lit le CSV (gzip accepte) et retourne les donnees nettoyees."""
    raw = pd.read_csv(path, usecols=ALL_COLUMNS)
    return clean_listings(raw)
