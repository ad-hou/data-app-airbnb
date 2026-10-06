"""Genere le jeu de donnees nettoye et leger utilise par l'application."""
from src.data import load_listings

RAW_PATH = "data/raw/listings.csv.gz"
OUT_PATH = "data/listings_clean.parquet"

df = load_listings(RAW_PATH)
df.to_parquet(OUT_PATH, index=False, compression="gzip")
print(df.shape, "->", OUT_PATH)
