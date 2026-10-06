import pandas as pd

from src.data import ALL_COLUMNS, PRICE_MAX, PRICE_MIN, clean_listings, parse_price


def make_df(prices, min_nights):
    n = len(prices)
    return pd.DataFrame({
        "neighbourhood_cleansed": ["Louvre"] * n,
        "room_type": ["Entire home/apt"] * n,
        "property_type": ["Entire rental unit"] * n,
        "accommodates": [2] * n,
        "bedrooms": [1.0] * n,
        "beds": [1.0] * n,
        "bathrooms": [1.0] * n,
        "number_of_reviews": [0] * n,
        "review_scores_rating": [None] * n,
        "availability_365": [100] * n,
        "minimum_nights": min_nights,
        "latitude": [48.85] * n,
        "longitude": [2.35] * n,
        "price": prices,
    })


def test_parse_price_handles_dollar_comma_and_missing():
    s = pd.Series(["$1,234.50", "$20.00", None])
    out = parse_price(s)
    assert out.iloc[0] == 1234.5
    assert out.iloc[1] == 20.0
    assert pd.isna(out.iloc[2])


def test_clean_drops_missing_and_out_of_range_prices():
    df = make_df(
        ["$100.00", None, "$5.00", "$5,000.00", "$20.00", "$1,000.00"],
        [2, 2, 2, 2, 2, 2],
    )
    out = clean_listings(df)
    assert len(out) == 3
    assert out["price"].between(PRICE_MIN, PRICE_MAX).all()


def test_clean_drops_long_stays():
    df = make_df(["$100.00", "$100.00", "$100.00"], [2, 30, 31])
    out = clean_listings(df)
    assert len(out) == 2
    assert out["minimum_nights"].max() == 30


def test_clean_keeps_expected_columns_and_missing_features():
    out = clean_listings(make_df(["$100.00"], [2]))
    assert list(out.columns) == ALL_COLUMNS
    assert out["review_scores_rating"].isna().all()
