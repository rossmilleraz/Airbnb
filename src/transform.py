"""
Airbnb listings data transformation module.

This script takes the cleaned dataset produced by clean.py and performs
feature engineering, outlier handling, encoding, and preparation for
downstream modeling. The goal is to create a model-ready dataset focused
on predicting Airbnb review scores and understanding quality drivers.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.extract import load_csv


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add host tenure and review recency features."""
    today = pd.Timestamp("today").normalize()

    if "host_since" in df.columns:
        df["host_tenure_days"] = (today - df["host_since"]).dt.days

    if "last_review" in df.columns:
        df["days_since_last_review"] = (today - df["last_review"]).dt.days

    # Month / Year components
    if "last_review" in df.columns:
        df["last_review_year"] = df["last_review"].dt.year
        df["last_review_month"] = df["last_review"].dt.month

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add listing-level ratio features like beds per guest."""
    if "bedrooms" in df.columns and "accommodates" in df.columns:
        df["beds_per_guest"] = df["bedrooms"] / df["accommodates"]

    if "reviews_per_month" in df.columns and "accommodates" in df.columns:
        df["reviews_pm_per_guest"] = df["reviews_per_month"] / df["accommodates"]

    if "availability_365" in df.columns:
        df["occupancy_ratio"] = 1 - (df["availability_365"] / 365)

    return df


def add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine review metrics into aggregate scores."""
    score_cols = [
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value"
    ]

    existing_scores = [col for col in score_cols if col in df.columns]

    if existing_scores:
        df["review_score_avg"] = df[existing_scores].mean(axis=1)
        df["review_score_std"] = df[existing_scores].std(axis=1)

    return df


def handle_outliers(df: pd.DataFrame, cols: list[str], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Clip outliers using thresholds."""
    for col in cols:
        if col in df.columns:
            low = df[col].quantile(lower_q)
            high = df[col].quantile(upper_q)
            df[col] = df[col].clip(low, high)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables for modeling."""
    # One-hot encode low-cardinality categoricals
    onehot_cols = ["room_type", "property_type"]
    for col in onehot_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Frequency encode neighborhoods
    if "neighbourhood_cleansed" in df.columns:
        freq = df["neighbourhood_cleansed"].value_counts(normalize=True)
        df["neighbourhood_freq"] = df["neighbourhood_cleansed"].map(freq)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove irrelevant columns for modeling."""
    drop_cols = [
        "id", "listing_url", "host_url", "host_thumbnail_url",
        "host_picture_url", "picture_url", "license"
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def transform_airbnb_data() -> pd.DataFrame:
    """
    Transform the cleaned Airbnb dataset into a fully model-ready dataset.

    Steps:
        1. Load cleaned dataset
        2. Add features
        3. Handle outliers
        4. Encode categoricals
        5. Drop unused columns
    """

    # Load cleaned data
    df = load_csv("listings_cleaned.csv", folder="processed")

    date_cols = ["host_since", "last_review", "first_review"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # feature engineering
    df = add_temporal_features(df)
    df = add_ratio_features(df)
    df = add_review_features(df)

    #  Outlier Handling
    numeric_cols_for_outliers = [
        "accommodates", "minimum_nights", "maximum_nights",
        "availability_365", "number_of_reviews", "reviews_per_month",
        "bathrooms", "bedrooms", "host_response_rate", "host_acceptance_rate",
        "host_listings_count", "host_total_listings_count"
    ]
    df = handle_outliers(df, numeric_cols_for_outliers)

    df = encode_categoricals(df)

    # Drop unused columns
    df = drop_unused_columns(df)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "processed" / "listings_transformed.csv"

    transformed_df = transform_airbnb_data()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_df.to_csv(output_path, index=False)

    print(f"\nTransformed dataset saved to: {output_path}")
