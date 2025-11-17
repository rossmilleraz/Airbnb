from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from src.extract import load_csv


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds tenure, recency, and temporal components."""
    today = pd.Timestamp("today").normalize()

    # Host tenure
    if "host_since" in df.columns:
        df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_tenure_days"] = (today - df["host_since"]).dt.days
        df["host_tenure_years"] = df["host_tenure_days"] / 365

        # Experience bins
        df["host_experience_level"] = pd.cut(
            df["host_tenure_years"],
            bins=[-1, 1, 3, 5, 10, np.inf],
            labels=["new", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"]
        )

    # Review recency
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
        df["days_since_last_review"] = (today - df["last_review"]).dt.days
        df["last_review_year"] = df["last_review"].dt.year
        df["last_review_month"] = df["last_review"].dt.month

        # Review activity bucket
        df["review_activity_level"] = pd.cut(
            df["days_since_last_review"],
            bins=[-1, 30, 90, 180, 365, np.inf],
            labels=["< 1 month", "< 3 months", "< 6 months", "< 1 year", "1+ years"]
        )

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ratios such as bed/bath consistency and occupancy."""

    # Bed/Bath ratio
    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"]
        df["bed_bath_ratio"] = df["bed_bath_ratio"].replace([np.inf, -np.inf], np.nan)

    # Beds per guest
    if "bedrooms" in df.columns and "accommodates" in df.columns:
        df["beds_per_guest"] = df["bedrooms"] / df["accommodates"]

    # Reviews per guest
    if "reviews_per_month" in df.columns and "accommodates" in df.columns:
        df["reviews_pm_per_guest"] = df["reviews_per_month"] / df["accommodates"]

    # Occupancy ratio
    if "availability_365" in df.columns:
        df["occupancy_ratio"] = 1 - (df["availability_365"] / 365)

    return df


def add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds predictive features by combining review metrics"""

    review_cols = [
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value"
    ]

    existing = [c for c in review_cols if c in df.columns]

    if existing:
        df["review_score_avg"] = df[existing].mean(axis=1)
        df["review_score_std"] = df[existing].std(axis=1)

    return df


def handle_outliers(df: pd.DataFrame, columns: list[str], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Remove extreme outliers (.01 / .99)"""
    for col in columns:
        if col in df.columns:
            low = df[col].quantile(lower_q)
            high = df[col].quantile(upper_q)
            df[col] = df[col].clip(lower=low, upper=high)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One hot encodes small categories and frequency-encodes neighborhoods."""

    # One hot for low-cardinality categories
    onehot_cols = ["room_type", "property_type"]
    for col in onehot_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

    # Neighborhood frequency encoding (powerful + lightweight)
    if "neighbourhood_cleansed" in df.columns:
        freq = df["neighbourhood_cleansed"].value_counts(normalize=True)
        df["neighbourhood_freq"] = df["neighbourhood_cleansed"].map(freq)

    return df


def add_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds quantile based latitude/longitude bins for neighborhood grouping."""

    if "latitude" in df.columns:
        df["lat_bin"] = pd.qcut(df["latitude"], q=5, duplicates="drop").astype("category")

    if "longitude" in df.columns:
        df["lon_bin"] = pd.qcut(df["longitude"], q=5, duplicates="drop").astype("category")

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes irrelevant identifiers and media URLs."""

    drop_cols = [
        "id", "listing_url", "host_url", "host_thumbnail_url",
        "host_picture_url", "picture_url", "license", "calendar_updated",
        "first_review"  # replaced by engineered features
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def transform_airbnb_data() -> pd.DataFrame:
    """Full feature engineering pipeline."""

    df = load_csv("listings_cleaned.csv", folder="processed")

    # Convert key dates
    date_cols = ["host_since", "last_review", "first_review"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Feature engineering
    df = add_temporal_features(df)
    df = add_ratio_features(df)
    df = add_review_features(df)
    df = add_geographic_features(df)

    # Outliers
    outlier_cols = [
        "accommodates", "minimum_nights", "maximum_nights", "availability_365",
        "number_of_reviews", "reviews_per_month", "bathrooms", "bedrooms",
        "host_response_rate", "host_acceptance_rate",
        "host_listings_count", "host_total_listings_count"
    ]
    df = handle_outliers(df, outlier_cols)

    # Encoding
    df = encode_categoricals(df)

    # Drop irrelevant columns
    df = drop_unused_columns(df)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "processed" / "listings_transformed.csv"

    transformed_df = transform_airbnb_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_df.to_csv(output_path, index=False)
    print('Saved transformed data to:', output_path)