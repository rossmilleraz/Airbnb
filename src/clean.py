# TODO Add more comments, fix file path so uploading to git actually works

"""
Airbnb listings data cleaning module.

This script loads raw Airbnb listing data, performs column cleaning,
type conversions, and some feature engineering.


"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path(r"C:\Users\Ross\Desktop\Data Analytics Proj\AirbnbProject\data\raw\listings.csv")


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the Airbnb listings CSV file into a DataFrame."""
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df):,} rows from {file_path}")
    return df


def drop_missing_columns(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """
    Drop columns with more than 40% missing data

    Returns
    pd.DataFrame with high-missing columns removed.
    """
    # Get percent of missing values and then drop columns above that percent
    missing_fraction = df.isna().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index  # drop columns

    if len(cols_to_drop) > 0:
        print(f"Dropping {len(cols_to_drop)} columns (>{threshold * 100:.0f}% missing):")
        for col in cols_to_drop:
            print(f"  - {col}")

    return df.drop(columns=cols_to_drop)


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, trim spaces, and replace empty strings with NaN in all object/string columns."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .str.lower()
            .str.strip()
            .replace("", np.nan)
        )
    return df


def clean_bathroom_info(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric bathroom counts and create a boolean shared-bath column."""
    if "bathrooms_text" not in df.columns:
        return df

    df["shared_bath"] = df["bathrooms_text"].str.contains("shared", case=False, na=False)
    df["bathrooms_text"] = df["bathrooms_text"].str.replace("half-bath", "0.5", case=False, regex=True)
    df["bathrooms"] = (
        df["bathrooms_text"]
        # Use regex to extract numeric bathroom counts from text and convert to float
        .str.extract(r"(\d*\.?\d+)")[0]
        .astype(float)
    )
    df["bathrooms"] = df["bathrooms"].fillna(0)
    df.drop(columns=["bathrooms_text"], inplace=True, errors="ignore")
    return df


def convert_percent_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert percentage strings (e.g., '85%') into decimal numeric format (e.g., 0.85)."""
    for col in cols:
        if col in df.columns:
            df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .replace("nan", np.nan)
                    .astype(float) / 100
            )
    return df


def convert_column_types(
        df: pd.DataFrame,
        date_cols: list[str],
        numeric_cols: list[str],
        bool_cols: list[str],
        str_cols: list[str],
) -> pd.DataFrame:
    """Convert columns to proper pandas data types."""
    # Datetime
    df[date_cols] = df[date_cols].apply(pd.to_datetime, errors="coerce")

    # Numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "t": True, "true": True, "yes": True,
                    "f": False, "false": False, "no": False,
                    "verified": True, "unverified": False,
                })
            ).astype("boolean")

    # Strings
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def clean_airbnb_data(file_path: str | Path) -> pd.DataFrame:
    """
    Full Airbnb data cleaning pipeline.

    Steps:
        1. Load CSV
        2. Drop high-missing columns
        3. Normalize text
        4. Clean bathroom fields
        5. Convert percentage columns
        6. Apply dtype conversions
    """
    df = load_data(file_path)
    df = drop_missing_columns(df)
    df = normalize_text_columns(df)
    df = clean_bathroom_info(df)
    df = convert_percent_columns(df, ["host_response_rate", "host_acceptance_rate"])

    # Define column type groups, so I can easily add / remove later if needed
    date_cols = ["host_since", "last_review", "first_review"]
    numeric_cols = [
        "accommodates", "minimum_nights", "maximum_nights",
        "availability_30", "availability_60", "availability_90", "availability_365",
        "number_of_reviews", "reviews_per_month",
        "review_scores_rating", "review_scores_cleanliness",
        "review_scores_communication", "review_scores_location",
        "review_scores_value", "host_listings_count",
        "host_total_listings_count", "latitude", "longitude",
        "bathrooms", "bedrooms", "host_response_rate", "host_acceptance_rate"
    ]
    bool_cols = ["host_is_superhost", "host_identity_verified", "shared_bath"]
    str_cols = ["host_response_time", "neighbourhood_cleansed", "property_type", "room_type"]

    df = convert_column_types(df, date_cols, numeric_cols, bool_cols, str_cols)
    return df


if __name__ == "__main__":
    cleaned_df = clean_airbnb_data(RAW_DATA_PATH)

    # Optional: save cleaned data
    output_path = RAW_DATA_PATH.parent.parent / "processed" / "listings_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
