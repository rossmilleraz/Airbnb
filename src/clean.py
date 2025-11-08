"""
Airbnb listings data cleaning module.

This script loads raw Airbnb listing data, performs column cleaning,
type conversions, and feature engineering to produce a standardized
clean dataset for downstream analysis or modeling.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from src.extract import load_csv


# ------------------------------------------------------------
# Core cleaning functions
# ------------------------------------------------------------

def drop_missing_columns(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """
    Drop columns with more than a given fraction of missing data.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Fraction above which columns are dropped (default = 0.4).

    Returns:
        pd.DataFrame: DataFrame with high-missing columns removed.
    """
    missing_fraction = df.isna().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index

    if len(cols_to_drop) > 0:
        print(f"Dropping {len(cols_to_drop)} columns (>{threshold * 100:.0f}% missing):")
        for col in cols_to_drop:
            print(f"  - {col}")
    else:
        print("No columns exceeded missing threshold.")

    return df.drop(columns=cols_to_drop, errors="ignore")


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text columns by lowercasing, trimming spaces,
    and replacing empty strings with NaN.
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({'': np.nan, 'nan': np.nan})
        )
    return df


def clean_bathroom_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric bathroom counts and create a boolean shared-bath column.
    Handles both datasets with 'bathrooms_text' and numeric-only 'bathrooms'.
    """
    if "bathrooms_text" in df.columns:
        df["shared_bath"] = df["bathrooms_text"].str.contains("shared", case=False, na=False)
        df["bathrooms_text"] = df["bathrooms_text"].str.replace("half-bath", "0.5", case=False, regex=True)
        df["bathrooms"] = (
            df["bathrooms_text"]
            .str.extract(r"(\d*\.?\d+)")[0]
            .astype(float)
        )
        df["bathrooms"] = df["bathrooms"].fillna(0)
        df.drop(columns=["bathrooms_text"], inplace=True, errors="ignore")
    else:
        print("No bathrooms_text column found — using numeric bathrooms only.")
        if "shared_bath" not in df.columns:
            df["shared_bath"] = np.nan

    return df


def convert_percent_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert percentage strings (e.g., '85%') into decimal numeric format (e.g., 0.85).
    """
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
    """
    Convert columns to proper pandas data types for consistency.
    """
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


# ------------------------------------------------------------
# Tracking and column management
# ------------------------------------------------------------

def keep_selected_columns(
    df: pd.DataFrame,
    date_cols: list[str],
    numeric_cols: list[str],
    bool_cols: list[str],
    str_cols: list[str]
) -> pd.DataFrame:
    """
    Keep only selected columns and log which ones were dropped.
    """
    keep_cols = date_cols + numeric_cols + bool_cols + str_cols
    existing_cols = [col for col in keep_cols if col in df.columns]
    removed_cols = [col for col in df.columns if col not in existing_cols]

    print(f"\n🧹 Keeping {len(existing_cols)} columns:")
    for col in existing_cols:
        print(f"  - {col}")

    print(f"\n🗑️ Dropping {len(removed_cols)} columns not in keep lists:")
    for col in removed_cols:
        print(f"  - {col}")

    # Save dropped columns for recordkeeping
    dropped_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "dropped_columns.txt"
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dropped_path, "w") as f:
        for col in removed_cols:
            f.write(col + "\n")

    return df[existing_cols]


# ------------------------------------------------------------
# Full cleaning pipeline
# ------------------------------------------------------------

def clean_airbnb_data() -> pd.DataFrame:
    """
    Full Airbnb data cleaning pipeline.

    Steps:
        1. Load CSV
        2. Drop high-missing columns
        3. Normalize text
        4. Clean bathroom fields
        5. Convert percentage columns
        6. Apply dtype conversions
        7. Keep only relevant columns
    """
    df = load_csv('listings.csv', 'raw')
    df = drop_missing_columns(df)
    df = normalize_text_columns(df)
    df = clean_bathroom_info(df)
    df = convert_percent_columns(df, ["host_response_rate", "host_acceptance_rate"])

    # Define column type groups
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
    df = keep_selected_columns(df, date_cols, numeric_cols, bool_cols, str_cols)

    return df


# ------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    listings_file = project_root / "data" / "raw" / "listings.csv"

    # Run cleaning
    cleaned_df = clean_airbnb_data()

    # Save cleaned data
    output_path = project_root / "data" / "processed" / "listings_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"\n✅ Cleaned data saved to: {output_path}")
