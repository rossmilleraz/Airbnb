import pandas as pd
from src.extract import load_listings_csv  # import extract function


def clean_listings(file_path="data/raw/listings.csv"):
    """
    Loads raw listing data and cleans it
    """
    # Get raw data
    df = load_listings_csv(file_path)

    if df is None:
        return None

    # Clean the data
    df = df.drop_duplicates()
    df = df[df['price'] > 0]  # remove zero/negative prices
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)  # replace missing values with 0
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')  # convert to datetime, convert to Nat

    # Feature Engineering
    df['price_per_bed'] = df['price'] / df['beds']

    return df


def save_cleaned_data():
    '''
    Saves cleaned data to csv file.
    '''
    pass


def featured_data():
    '''
    Loads cleaned data and returns neccessary data
    '''
    pass
