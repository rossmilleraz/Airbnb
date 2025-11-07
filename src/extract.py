import pandas as pd

def load_listings_csv(file_path = r"C:\Users\Ross\Desktop\Data Analytics Proj\AirbnbProject\data\raw\listings.csv"):

    """
    Reads the Airbnb listings CSV from local storage and returns a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

