import pandas as pd
from pathlib import Path


def load_csv(filename: str, folder: str = "raw"):
    """
    Load a CSV file from the project's data directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    file_path = project_root / "data" / folder / filename

    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded {len(df):,} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
