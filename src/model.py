"""
Loads the transformed dataset, trains a regression model, evaluates performance,
and saves the model.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib


def load_transformed() -> pd.DataFrame:
    """
    Load the transformed data
    """
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data" / "processed" / "listings_transformed.csv"
    return pd.read_csv(path, low_memory=False)


def train_model(df: pd.DataFrame):
    """
    Train a Random Forest regression model to predict review_score_avg.
    """

    target = "review_score_avg"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in transformed dataset.")

    # Drop rows without a target
    df = df.dropna(subset=[target]).copy()

    # Feature matrix
    X = df.drop(columns=[target])

    # make sure all features are numeric
    X = pd.get_dummies(X, drop_first=True)
    X = X.select_dtypes(include=[np.number]).fillna(0)

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Eval
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return model, metrics, X_train.columns


def save_model(model, feature_names):
    """Save the trained model and its feature set."""
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "models"
    out_dir.mkdir(exist_ok=True)

    model_path = out_dir / "random_forest_review_model.pkl"
    features_path = out_dir / "feature_columns.json"

    joblib.dump(model, model_path)
    pd.Series(feature_names).to_json(features_path, orient="values")

    print(f"Model saved to: {model_path}")
    print(f"Feature list saved to: {features_path}")


if __name__ == "__main__":
    df = load_transformed()
    model, metrics, features = train_model(df)

    print("\nModel Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    save_model(model, features)
