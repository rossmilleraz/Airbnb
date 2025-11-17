"""
Run the full Airbnb Review Quality Prediction pipeline.

1. Cleaning raw InsideAirbnb data
2. Transforming the cleaned dataset with feature engineering
3. Training the machine learning model
4. Saving the model and reporting metrics

"""

from pathlib import Path
from src.clean import clean_airbnb_data
from src.transform import transform_airbnb_data
from src.model import train_model, save_model


def main():
    print("\n==============================")
    print(" AIRBNB PIPELINE: STARTING")
    print("==============================\n")

    # 1. CLEAN DATA
    print("Step 1: Cleaning raw dataset...")
    cleaned_df = clean_airbnb_data()
    print(f"Cleaned dataset created. Shape: {cleaned_df.shape}\n")

    # 2. TRANSFORM DATA
    print("Step 2: Transforming cleaned data...")
    transformed_df = transform_airbnb_data()
    print(f"Transformed dataset created. Shape: {transformed_df.shape}\n")

    # 3. TRAIN MODEL
    print("Step 3: Training model...")
    model, metrics, features = train_model(transformed_df)
    print("Model trained successfully.\n")

    # 4. SAVE OUTPUTS
    print("Step 4: Saving model and metadata...")
    save_model(model, features)
    print("Model and feature list saved.\n")

    # 5. PRINT METRICS)
    print(" PIPELINE: COMPLETE\n")

    print("Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nAll steps finished successfully.")


if __name__ == "__main__":
    main()
