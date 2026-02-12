import pandas as pd
import os

def create_final_dataset():
    # Paths
    raw_path = "data/raw/raw_data.csv"
    processed_dir = "data/processed"
    final_path = "data/processed/final_dataset.csv"

    # Create processed folder if not exists
    os.makedirs(processed_dir, exist_ok=True)

    # Check if raw file exists
    if not os.path.exists(raw_path):
        print("Error: Raw dataset not found!")
        return

    # Load raw dataset
    df = pd.read_csv(raw_path)
    print("Raw dataset loaded")
    print("Initial shape:", df.shape)

    # ---------------- PHASE 2 PROCESSING ----------------

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    # Clean text columns
    text_columns = df.select_dtypes(include="object").columns
    for col in text_columns:
        df[col] = df[col].str.strip().str.lower()

    # ----------------------------------------------------

    # Check if dataframe is empty
    if df.empty:
        print("Error: Final dataset is empty after processing!")
        return

    # Save final dataset
    df.to_csv(final_path, index=False)

    print("Final dataset created successfully!")
    print("Final shape:", df.shape)
    print("Saved at:", final_path)
