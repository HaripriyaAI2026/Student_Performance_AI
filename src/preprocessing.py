import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "student-mat.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "student_performance_clean.csv"


def load_data():
    """Load the raw student performance dataset."""
    df = pd.read_csv(RAW_FILE, sep=";")
    return df


def clean_data(df):
    """Clean and validate the dataset."""
    
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Check missing values
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype == "object":
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].median())

    return df


def save_processed_data(df):
    """Save cleaned data to the processed folder."""
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)


if __name__ == "__main__":
    data = load_data()
    cleaned_data = clean_data(data)
    save_processed_data(cleaned_data)

    print("Data preprocessing completed successfully!")
    print("Rows:", cleaned_data.shape[0])
    print("Columns:", cleaned_data.shape[1])
    print("Saved to:", PROCESSED_FILE)