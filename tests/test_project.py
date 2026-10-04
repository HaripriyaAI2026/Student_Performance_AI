import os
import joblib
import pandas as pd


def test_dataset_exists():
    path = "data/processed/student_performance_clean.csv"
    assert os.path.exists(path)


def test_dataset_loads():
    path = "data/processed/student_performance_clean.csv"

    df = pd.read_csv(path)

    assert len(df) > 0
    assert "G3" in df.columns


def test_regression_model_exists():
    path = "models/best_regression_model.pkl"

    assert os.path.exists(path)


def test_classification_model_exists():
    path = "models/best_classification_model.pkl"

    assert os.path.exists(path)


def test_regression_model_loads():
    path = "models/best_regression_model.pkl"

    model = joblib.load(path)

    assert model is not None


def test_classification_model_loads():
    path = "models/best_classification_model.pkl"

    model = joblib.load(path)

    assert model is not None