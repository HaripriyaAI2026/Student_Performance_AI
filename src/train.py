import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_PATH = "data/processed/student_performance_clean.csv"
MODEL_PATH = "models/best_regression_model.pkl"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# -----------------------------
# 2. Prepare features and target
# -----------------------------
X = df.drop(columns=["G3", "G1", "G2"])
y = df["G3"]


# -----------------------------
# 3. Identify feature types
# -----------------------------
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()


# -----------------------------
# 4. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# -----------------------------
# 5. Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------
# 6. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------
# 7. Train model
# -----------------------------
pipeline.fit(X_train, y_train)


# -----------------------------
# 8. Evaluate
# -----------------------------
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-" * 40)
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# -----------------------------
# 9. Save model
# -----------------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved successfully!")
print(MODEL_PATH)
