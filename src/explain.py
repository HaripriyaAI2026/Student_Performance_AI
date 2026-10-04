import joblib
import pandas as pd
from sklearn.inspection import permutation_importance


MODEL_PATH = "models/best_regression_model.pkl"
DATA_PATH = "data/processed/student_performance_clean.csv"


# Load model and dataset
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Prepare data
X = df.drop(columns=["G3", "G1", "G2"])
y = df["G3"]

print("Model and dataset loaded successfully!")


# Calculate permutation importance
result = permutation_importance(
    model,
    X,
    y,
    n_repeats=5,
    random_state=42,
    scoring="neg_mean_absolute_error"
)


# Create importance table
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print("\nTop Important Features")
print("-" * 40)

print(
    importance_df.head(10).to_string(index=False)
)