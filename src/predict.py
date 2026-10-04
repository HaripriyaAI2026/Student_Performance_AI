import joblib
import pandas as pd


MODEL_PATH = "models/best_regression_model.pkl"


# Load trained model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# Load dataset only to get a valid sample student
data_path = "data/processed/student_performance_clean.csv"
df = pd.read_csv(data_path)

# Remove target and leakage columns
student_data = df.drop(
    columns=["G3", "G1", "G2"]
).iloc[[0]]


# Make prediction
prediction = model.predict(student_data)[0]

print("\nStudent Performance Prediction")
print("-" * 40)
print(f"Predicted Final Score (G3): {prediction:.2f}")