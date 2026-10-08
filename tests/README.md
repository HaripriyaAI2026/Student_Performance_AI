# Intelligent Student Performance Prediction and Personalized Learning Recommendation System

## Project Overview

This project is an AI/ML-based system that predicts student academic performance and provides personalized learning recommendations.

The system uses student academic, demographic, attendance, study-habit and lifestyle-related features to predict the final performance score and identify the level of support required.

## Objectives

- Predict student final academic performance
- Identify students who may need additional academic support
- Compare multiple machine learning algorithms
- Provide model explainability using feature importance
- Generate personalized learning recommendations
- Provide an interactive Streamlit web application
- Maintain reproducible and testable ML workflows

## Machine Learning Models

### Regression Models

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. K-Nearest Neighbors Regressor

### Classification Models

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. K-Nearest Neighbors Classifier

## Dataset

The project uses the UCI Student Performance dataset.

The prediction target is:

- `G3` - Final student performance score

`G1` and `G2` are excluded from the prediction features to reduce target leakage for the intended prediction scenario.

## Project Structure

```text
Student_Performance_AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_regression_model.pkl
│   ├── best_classification_model.pkl
│   └── feature_importance.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── explain.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── recommendation.py
│   └── train.py
│
├── tests/
│   └── test_project.py
│
├── requirements.txt
├── requirements.full.txt
└── README.md