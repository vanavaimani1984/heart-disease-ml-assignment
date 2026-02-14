import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Dropdown for model selection
model_choice = st.selectbox(
    "Select a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Map model names to saved files
model_map = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# Load the selected model
model_file = model_map.get(model_choice)
model = joblib.load(model_file)

# Collect all 13 inputs
age = st.number_input("Age", 0, 120, 50)
sex = st.selectbox("Sex (0=female, 1=male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (0/1)", [0, 1])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])
oldpeak = st.number_input("Oldpeak", -2.0, 6.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0=normal, 1=fixed defect, 2=reversible defect, 3=other)", [0, 1, 2, 3])

# Build input vector
input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal]]

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")

    # Probability output (if supported by model)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of Heart Disease: {prob:.2f}")

    # Accuracy display (requires test set)
    # Load your test data (replace with actual test set path or variable)
    try:
        df = pd.read_csv("test_data.csv")  # ensure this file exists
        X_test = df.drop("target", axis=1)
        y_test = df["target"]

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy on Test Data: {acc:.2f}")
    except Exception as e:
        st.write("Accuracy could not be calculated. Ensure test_data.csv is available.")
