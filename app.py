import streamlit as st
import pandas as pd

# Title
st.title("Heart Disease ML Assignment Demo")

# Load test dataset
DATA_URL = "https://raw.githubusercontent.com/vanavaimani1984/Machine_learning/main/test_data.csv"
df = pd.read_csv(DATA_URL)

# Show dataset preview
st.subheader("Test Dataset Preview")
st.write(df.head())

# Placeholder for model selection
st.subheader("Model Selection (Demo Only)")
model_choice = st.selectbox(
    "Select a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

st.info(f"You selected: {model_choice}. Model integration will be added later.")
