import streamlit as st
import pandas as pd

st.title("Heart Disease ML Assignment Demo")

# ✅ Use the correct raw GitHub link
DATA_URL = "https://raw.githubusercontent.com/vanavaimani1984/heart-disease-ml-assignment/main/test_data.csv"

# Add error handling so the app doesn't crash if the link fails
try:
    df = pd.read_csv(DATA_URL)
    st.subheader("Test Dataset Preview")
    st.write(df.head())
except Exception as e:
    st.error(f"Failed to load dataset from GitHub: {e}")
    st.stop()

st.subheader("Model Selection (Demo Only)")
model_choice = st.selectbox(
    "Select a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

st.info(f"You selected: {model_choice}. Model integration will be added later.")
