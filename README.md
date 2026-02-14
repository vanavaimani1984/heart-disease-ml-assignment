# Heart Disease ML Assignment

## Dataset
- Source: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Only the **test dataset** (`test_data.csv`) is uploaded here, as per assignment requirement.

## Notebook
- The notebook (`Machine_learning.ipynb`) contains:
  - Preprocessing
  - Model training (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost)
  - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  - Confusion matrices
  - Feature importance
  - Analysis write‑up


## Test Dataset
- Quick download link: [test_data.csv](https://raw.githubusercontent.com/vanavaimani1984/heart-disease-ml-assignment/main/test_data.csv)


## Execution
- Run the notebook in **BITS Virtual Lab** for reproducibility.
- Use the test dataset from GitHub for validation.

## 📂 Repository Structure
heart-disease-ml-assignment/

└── models/                 # Trained ML models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl


## 🧪 Results Summary
- **Best models**: Random Forest & XGBoost (~99% accuracy, AUC ~1.0)  
- Logistic Regression: ~85% CV accuracy, balanced baseline  
- Naive Bayes: ~82% CV accuracy, strong recall  
- KNN: ~75% CV accuracy, weaker performance  
- Decision Tree: Overfitting (perfect CV accuracy)  

## 🌐 Streamlit Demo
The app is deployed on **Streamlit Cloud**:  
👉 Live App Link: [Heart Disease Prediction App](https://heart-disease-ml-assignment-meatzvtytsbuydbamvdt6h.streamlit.app/)







