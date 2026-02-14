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
- Quick download link: [test_data.csv](https://raw.githubusercontent.com/vanavaimani1984/Machine_learning/main/test_data.csv)


## Results Summary
- Best models: Random Forest & XGBoost (~99% accuracy, AUC ~1.0).
- Logistic Regression: ~85% CV accuracy, balanced baseline.
- Naive Bayes: ~82% CV accuracy, strong recall.
- KNN: ~75% CV accuracy, weaker performance.
- Decision Tree: Overfitting (perfect CV accuracy).

## Execution
- Run the notebook in **BITS Virtual Lab** for reproducibility.
- Use the test dataset from GitHub for validation.

## Streamlit Demo (Optional)
- Run `streamlit run app.py` to launch the interactive demo.
- Demo includes dataset preview, model selection, metrics, and confusion matrix.
