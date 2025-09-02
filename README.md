# Churn Prediction Model

This project trains machine learning models to predict customer churn using a telecom dataset. 
I implemented the data preprocessing, model training pipeline, and evaluation. Along the way, I debugged 
several issues related to data types, missing values, and model leakage. I also used an AI-assisted tool 
(TestSprite) to help identify potential problems, but I validated and refined each fix myself based on my 
understanding of machine learning best practices.

---

## 🔧 Debugging & Fixes

### 1. Data Type Mismatch
- **Issue**: The `Churn` column contained string values ("Yes"/"No") instead of integers (0/1).
- **Fix**: Converted the column into binary integers: `1` for "Yes", `0` for "No".
- **Knowledge**: Models require numeric targets. String targets cause type conversion errors. I knew the fix 
  had to involve mapping strings to numeric values.

### 2. Deprecated Parameter in Scikit-learn
- **Issue**: `OneHotEncoder` no longer supports the `sparse=True` parameter in newer versions of scikit-learn.
- **Fix**: Updated to `sparse_output=True`.
- **Knowledge**: I keep track of scikit-learn’s API changes. The error message confirmed the issue, and I 
  verified the correct parameter in the documentation.

### 3. Missing Value Handling
- **Issue**: LogisticRegression failed due to `NaN` values in the dataset.
- **Fix**: Added imputers: median imputation for numeric features, constant “missing” category for categorical features.
- **Knowledge**: I know most sklearn estimators don’t accept NaNs. Imputation is a standard preprocessing step 
  to make data usable without dropping rows.

### 4. Data Leakage
- **Issue**: The dataset included a derived column (`Churn_bin`) that duplicated the target variable, 
  leading to artificially perfect model scores (ROC-AUC = 1.0).
- **Fix**: Excluded any features containing “churn” in their name.
- **Knowledge**: Data leakage produces suspiciously high scores. I recognized that including target-derived 
  features invalidates the model evaluation.

---

## 📊 Final Results
- Logistic Regression: **ROC-AUC = 0.842**, **PR-AUC = 0.636**
- Random Forest: **ROC-AUC = 0.819**, **PR-AUC = 0.610**
- These scores are realistic and show meaningful predictive performance without leakage.

---

## 🧠 Key Learnings
- Always validate data types before training.
- Keep dependencies updated to avoid deprecated parameters.
- Use systematic missing value handling to improve robustness.
- Actively watch for data leakage when feature engineering.

---

## 🚀 Next Steps
- Add cross-validation for more robust evaluation.
- Implement feature importance analysis and interpretability tools.
- Expand automated testing for pipeline reliability.

---

This project demonstrates both my ability to **build and debug ML pipelines** and my ability to **use AI tools 
effectively without relying on them blindly**. I combined AI suggestions with my own knowledge of machine 
learning principles to ensure the code is correct and production-ready.
