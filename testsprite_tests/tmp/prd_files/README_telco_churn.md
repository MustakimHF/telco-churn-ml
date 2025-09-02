# 📉 Customer Churn Prediction (Telco Dataset)

A machine learning project using the **Telco Customer Churn dataset** to predict customer churn.  
It demonstrates **ETL (Extract-Transform-Load)**, **feature engineering**, **supervised learning (classification)**, **model evaluation**, and **business framing** (retention campaigns & segmentation).  

---

## 🚀 What This Project Does  
- 📥 **Ingests & cleans** the Telco dataset (handles missing values, encodes categories, scales numerics)  
- 🔍 **Defines churn** as customers with `Churn = Yes` in the dataset  
- 📊 **Segments customers** (e.g., high/medium/low usage via RFM-style buckets)  
- 🤖 **Trains ML models** (Logistic Regression, Random Forest)  
- 📈 **Evaluates** with ROC-AUC, PR-AUC, and classification reports  
- 📝 **Exports predictions** (`churn_scored.csv`) for Power BI/Tableau/Excel dashboards  

---

## 🧰 Tech Stack  
- **Python**: `pandas`, `numpy`, `scikit-learn`  
- **ML**: Logistic Regression, Random Forest  
- **Exports**: `joblib` (saved models), CSVs for BI tools  
- **Optional**: Streamlit/Flask for deployment  

---

## 📁 Repository Structure  

```
telco-churn-ml/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # dataset (Kaggle download)
├── outputs/
│   ├── models/       # saved best model
│   └── bi_exports/   # clean table + churn_scored.csv
└── scripts/
    ├── ingest_clean.py   # prepare clean dataset (etl)
    └── train_model.py    # train, evaluate, export predictions
```

---

## ▶️ How to Run  

### 1. Create a virtual environment  

**Windows PowerShell**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python -m venv venv
source venv/bin/activate
```

---

### 2. Install dependencies  

```bash
pip install -r requirements.txt
```

---

### 3. Add the dataset  

Download the Telco Churn dataset from Kaggle:  
[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  

Place it here:  

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

### 4. Run ETL (clean data)  

```bash
python scripts/ingest_clean.py
```

This will create:

```
outputs/bi_exports/telco_clean.csv
```

---

### 5. Train models & export predictions  

```bash
python scripts/train_model.py
```

This will:  
- Train Logistic Regression & Random Forest  
- Save the best model → `outputs/models/best_model.joblib`  
- Export scored dataset → `outputs/bi_exports/churn_scored.csv`  

---

### 6. Sanity checks  

```bash
# File exists?
dir outputs/bi_exports/churn_scored.csv

# Peek data
python -c "import pandas as pd;print(pd.read_csv('outputs/bi_exports/churn_scored.csv').head())"
```

---

## 📊 Example BI Visuals  

- KPI Cards: % Churn, Predicted Churners, High-Risk Customers  
- Bar: Churn Rate by Contract Type  
- Line: Monthly Churn Trend  
- Matrix: Segment × Churn (geo/usage bucket)  
- Slicers: tenure group, contract type, payment method  

---

## 🎯 Why This Project Matters  

This project demonstrates:  
- **Business framing**: churn tied to customer retention campaigns  
- **Segmentation**: usage buckets (high/medium/low) and geography  
- **ML workflow**: preprocessing, training, evaluation, model saving  
- **Actionable outputs**: churn scores for dashboards and interventions  

📌 *This mirrors how real subscription businesses tackle churn, making it ideal for a student portfolio or internship application.*  

---

## 🔒 Notes  
- `.env` not required (no API keys).  
- Dataset is public, from Kaggle.  
- Use `gitignore` to exclude large files (e.g., models, raw data).  

---

## 📄 Licence  
MIT Licence – free to use and adapt.  
