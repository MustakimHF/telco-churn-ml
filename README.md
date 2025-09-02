# 📉 Customer Churn Prediction (Telco Dataset)

A machine learning project using the **Telco Customer Churn** dataset to predict which customers are likely to leave.  
This repo demonstrates **end-to-end churn modelling** with business framing:  
ETL → Feature Engineering → Model Training → Scoring → BI integration (Power BI / Tableau).

---

## 🚀 What This Project Does

- 📥 **Loads & cleans** the Telco dataset (demographics, services, billing, churn flag).  
- 🧹 **Prepares features** (numeric scaling, categorical encoding, leakage control).  
- 🤖 **Trains models**: Logistic Regression & Random Forest.  
- 📊 **Evaluates** with ROC-AUC, PR-AUC, classification report.  
- 🏆 **Saves the best model** (by ROC-AUC).  
- 🔮 **Scores all customers** with churn probabilities → exports to CSV for BI dashboards.  
- 📈 **Supports business actions** like targeting high-risk customers with retention campaigns.  

---

## 🧰 Tech Stack

- **Python**: `pandas`, `numpy`, `scikit-learn`, `joblib`  
- **Visualisation**: Power BI / Tableau / Excel (via CSV exports)  
- **ETL**: Clean + preprocess pipeline  
- **ML**: Logistic Regression, Random Forest  

---

## 📁 Repository Structure

```
telco-churn-ml/
├── README.md
├── requirements.txt
├── data/
│   └── raw/                           # Original Telco dataset (CSV)
├── outputs/
│   ├── models/
│   │   └── best_model.joblib          # Best saved model (logit or RF)
│   └── bi_exports/
│       ├── telco_clean.csv            # Cleaned dataset
│       └── churn_scored.csv           # Full dataset with churn predictions
└── scripts/
    ├── ingest_clean.py                # Clean and preprocess Telco dataset
    └── train_model.py                 # Train, evaluate, save model + score full dataset
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

### 3. Ingest & Clean Data

Prepare the dataset for ML:  
```bash
python scripts/ingest_clean.py
```

Outputs:  
- `outputs/bi_exports/telco_clean.csv`

---

### 4. Train & Score Models

Run training + scoring:  
```bash
python scripts/train_model.py
```

Outputs:  
- `outputs/models/best_model.joblib`  
- `outputs/bi_exports/churn_scored.csv`  

---

### 5. Sanity Checks

```bash
# Does model file exist?
dir outputs/models/best_model.joblib

# Peek at churn_scored.csv
python - <<'PY'
import pandas as pd
d = pd.read_csv('outputs/bi_exports/churn_scored.csv')
print(d.head())
print('Rows:', len(d), '| Pred churners:', d['pred_label'].sum())
print('Proba range:', float(d['pred_proba'].min()), '→', float(d['pred_proba'].max()))
PY
```

---

## 📊 Example BI Dashboard

Use `outputs/bi_exports/churn_scored.csv` in **Power BI / Tableau / Excel**:  
- KPI Cards: Predicted churners, churn rate, retention rate.  
- Bar chart: Churn rate by Contract type, Internet service, Payment method.  
- Matrix: Churn by Gender × Senior Citizen × Tenure buckets.  
- Slicers: contract type, payment method, region.  

---

## 🎯 Why This Project Matters

- **Business framing**: ties churn prediction to retention campaigns and revenue impact.  
- **ETL → ML → BI**: full data pipeline that mirrors real data science workflows.  
- **Explains trade-offs**: precision vs recall, targeting the right customers.  
- **Recruiter appeal**: interpretable logistic regression + robust random forest baseline.  

---

## 🔒 Notes

- The dataset comes from the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).  
- `.env` is not needed for this project.  
- `outputs/` can be safely `.gitignore`d except for sample exports.  

---

## 📄 Licence

MIT Licence – free to use and adapt.  
