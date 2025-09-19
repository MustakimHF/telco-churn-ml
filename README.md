# 📉 Project 5 – Customer Churn Prediction (Telco)

This project predicts **customer churn** using the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).  
It demonstrates **ETL, feature engineering, machine learning (Logistic Regression + Random Forest), model evaluation, and business framing**.  

---

## 🚀 What This Project Does

- 🧹 **Cleans and prepares** customer data for analysis  
- 📊 **Explores churn patterns** (customer tenure, contract type, charges)  
- 🤖 **Trains models** to predict churn probability  
- 🛡️ **Handles leakage, missing values, API changes** (see notes below)  
- 📈 **Exports scored dataset** (`churn_scored.csv`) for BI dashboards (Power BI / Tableau)  
- ✅ **Incorporates AI-assisted testing** (TestSprite) for debugging & validation  

---

## 🧰 Tech Stack

- **Python**: pandas, numpy, scikit-learn, joblib  
- **ML Models**: Logistic Regression, Random Forest  
- **Evaluation**: ROC-AUC, PR-AUC, Classification Report  
- **Visualisation / BI**: Power BI, Tableau  
- **Testing**: TestSprite (AI-assisted debugging, fixes validated manually)  

---

## 📁 Repository Structure

```
telco-churn-ml/
├── README.md                      # Project overview (this file)
├── requirements.txt               # Python dependencies
├── data/
│   └── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── outputs/
│   ├── models/                    # Saved ML models (.joblib)
│   ├── plots/                     # Optional plots
│   └── bi_exports/
│       ├── telco_clean.csv        # Cleaned dataset
│       └── churn_scored.csv       # Model-scored dataset for BI
└── scripts/
    ├── ingest_clean.py            # Cleans raw Telco CSV → telco_clean.csv
    └── train_model.py             # Trains, evaluates, exports best model
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

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the ingest/clean script

```bash
python scripts/ingest_clean.py
```

This produces: `outputs/bi_exports/telco_clean.csv`

### 4. Train and evaluate models

```bash
python scripts/train_model.py
```

This will:  
- Train Logistic Regression + Random Forest  
- Print ROC-AUC / PR-AUC / classification report  
- Save the **best model** → `outputs/models/best_model.joblib`  
- Export scored dataset → `outputs/bi_exports/churn_scored.csv`  

### 5. Use BI dashboards

- Load `churn_scored.csv` into Power BI or Tableau  
- Build KPIs (e.g., churn % by contract type, high-risk segments)  

---

## 🧪 TestSprite-Assisted Debugging

This project used **TestSprite (AI-assisted testing)** to strengthen reliability.  
Key contributions from TestSprite (validated manually by me):  

### 1. Data Type Mismatch
- **Issue**: `Churn` column had string values ("Yes"/"No").  
- **Fix**: Converted to binary integers (`1` = churned, `0` = not).  
- **Knowledge**: Models require numeric targets; mapping strings was the correct fix.

### 2. Deprecated Parameter in Scikit-learn
- **Issue**: `OneHotEncoder` no longer supports `sparse=True`.  
- **Fix**: Updated to `sparse_output=True`.  
- **Knowledge**: I track scikit-learn API changes; confirmed in docs.

### 3. Missing Value Handling
- **Issue**: LogisticRegression failed due to NaNs.  
- **Fix**: Added imputers (median for numeric, "missing" for categoricals).  
- **Knowledge**: Most sklearn models don’t handle NaNs → imputation is standard.

### 4. Data Leakage
- **Issue**: Column (`Churn_bin`) duplicated the target → artificial ROC-AUC = 1.0.  
- **Fix**: Excluded any columns containing "churn".  
- **Knowledge**: Leakage inflates scores; removing target-derived features restored realism.

👉 While TestSprite detected these, I validated and refined the fixes myself.

---

## 📊 Example Results

- Logistic Regression: ROC-AUC ~ **0.84**, PR-AUC ~ **0.64**  
- Random Forest: ROC-AUC ~ **0.82**, PR-AUC ~ **0.61**  
- Target distribution: ~26% churn, ~74% non-churn  
- Scored dataset → includes `pred_proba` (likelihood of churn) & `pred_label` (0/1)

### https://img.shields.io/badge/power_bi-F2C811?style=for-the-badge&logo=powerbi&logoColor=black
<img width="1294" height="726" alt="image" src="https://github.com/user-attachments/assets/0c7e42b0-2a04-433f-b961-236d785d92e6" />



---

## 🎯 Why This Project Matters

This project demonstrates:  
- **Business framing**: churn → lost revenue → retention campaigns  
- **ETL + ML pipeline**: clean, train, export for BI  
- **Leakage prevention & testing**: avoids false success  
- **Realistic performance**: balanced precision/recall trade-offs  
- **Communication**: can explain model outputs to non-technical stakeholders  

---

## 📄 Licence  
MIT Licence – free to use and adapt.  
