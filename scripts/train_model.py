#!/usr/bin/env python3
"""
Train churn models (Logistic Regression + Random Forest) on Telco dataset.

- Loads cleaned table from outputs/bi_exports/telco_clean.csv  (created by your ingest/clean step)
- Splits into train/test correctly (no leakage)
- Builds pipelines:
    * logit: StandardScaler + LogisticRegression(max_iter=2000)
    * rf: RandomForestClassifier
- Evaluates (ROC-AUC, PR-AUC, classification report)
- Picks best by ROC-AUC, saves model
- Scores full clean table and writes outputs/bi_exports/churn_scored.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
BI_DIR = ROOT / "outputs" / "bi_exports"
MODEL_DIR = ROOT / "outputs" / "models"
BI_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_CSV = BI_DIR / "telco_clean.csv"  # produced by your ingest/clean script
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_clean():
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Clean file not found: {CLEAN_CSV}\n"
            "Run your ingest/clean script first to produce telco_clean.csv."
        )
    df = pd.read_csv(CLEAN_CSV)
    # Expect a binary target column named 'Churn' (Yes/No or 0/1). Adjust here if different.
    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column not found in telco_clean.csv")
    
    # Debug info
    print(f"✅ Loaded {len(df)} rows from {CLEAN_CSV}")
    print(f"📊 Churn column unique values: {df['Churn'].unique()}")
    print(f"📊 Churn column dtype: {df['Churn'].dtype}")
    
    return df

def build_preprocessor(df: pd.DataFrame):
    # Identify numeric vs categorical (excluding target and any churn-related columns)
    target = "Churn"
    # Exclude target and any columns that contain "churn" (case-insensitive) to prevent data leakage
    feature_cols = [c for c in df.columns if c != target and "churn" not in c.lower()]

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Pipelines for preprocessing
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # with_mean=False safe for sparse matrices
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )
    return pre, feature_cols

def main():
    df = load_clean()

    # Drop strong leakage columns if present (safety belt)
    leak_candidates = {"customerID", "CustomerID", "customer_id"}  # IDs shouldn’t be predictors
    leak_present = [c for c in leak_candidates if c in df.columns]
    if leak_present:
        df = df.drop(columns=leak_present)

    # Convert Churn column from "Yes"/"No" to 1/0
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn"])

    pre, feature_cols = build_preprocessor(df)
    
    # Debug info
    print(f"🔧 Preprocessor created with {len(feature_cols)} features")
    print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    print(f"📊 Feature types - Numeric: {len([c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])])}, Categorical: {len([c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])])}")
    print(f"🔍 Excluded columns (potential leakage): {[c for c in df.columns if c not in feature_cols and c != 'Churn']}")
    print(f"🔍 Feature columns: {feature_cols}")

    # Models
    logit = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None, random_state=RANDOM_STATE))
    ])

    rf = Pipeline(steps=[
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Fit + evaluate
    results = {}

    for name, pipe in [("logit", logit), ("rf", rf)]:
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        roc = roc_auc_score(y_test, proba)
        pr = average_precision_score(y_test, proba)
        print(f"\n=== {name} ===")
        print(f"ROC-AUC: {roc:.3f}   PR-AUC: {pr:.3f}")
        print(classification_report(y_test, pred, digits=3))
        results[name] = {"model": pipe, "roc": roc, "pr": pr}

    # Pick best by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc"])
    best_model = results[best_name]["model"]
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\n💾 Saved best model ({best_name}) → {BEST_MODEL_PATH}")

    # Score the FULL clean table (for BI & actioning)
    proba_full = best_model.predict_proba(X)[:, 1]
    pred_full = (proba_full >= 0.5).astype(int)

    scored = X.copy()
    scored["pred_proba"] = proba_full
    scored["pred_label"] = pred_full
    scored["Churn"] = y.values  # actuals for comparison

    out_csv = BI_DIR / "churn_scored.csv"
    scored.to_csv(out_csv, index=False)
    print(f"✅ Wrote scores → {out_csv} (rows={len(scored)})")

if __name__ == "__main__":
    main()
