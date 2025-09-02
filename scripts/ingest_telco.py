#!/usr/bin/env python3
"""
Load Telco dataset, clean types, engineer simple engagement signals,
and ensure a binary churn label. Exports tidy CSV for modelling & BI.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW_CSV = Path("data/raw/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT_CSV = Path("outputs/bi_exports/telco_clean.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# If the raw CSV lacks "Churn", we can create a proxy label (business rule).
MAKE_PROXY_CHURN_IF_MISSING = True

def yesno_to_int(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
              .map({"yes": 1, "no": 0, "true": 1, "false": 0})
              .astype("Int64"))

def main():
    df = pd.read_csv(RAW_CSV)

    # Coerce TotalCharges to numeric (some rows have spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Standardise Yes/No columns to 1/0 (keep original labels for BI if you like)
    yn_cols = [c for c in df.columns
               if df[c].dtype == "object"
               and set(df[c].dropna().unique()).issubset({"Yes","No","yes","no","True","False"})]
    for c in yn_cols:
        df[c + "_bin"] = yesno_to_int(df[c])

    # Light “RFM-style” engagement signals (not true RFM, but business-friendly):
    # - service_count: number of opted-in services among typical service columns
    service_cols = [c for c in df.columns if c in [
        "PhoneService","MultipleLines","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","PaperlessBilling"
    ]]
    if service_cols:
        df["service_count"] = (df[service_cols] == "Yes").sum(axis=1)

    # - charge_bucket: terciles of MonthlyCharges (low/med/high spend)
    if "MonthlyCharges" in df.columns:
        df["charge_bucket"] = pd.qcut(df["MonthlyCharges"], q=3, labels=["Low","Mid","High"])

    # - tenure_bucket: terciles of tenure (new/established/loyal)
    if "tenure" in df.columns:
        # Make sure tenure numeric
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
        df["tenure_bucket"] = pd.qcut(df["tenure"].fillna(0), q=3, labels=["New","Established","Loyal"])

    # Ensure a binary target "Churn" exists (Yes/No)
    if "Churn" not in df.columns and MAKE_PROXY_CHURN_IF_MISSING:
        # Proxy definition (basic, editable):
        # churn = month-to-month & short tenure & low service_count  → "Yes"
        df["Churn"] = np.where(
            (df.get("Contract","Month-to-month") == "Month-to-month")
            & (df.get("tenure", 0) <= 2)
            & (df.get("service_count", 0) <= 2),
            "Yes", "No"
        )

    # Save for modelling/BI
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} with {len(df)} rows and {df.shape[1]} columns.")

if __name__ == "__main__":
    main()
