#!/usr/bin/env python3
"""
One-hot encode categoricals, keep numerics, split into train/test,
and save matrices for modelling.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CLEAN_CSV = Path("outputs/bi_exports/telco_clean.csv")
OUT_DIR   = Path("outputs/bi_exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "Churn"       # expects "Yes"/"No"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

def main():
    df = pd.read_csv(CLEAN_CSV)

    # Target → binary 1/0
    y = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Basic feature selection: numerics + selected categoricals
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]

    # Remove ID-like or obvious leaks if present
    drop_like = {"customerID","customer_id","Churn"}  # extend if needed
    numeric_cols = [c for c in numeric_cols if c not in drop_like]
    cat_cols     = [c for c in cat_cols     if c not in drop_like]

    # One-hot encode categoricals
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X_num = df[numeric_cols].copy()
    X     = pd.concat([X_num, X_cat], axis=1).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save
    X_train.to_parquet(OUT_DIR / "X_train.parquet")
    X_test.to_parquet(OUT_DIR / "X_test.parquet")
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    print("✅ Saved train/test matrices in outputs/bi_exports/")
    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

if __name__ == "__main__":
    main()
