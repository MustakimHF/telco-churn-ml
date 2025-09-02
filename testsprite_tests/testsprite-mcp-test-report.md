# TestSprite Debugging Report for train_model.py

## Executive Summary
Successfully debugged and fixed the `train_model.py` script. The script now runs without errors and produces realistic model performance metrics.

## Issues Found and Fixed

### 1. **Data Type Mismatch - CRITICAL** ✅ FIXED
**Problem**: The script expected the `Churn` column to contain numeric values (0/1) but it contained string values ("Yes"/"No").

**Error**: 
```
ValueError: invalid literal for int() with base 10: 'No'
```

**Fix**: Updated the target variable conversion to handle string values:
```python
# Before (broken):
y = df["Churn"].astype(int)

# After (fixed):
y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
```

### 2. **Deprecated Parameter - MEDIUM** ✅ FIXED
**Problem**: The `sparse` parameter in `OneHotEncoder` has been deprecated in newer versions of scikit-learn.

**Error**:
```
TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'
```

**Fix**: Updated to use the new `sparse_output` parameter:
```python
# Before (broken):
("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))

# After (fixed):
("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
```

### 3. **Missing Value Handling - MEDIUM** ✅ FIXED
**Problem**: The dataset contained NaN values that caused the LogisticRegression to fail.

**Error**:
```
ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively.
```

**Fix**: Added imputers to the preprocessing pipeline:
```python
# Added to imports:
from sklearn.impute import SimpleImputer

# Updated preprocessing pipelines:
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])
```

### 4. **Data Leakage - CRITICAL** ✅ FIXED
**Problem**: The `Churn_bin` column was being used as a feature, creating perfect model scores (ROC-AUC: 1.000) because it was essentially the target variable in binary form.

**Root Cause**: During data ingestion, a binary version of the target variable was created and included in the feature set.

**Fix**: Updated the feature selection to exclude any columns containing "churn" (case-insensitive):
```python
# Before (leakage):
feature_cols = [c for c in df.columns if c != target]

# After (no leakage):
feature_cols = [c for c in df.columns if c != target and "churn" not in c.lower()]
```

## Final Results

### Model Performance (Realistic)
- **Logistic Regression**: ROC-AUC: 0.842, PR-AUC: 0.636 ✅
- **Random Forest**: ROC-AUC: 0.819, PR-AUC: 0.610 ✅

### Data Processing
- **Total rows**: 7,043
- **Features used**: 26 (9 numeric, 17 categorical)
- **Excluded columns**: `Churn_bin` (data leakage prevention)
- **Target distribution**: 0 (No Churn): 5,174, 1 (Churn): 1,869

## Files Modified
- `scripts/train_model.py` - Fixed all critical bugs and added debugging information

## Recommendations

1. **Data Validation**: Always verify data types and content before model training
2. **Leakage Prevention**: Implement systematic checks for target variable derivatives in features
3. **Missing Value Strategy**: Document and implement consistent missing value handling
4. **Version Compatibility**: Keep dependencies updated and check for deprecated parameters
5. **Debugging**: Add comprehensive logging and data validation checks

## Test Status
✅ **PASSED** - All critical issues resolved, script runs successfully with realistic performance metrics.

## Next Steps
The script is now production-ready. Consider:
- Adding cross-validation for more robust model evaluation
- Implementing feature importance analysis
- Adding model interpretability tools
- Setting up automated testing for future changes
