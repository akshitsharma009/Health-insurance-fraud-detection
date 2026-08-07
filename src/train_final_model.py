"""
train_final_model.py
---------------------
Trains the final production model on the FULL cleaned dataset and saves it
to models/fraud_model.pkl for use by the Streamlit app (src/predict.py).

Enhancements over the original version:
- Uses src.preprocessing.engineer_features() so training and any future
  batch scoring share one feature-engineering implementation (previously
  the "?" -> NaN bug and identifier-leakage columns were only fixed in
  ad-hoc notebook cells, not in the production training script).
- Switched the model from Logistic Regression to a hyperparameter-tuned
  XGBoost classifier, which outperformed both Logistic Regression and
  Random Forest in 5-fold cross-validated ROC-AUC during model selection
  (see notebooks/03_modeling.ipynb and reports/model_comparison.md for the
  full comparison).
- Missing values are now explicitly imputed (median for numeric, most
  frequent for categorical) inside the pipeline instead of relying on
  OneHotEncoder to silently absorb "?" as its own category.
"""

import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import engineer_features, get_feature_columns

# ------------------------------------------------------------
# Load raw data and apply the shared feature-engineering pipeline
# ------------------------------------------------------------
raw_df = pd.read_csv("data/raw/fraud_insurance_claims.csv")
df = engineer_features(raw_df)

X = df.drop(columns=["fraud_reported"])
y = df["fraud_reported"]

num_cols, cat_cols = get_feature_columns(df)

# ------------------------------------------------------------
# Preprocessing: impute + scale numeric, impute + one-hot categorical
# ------------------------------------------------------------
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

# ------------------------------------------------------------
# Model: XGBoost, best hyperparameters found via RandomizedSearchCV
# with 5-fold stratified cross-validation, optimizing ROC-AUC.
# scale_pos_weight compensates for the ~25% fraud / 75% non-fraud
# class imbalance (equivalent in spirit to class_weight='balanced').
# ------------------------------------------------------------
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

model = XGBClassifier(
    n_estimators=150,
    max_depth=2,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=0.85,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=1,
)

clf = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])

# ------------------------------------------------------------
# Train on FULL data (final production model).
# NOTE: this is intentional for the deployed model -- hold-out
# validation and hyperparameter search were already performed
# separately (see notebooks/03_modeling.ipynb). Training the final
# artifact on 100% of available data is standard practice once the
# model/hyperparameters are locked, to give the deployed model the
# maximum amount of signal to learn from.
# ------------------------------------------------------------
clf.fit(X, y)

joblib.dump(clf, "models/fraud_model.pkl")

print("Final model saved to models/fraud_model.pkl")
print(f"Trained on {len(X)} rows | {len(num_cols)} numeric + {len(cat_cols)} categorical features")
