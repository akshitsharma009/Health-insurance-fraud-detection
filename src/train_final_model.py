import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load processed data
df = pd.read_csv("data/processed/insurance_fraud_fe.csv")

X = df.drop(columns=["fraud_reported"])
y = df["fraud_reported"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# Train on FULL data (final model)
clf.fit(X, y)

# Save model
joblib.dump(clf, "models/fraud_model.pkl")

print("✅ Final model saved to models/fraud_model.pkl")
