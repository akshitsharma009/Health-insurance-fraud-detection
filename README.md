> This project was designed, implemented, and deployed by Akshit Sharma.

# Health Insurance Fraud Detection

An end-to-end Machine Learning project to identify potentially fraudulent health insurance claims using structured claim data.
The project covers data analysis, feature engineering, model training, threshold tuning, and deployment using Streamlit.

---

## 🚀 Live Demo
👉 *Streamlit app deployed on Streamlit Cloud*
(https://health-insurance-fraud-detection-hryor8gtsezvr72hyskzam.streamlit.app/)

---

## 📌 Problem Statement
Health insurance fraud leads to significant financial losses for insurers.
This project aims to build a predictive system that flags high-risk insurance claims by learning patterns from historical claim data.

---

## 🧠 Solution Overview
- Performed Exploratory Data Analysis (EDA)
- Engineered domain-specific fraud indicators
- Fixed hidden data-quality issues found during a model review (see **Model Improvements** below)
- Trained and compared three classification models with cross-validation and hyperparameter tuning
- Tuned decision threshold using unbiased out-of-fold predictions
- Deployed the final model as an interactive web application

---

## 📂 Project Structure
```
Health-insurance-fraud-detection/
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── data/
│ ├── raw/
│ │ └── fraud_insurance_claims.csv # Raw dataset
│ └── processed/
│ └── insurance_fraud_fe.csv # Feature engineered data
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modeling.ipynb
│
├── src/
│ ├── input_schema.py # Input feature construction
│ ├── preprocessing.py # Shared feature-engineering pipeline (raw -> model-ready)
│ ├── train_final_model.py # Final model training (uses preprocessing.py)
│ ├── predict.py # Prediction logic
│ └── evaluate.py # Reusable evaluation utilities
│
└── models/
└── fraud_model.pkl # Trained ML model
```

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- Git & GitHub

---

## 📊 Model Details
- **Algorithms compared:** Logistic Regression, Random Forest, XGBoost (all with 5-fold stratified cross-validation)
- **Final model:** XGBoost, hyperparameter-tuned via `RandomizedSearchCV` optimizing ROC-AUC
  (`n_estimators=150, max_depth=2, learning_rate=0.05, subsample=1.0, colsample_bytree=0.85`)
- **Class imbalance handling:** `scale_pos_weight` set to the fraud/non-fraud ratio (~25% fraud rate in the dataset)
- **Threshold:** 0.30, selected using 5-fold out-of-fold predictions on the training set (not the held-out test set), to avoid tuning the threshold on the same data used to report final metrics

---

## 🔧 Model Improvements (Performance Enhancement Pass)

A follow-up review of the original pipeline surfaced two real data-quality issues and one modeling gap. All three were fixed, and the fixes are implemented in `src/preprocessing.py` and `src/train_final_model.py` (not just a one-off notebook cell), so they persist for any future retraining.

**1. Hidden missing values.** `collision_type`, `property_damage`, and `police_report_available` — three of the project's own headline "investigation signal" features — had missing values encoded as the literal string `"?"` rather than `NaN` (17.8%, 36.0%, and 34.3% missing respectively). The original `df.isnull().sum()` check in the EDA notebook did not catch this, so the model was silently treating "unknown" as its own category instead of an informative gap. Fix: these are now converted to real `NaN`, properly imputed, and three new `*_missing` indicator flags were added as explicit features, since missingness itself turned out to carry signal (e.g. a claim with no police report on file).

**2. Identifier leakage.** `policy_number` (1,000 unique values across 1,000 rows) and `incident_location` (a free-text street address, also unique per row) were being one-hot encoded and fed directly into the model. Encoding a column that's effectively a row index lets a model latch onto per-row artifacts on the training set rather than generalizable patterns, hurting real-world performance. Fix: these, along with `insured_zip` (995 unique values, same risk), are now dropped before training.

**3. No model comparison beyond Logistic Regression / Random Forest, and no cross-validation.** The original evaluation used a single train/test split, which has high variance on a dataset this size (1,000 rows). Fix: added 5-fold stratified cross-validation for model selection, a hyperparameter search (`RandomizedSearchCV`) for Random Forest and XGBoost, and XGBoost as a third candidate model — which won.

**4. Inconsistent claim sub-components at inference time.** The Streamlit form only collects a single `Total Claim Amount`, while `injury_claim`, `property_claim`, and `vehicle_claim` silently stayed at fixed defaults (₹30,000 / ₹30,000 / ₹40,000) no matter what total the user entered. In the training data, these three components sum *exactly* to `total_claim_amount` for every single row — a strict pattern the model relies on. Entering a total far from the ~₹1,00,000 default (e.g. ₹12,00,000) produced an internally-inconsistent row unlike anything seen in training, which could distort predictions for unusually large or small claims. Fix: `src/input_schema.py` now auto-derives the three sub-claims from whichever total claim amount is entered, using the average split observed in the training data (≈13.9% injury / 13.9% property / 72.2% vehicle), so the sub-claims always sum back to the entered total.

### Results: Before vs. After

| Metric (fraud / positive class) | Original (Logistic Regression, threshold 0.3) | Enhanced (XGBoost, threshold 0.3) |
|---|---:|---:|
| ROC-AUC (held-out test) | 0.817 | **0.827** |
| ROC-AUC (5-fold CV, more reliable given small dataset) | — | **0.866** |
| Precision | 0.532 | **0.631** |
| Recall | 0.806 | **0.855** |
| F1-score | 0.641 | **0.726** |
| Overall accuracy | 77.6% | **84.0%** |

The biggest practical win is **precision jumping from 53% to 63% while recall also improved** — the original model was flagging a large number of legitimate claims as fraud to hit its recall target; the enhanced model catches slightly *more* actual fraud while generating noticeably fewer false alarms for investigators to chase.

*(All numbers above are from an actual re-run of the training and evaluation pipeline on this dataset, not estimated.)*

---

## 🎯 Key Features
- Claim-to-premium ratio analysis
- Investigation signal flags (police report, witnesses) — now including explicit missing-data flags
- Damage severity consistency checks
- Interactive fraud probability prediction

---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model from scratch after any data or code changes:
```bash
python src/train_final_model.py
```

---

## 📈 Results

- ROC-AUC score: **~0.83** (test) / **~0.87** (5-fold cross-validated)
- Fraud recall: **~86%**, fraud precision: **~63%** at the selected threshold
- Model responds realistically to high-risk claim scenarios (verified with synthetic suspicious vs. normal claim inputs through the full app pipeline)

---

## 🔮 Future Improvements
- Add SHAP-based model explainability
- Integrate FastAPI for REST-based inference
- Use real-world Indian insurance datasets
- Add user authentication for enterprise use
- Explore ROC/precision-recall-curve-based threshold optimization instead of a manual sweep
- Add automated model-performance regression tests (e.g. fail CI if retraining drops ROC-AUC below a floor)

---

## 👤 Author

Akshit Sharma
B.Tech Engineering Student | Machine Learning Enthusiast
