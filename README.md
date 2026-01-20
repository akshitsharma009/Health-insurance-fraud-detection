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
- Trained and evaluated classification models
- Tuned decision threshold to balance precision and recall
- Deployed the final model as an interactive web application

---

## 🗂 Project Structure
Health-insurance-fraud-detection/
│
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── data/
│ ├── raw/ # Raw dataset
│ └── processed/ # Feature engineered data
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modeling.ipynb
│
├── src/
│ ├── input_schema.py # Input feature construction
│ ├── preprocessing.py # Data preprocessing utilities
│ ├── train.py # Model training
│ ├── train_final_model.py # Final model training
│ ├── predict.py # Prediction logic
│ └── evaluate.py # Model evaluation
│
└── models/
└── fraud_model.pkl # Trained ML model

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Git & GitHub

---

## 📊 Model Details
- Algorithms explored: Logistic Regression, Random Forest
- Final model selected based on ROC-AUC and recall
- Threshold tuning applied to improve fraud detection sensitivity

---

## 🎯 Key Features
- Claim-to-premium ratio analysis
- Investigation signal flags (police report, witnesses)
- Damage severity consistency checks
- Interactive fraud probability prediction

---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
📈 Results

ROC-AUC score: ~0.82

Improved fraud recall after threshold tuning

Model responds realistically to high-risk claim scenarios

🔮 Future Improvements

Add SHAP-based model explainability

Integrate FastAPI for REST-based inference

Use real-world Indian insurance datasets

Add user authentication for enterprise use

👤 Author

Akshit Sharma
B.Tech Engineering Student | Machine Learning Enthusiast
