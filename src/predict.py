import joblib

MODEL_PATH = "models/fraud_model.pkl"
THRESHOLD = 0.3

model = joblib.load(MODEL_PATH)

def predict_fraud(input_df):

    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= THRESHOLD)
    return prediction, prob
