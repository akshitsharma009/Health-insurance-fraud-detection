import joblib

MODEL_PATH = "models/fraud_model.pkl"

# Threshold selected via 5-fold cross-validated out-of-fold predictions on
# the training set (not the held-out test set, to avoid tuning on data used
# for reporting). 0.30 gives the best F1 while keeping fraud recall high,
# which matches the project's stated priority: missing real fraud is more
# costly than investigating a false positive.
THRESHOLD = 0.3

model = joblib.load(MODEL_PATH)

def predict_fraud(input_df):

    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= THRESHOLD)
    return prediction, prob
