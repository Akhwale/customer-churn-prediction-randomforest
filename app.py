# app.py
# Responsible for: serving the web app and handling predictions
# Flask receives raw form data from the user, processes it intelligently,
# and returns a churn prediction without the user knowing anything about
# how the model works under the hood

from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from data_loader import load_data
from data_prep import prepare_data, split_data
from model import load_model, build_model

# Initialize the Flask app
# Flask uses this to know where to look for templates and static files
app = Flask(__name__)


# ── Load everything ONCE when the server starts ─────────────────────────
# This is the most important performance decision in the app
# We load the data and model into memory once at startup
# Every prediction request then uses what's already in memory — instant response
# Never load the model inside a route — that would retrain on every request

print("🚀 Starting server — loading data and model...")

df = load_data()
df = prepare_data(df)
X, y = split_data(df)

# Get the exact column names the model was trained on
# Every prediction must match these columns exactly
TRAINING_COLUMNS = X.columns.tolist()

# Load saved model if it exists, otherwise train a new one
model_path = os.path.join("models", "churn_model.pkl")
if os.path.exists(model_path):
    model = load_model()
else:
    model, _, _, _, _ = build_model(X, y)

print("✅ Server ready — listening for requests")


# ── Helper function: convert raw form data into model input ─────────────
# This is the intelligent conversion layer
# It takes exactly what the user typed in the form
# and transforms it into what the model expects
# The user never sees any of this happening

def process_customer(raw_data):

    # Step 1: Convert all numeric fields from strings to numbers
    # Form submissions always come in as strings — even numbers
    # e.g. "85.0" needs to become 85.0 before the model can use it
    numeric_fields = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for field in numeric_fields:
        raw_data[field] = float(raw_data[field])

    # Step 2: Convert the raw data dictionary into a single-row DataFrame
    # The model was trained on a DataFrame so it needs a DataFrame to predict on
    customer_df = pd.DataFrame([raw_data])

    # Step 3: One-hot encode the text columns
    # Converts e.g. gender="Male" into gender_Male=1, gender_Female=0
    # This mirrors exactly what data_prep.py did during training
    customer_df = pd.get_dummies(customer_df)

    # Step 4: Reindex to match the exact columns the model was trained on
    # The form might produce slightly different columns after encoding
    # reindex makes sure we have exactly the right columns in the right order
    # Any missing columns get filled with 0 — they just weren't selected
    customer_df = customer_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    return customer_df


# ── Route 1: Home page ───────────────────────────────────────────────────
# When someone visits http://127.0.0.1:5000 they see the form
@app.route("/")
def home():
    return render_template("index.html")


# ── Route 2: Prediction endpoint ─────────────────────────────────────────
# When the form is submitted it sends a POST request to /predict
# We process the raw data and return a JSON prediction
# JSON is used because it's the universal language between systems
@app.route("/predict", methods=["POST"])
def predict():

    # Get the raw form data sent from the browser
    # request.form contains all the fields the user filled in
    # We convert it to a regular dictionary so we can work with it
    raw_data = request.form.to_dict()

    # Run the intelligent conversion — raw form data → model input
    customer_df = process_customer(raw_data)

    # Make the prediction
    # predict() returns 0 (stays) or 1 (churns)
    prediction = int(model.predict(customer_df)[0])

    # predict_proba() returns probabilities for both outcomes
    # [0] = probability of staying, [1] = probability of churning
    probability = model.predict_proba(customer_df)[0]
    churn_probability = round(float(probability[1]) * 100, 2)
    stay_probability  = round(float(probability[0]) * 100, 2)

    # Build a clean human readable result
    result = "LIKELY TO CHURN" if prediction == 1 else "LIKELY TO STAY"

    # Determine risk level based on churn probability
    # This gives the business a simple traffic light system
    if churn_probability >= 60:
        risk_level = "High Risk"
    elif churn_probability >= 30:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    # Return the prediction as JSON
    # jsonify converts a Python dictionary into a proper JSON response
    return jsonify({
        "prediction": result,
        "churn_probability": f"{churn_probability}%",
        "stay_probability": f"{stay_probability}%",
        "risk_level": risk_level
    })


# ── Start the server ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=True means the server restarts automatically when you save changes
    # NEVER use debug=True in production — only for local development
    app.run(debug=True)