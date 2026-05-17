# predict.py
# Responsible for: predicting churn for a brand new customer
# This is where the model gets put to real use
# We pass in a new customer's details and get back a churn prediction
# This is exactly what a business would use in production

import pandas as pd
import numpy as np


def predict_customer(model, customer_data, training_columns):

    # customer_data is a dictionary of a single customer's details
    # We convert it to a DataFrame because that's what the model expects
    # A model trained on a DataFrame needs a DataFrame to predict on
    customer_df = pd.DataFrame([customer_data])

    # One-hot encode the customer data exactly like we did in data_prep.py
    # This converts text columns like gender into binary columns like gender_Male
    customer_df = pd.get_dummies(customer_df)

    # The model was trained on 30 specific columns
    # The new customer data might have more or fewer columns after encoding
    # reindex makes sure we have EXACTLY the same columns in the same order
    # fill_value=0 means any missing columns get filled with 0
    customer_df = customer_df.reindex(columns=training_columns, fill_value=0)

    # Make the prediction
    # predict() returns 0 (stays) or 1 (churns)
    # predict_proba() returns the probability of each outcome
    # e.g. [0.3, 0.7] means 30% chance of staying, 70% chance of churning
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0]

    # probability is an array of two values [prob_stayed, prob_churned]
    churn_probability = probability[1]  # we only want the churn probability

    # Print the result in a human readable way
    print("\n--- Churn Prediction ---")
    if prediction == 1:
        print(f"⚠️  This customer is LIKELY TO CHURN")
        print(f"    Churn probability: {churn_probability:.2%}")
    else:
        print(f"✅  This customer is LIKELY TO STAY")
        print(f"    Churn probability: {churn_probability:.2%}")

    return prediction, churn_probability


# Only runs when you execute this file directly with python predict.py
# If another file imports predict.py, this block is skipped
if __name__ == "__main__":

    from data_loader import load_data
    from data_prep import prepare_data, split_data
    from model import load_model

    # Load and prepare data to get the training columns
    # We need these to make sure our new customer has the exact same columns
    df = load_data()
    df = prepare_data(df)
    X, y = split_data(df)

    # Load the saved model from disk
    model = load_model()

    # Define a brand new customer the model has never seen before
    # These are the RAW values — exactly like a real form submission
    # Customer A — high risk profile
    # Month to month contract, high charges, new customer, no online security
    high_risk_customer = {
        "SeniorCitizen": 0,
        "tenure": 2,                          # only been a customer for 2 months
        "MonthlyCharges": 85.0,               # high monthly bill
        "TotalCharges": 170.0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",     # fiber optic tends to have higher churn
        "OnlineSecurity": "No",               # no online security
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",         # month to month = easiest to leave
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"   # electronic check has higher churn rate
    }

    # Customer B — low risk profile
    # Two year contract, low charges, long tenure, has online security
    low_risk_customer = {
        "SeniorCitizen": 0,
        "tenure": 60,                         # been a customer for 5 years
        "MonthlyCharges": 45.0,               # lower monthly bill
        "TotalCharges": 2700.0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",              # has online security
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",               # two year contract = committed customer
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)"
    }

    # Get the training columns — everything except Churn
    training_columns = X.columns.tolist()

    # Predict for both customers
    print("\n========== Customer A (High Risk) ==========")
    predict_customer(model, high_risk_customer, training_columns)

    print("\n========== Customer B (Low Risk) ==========")
    predict_customer(model, low_risk_customer, training_columns)