# main.py
# Responsible for: tying the entire pipeline together in one place
# This is the single entry point for the whole project
# Running this file runs everything in the correct order
# Just like the movie project's main.py tied everything together

import os
from setup import download_data
from data_loader import load_data
from data_prep import prepare_data, split_data
from model import build_model, load_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance
from predict import predict_customer


def run_pipeline():

    print("=" * 50)
    print("   CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 50)

    # ── Step 1: Download data if it doesn't exist ──────────
    # setup.py checks if the CSV is already in data/
    # if it is, it skips the download entirely
    print("\n📥 STEP 1: Checking dataset...")
    download_data()

    # ── Step 2: Load the raw CSV into pandas ───────────────
    print("\n📂 STEP 2: Loading data...")
    df = load_data()

    # If data failed to load stop the pipeline immediately
    # No point continuing without data
    if df is None:
        print("❌ Pipeline stopped — could not load data.")
        return

    # ── Step 3: Clean and prepare the data ────────────────
    print("\n🧹 STEP 3: Preparing data...")
    df = prepare_data(df)
    X, y = split_data(df)

    # ── Step 4: Load or train the model ───────────────────
    # If a saved model exists load it from disk — fast
    # If no saved model exists train a new one and save it
    # Exactly like the movie project's smart loading pattern
    print("\n🤖 STEP 4: Loading or training model...")
    model_path = os.path.join("models", "churn_model.pkl")

    if os.path.exists(model_path):
        # Model already exists — load it instantly
        model = load_model()

        # We still need X_test and y_test to evaluate the model
        # We use the same random_state=42 and stratify=y as model.py
        # This guarantees we get the exact same split every time
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    else:
        # No saved model — train from scratch and save it
        model, _, X_test, _, y_test = build_model(X, y)

    # ── Step 5: Evaluate the model ────────────────────────
    # See how well the model performs on data it has never seen
    print("\n📊 STEP 5: Evaluating model...")
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, X)

    # ── Step 6: Predict on new customers ──────────────────
    # Run predictions on two example customers
    # In production this would come from a live database or API
    print("\n🎯 STEP 6: Predicting on new customers...")

    # Get the exact column names the model was trained on
    training_columns = X.columns.tolist()

    # High risk customer — likely to churn
    high_risk_customer = {
        "SeniorCitizen": 0,
        "tenure": 2,
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }

    # Low risk customer — likely to stay
    low_risk_customer = {
        "SeniorCitizen": 0,
        "tenure": 60,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)"
    }

    print("\n========== Customer A (High Risk) ==========")
    predict_customer(model, high_risk_customer, training_columns)

    print("\n========== Customer B (Low Risk) ==========")
    predict_customer(model, low_risk_customer, training_columns)

    print("\n" + "=" * 50)
    print("   ✅ PIPELINE COMPLETE")
    print("=" * 50)


# Only runs when you execute this file directly with python main.py
# If another file imports main.py, this block is skipped
if __name__ == "__main__":
    run_pipeline()