# model.py
# Responsible for: building, training, saving and loading the model
# We use Random Forest — a powerful algorithm perfect for churn prediction
# Just like the movie project used joblib to save the KNN model,
# we save our Random Forest model so we never have to retrain from scratch

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def build_model(X, y):

    # Split data into training and testing sets
    # 80% of data trains the model, 20% is held back to test it
    # Think of it like studying with 80% of past exam papers
    # and testing yourself on the remaining 20%
    # random_state=42 means the split is always the same — reproducible results
    # stratify=y ensures both splits have the same 27/73 churn ratio we saw earlier
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✅ Training set: {X_train.shape[0]} rows")
    print(f"✅ Testing set: {X_test.shape[0]} rows")

    # Build the Random Forest model
    # A Random Forest builds hundreds of decision trees and combines their
    # answers — like asking 100 doctors for a diagnosis instead of just one
    # n_estimators=100 means 100 trees
    # class_weight="balanced" handles our imbalanced data (27% churn vs 73% stayed)
    # it tells the model to pay extra attention to the minority class (churned customers)
    # random_state=42 ensures we get the same result every time we train
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )

    # Train the model — this is where the learning happens
    # The model studies X_train (features) and y_train (answers)
    # and figures out which features best predict churn
    print("\n⏳ Training model...")
    model.fit(X_train, y_train)
    print("✅ Model trained successfully")

    # Save the model to the models/ folder using joblib
    # This means we never have to retrain — just load it next time
    # Exactly like the movie project saved model.pkl
    model_path = os.path.join("models", "churn_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    return model, X_train, X_test, y_train, y_test


def load_model():

    # Load the saved model from disk
    # This is much faster than retraining from scratch every time
    model_path = os.path.join("models", "churn_model.pkl")

    if not os.path.exists(model_path):
        print("❌ No saved model found. Please run python model.py first.")
        return None

    model = joblib.load(model_path)
    print("✅ Model loaded from disk")
    return model


# Only runs when you execute this file directly with python model.py
# If another file imports model.py, this block is skipped
if __name__ == "__main__":

    # Load and prepare the data first
    from data_loader import load_data
    from data_prep import prepare_data, split_data

    df = load_data()
    df = prepare_data(df)
    X, y = split_data(df)

    # Build and save the model
    model, X_train, X_test, y_train, y_test = build_model(X, y)