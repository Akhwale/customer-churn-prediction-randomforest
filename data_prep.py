# data_prep.py
# Responsible for: cleaning and preparing the raw data for machine learning
# Raw data is messy — models can only work with clean, numerical data
# This file fixes every problem we spotted in data_loader.py

import pandas as pd


def prepare_data(df):

    # Step 1: Drop customerID — it's just a unique identifier
    # It has no relationship with churn so it would confuse the model
    df = df.drop(columns=["customerID"])

    # Step 2: Fix TotalCharges — it's a number stored as text
    # First replace any hidden spaces " " with NaN so pandas can see them
    # Then convert the whole column from text to a proper number (float)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", float("nan"))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Step 3: Drop the rows where TotalCharges is now NaN
    # There are only 11 such rows out of 7043 — safe to remove them
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    after = len(df)
    print(f"✅ Dropped {before - after} rows with missing TotalCharges")

    # Step 4: Convert Churn from text to numbers
    # Yes → 1 (customer left), No → 0 (customer stayed)
    # Models need numbers, not words
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Step 5: Convert all remaining text columns into numbers
    # This is called One-Hot Encoding — it creates a new column for each
    # unique value in a text column
    # Example: gender becomes gender_Male and gender_Female (0 or 1)
    # drop_first=True removes one column to avoid redundancy
    # if gender_Male is 0 we already know it must be Female
    df = pd.get_dummies(df, drop_first=True)

    print(f"✅ Data prepared — {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\n--- Column list after preparation ---")
    print(df.columns.tolist())

    return df


def split_data(df):

    # Separate the data into:
    # X — the features (everything the model learns FROM)
    # y — the target (what the model is trying to PREDICT)
    # Think of X as the exam questions and y as the answer key

    # X is every column EXCEPT Churn
    X = df.drop(columns=["Churn"])

    # y is the Churn column only
    y = df["Churn"]

    print(f"\n✅ Features (X): {X.shape[1]} columns")
    print(f"✅ Target (y): {y.shape[0]} rows, {y.sum()} churned customers")

    return X, y


# Only runs when you execute this file directly with python data_prep.py
# If another file imports data_prep.py, this block is skipped
if __name__ == "__main__":

    # We need to load the data first before we can prepare it
    from data_loader import load_data
    df = load_data()

    if df is not None:
        df = prepare_data(df)
        X, y = split_data(df)