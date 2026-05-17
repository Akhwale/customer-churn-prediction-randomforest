# data_loader.py
# Responsible for: loading the raw CSV file into pandas
# This is the only file in the project that knows where the data lives
# Every other file gets its data by calling load_data() from here

import pandas as pd
import os


def load_data():

    # Build the path to the CSV file inside the data/ folder
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # If the file doesn't exist, tell the user to run setup.py first
    # This prevents a confusing error message later in the pipeline
    if not os.path.exists(data_path):
        print("❌ Dataset not found. Please run python setup.py first.")
        return None

    # Load the CSV into a pandas DataFrame
    # A DataFrame is like an Excel spreadsheet but inside Python
    df = pd.read_csv(data_path)

    # Print a quick summary so we know the data loaded correctly
    # shape returns (number of rows, number of columns)
    print(f"✅ Data loaded successfully — {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def preview_data(df):

    # Show the first 5 rows so we can see what the data looks like
    print("\n--- First 5 rows ---")
    print(df.head())

    # Show column names and their data types
    # This tells us which columns are numbers vs text
    print("\n--- Column names and data types ---")
    print(df.dtypes)

    # Show how many missing values exist in each column
    # Missing values will need to be handled in data_prep.py
    print("\n--- Missing values per column ---")
    print(df.isnull().sum())


# Only runs when you execute this file directly with python data_loader.py
# If another file imports data_loader.py, this block is skipped
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        preview_data(df)