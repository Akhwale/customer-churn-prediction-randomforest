# setup.py
# Responsible for: downloading the dataset from Kaggle automatically
# This makes the project fully reproducible — no manual downloading needed

import os
import zipfile
from dotenv import load_dotenv  # reads key=value pairs from your .env file


def download_data():

    # Load the KAGGLE_USERNAME and KAGGLE_KEY from your .env file
    # This keeps your credentials out of your code
    load_dotenv()

    # Check if the dataset already exists in the data/ folder
    # If it does, skip the download entirely
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if os.path.exists(data_path):
        print("Dataset already exists, skipping download.")
        return

    # Use the Kaggle CLI to download the dataset
    # The CLI automatically reads KAGGLE_USERNAME and KAGGLE_KEY from environment
    # -d specifies the dataset, -p specifies the destination folder
    print("Downloading dataset from Kaggle...")
    os.system("kaggle datasets download -d blastchar/telco-customer-churn -p data/")

    # The Kaggle CLI downloads a .zip file, so we unzip it into data/
    zip_path = os.path.join("data", "telco-customer-churn.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("data/")

        # Clean up the .zip file after extracting — we only need the CSV
        os.remove(zip_path)
        print("Dataset downloaded and ready in data/")
    else:
        # If the zip wasn't created, the download failed
        # Most likely cause: wrong credentials in your .env file
        print("Download failed. Check your Kaggle credentials in your .env file.")


# Only run download_data() when this file is executed directly
# If another file imports setup.py, this block is skipped
if __name__ == "__main__":
    download_data()