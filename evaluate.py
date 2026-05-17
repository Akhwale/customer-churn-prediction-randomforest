# evaluate.py
# Responsible for: measuring how well our model actually performs
# Training a model is only half the job — we need to know if it's any good
# We use multiple metrics because no single number tells the full story

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,       # what % of predictions were correct overall
    precision_score,      # of all customers we predicted would churn, how many actually did
    recall_score,         # of all customers who actually churned, how many did we catch
    f1_score,             # balance between precision and recall
    confusion_matrix,     # table showing correct vs incorrect predictions
    classification_report # full summary of all metrics above
)


def evaluate_model(model, X_test, y_test):

    # Use the model to predict churn on the test set
    # Remember — the model has NEVER seen X_test before
    # This is like giving a student an exam they haven't studied for specifically
    y_pred = model.predict(X_test)

    # Calculate all our metrics by comparing predictions (y_pred)
    # against the real answers (y_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print("--- Model Performance ---")
    print(f"Accuracy:  {accuracy:.2%}")   # e.g. 0.82 prints as 82.00%
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")

    # Full breakdown by class (0 = stayed, 1 = churned)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    return y_pred


def plot_confusion_matrix(y_test, y_pred):

    # A confusion matrix shows us 4 things in a grid:
    # Top left:     True Negatives  — predicted stayed,  actually stayed  ✅
    # Top right:    False Positives — predicted churned,  actually stayed  ❌
    # Bottom left:  False Negatives — predicted stayed,  actually churned ❌
    # Bottom right: True Positives  — predicted churned, actually churned ✅
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix as a heatmap so it's easy to read
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,        # show the numbers inside each cell
        fmt="d",           # show as whole numbers not scientific notation
        cmap="Blues",      # blue color scheme
        xticklabels=["Stayed", "Churned"],   # column labels
        yticklabels=["Stayed", "Churned"]    # row labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")     # rows = what really happened
    plt.xlabel("Predicted")  # columns = what model guessed
    plt.tight_layout()       # makes sure nothing gets cut off
    plt.savefig("models/confusion_matrix.png")  # save it to models/ folder
    plt.show()
    print("✅ Confusion matrix saved to models/confusion_matrix.png")


def plot_feature_importance(model, X):

    # Feature importance tells us WHICH features the model relied on most
    # to make its predictions — this is incredibly valuable business insight
    # e.g. if tenure is the #1 feature, the business knows to focus on new customers

    # Get importance scores for every feature and pair them with column names
    importance = model.feature_importances_
    feature_names = X.columns.tolist()

    # Sort them from most important to least important
    import pandas as pd
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(10)  # top 10 only

    # Plot as a horizontal bar chart — easiest to read for many features
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="Blues_r")
    plt.title("Top 10 Most Important Features for Predicting Churn")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")  # save it to models/ folder
    plt.show()
    print("✅ Feature importance chart saved to models/feature_importance.png")


# Only runs when you execute this file directly with python evaluate.py
# If another file imports evaluate.py, this block is skipped
if __name__ == "__main__":

    # Load and prepare data, build model, then evaluate
    from data_loader import load_data
    from data_prep import prepare_data, split_data
    from model import build_model, load_model
    import os

    df = load_data()
    df = prepare_data(df)
    X, y = split_data(df)

    # Load saved model if it exists, otherwise train a new one
    if os.path.exists(os.path.join("models", "churn_model.pkl")):
        model = load_model()
        # We still need X_test and y_test to evaluate
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    else:
        model, _, X_test, _, y_test = build_model(X, y)

    # Run evaluation and plot charts
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, X)