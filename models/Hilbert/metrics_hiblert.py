from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
# from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # To load .npy files
from pathlib import Path
import sys
import warnings
import json

warnings.filterwarnings("ignore")

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, model_name, file_name):
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix of the {model_name}")
    plt.savefig(f"metrics/{file_name}/cm_{file_name}.png")
    plt.close()

# Function to plot training and validation metrics
def plot_metric(train_metric, val_metric, ylabel, title, path, model_name):
    plt.figure()
    plt.plot(train_metric, label=f"Train {ylabel}")
    plt.plot(val_metric, label=f"Validation {ylabel}")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f"{title} Curves - {model_name}")
    plt.savefig(path)
    plt.close()

# Function to save metrics in JSON format
def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f)

def main():
    history_file = sys.argv[1]
    y_pred_file = sys.argv[2]  # Path to y_pred_voted.npy
    y_test_file = sys.argv[3]  # Path to y_test_voted.npy

    print(f"Processing file: {history_file}")

    # Load history data
    with open(history_file, "r") as f:
        history_data = json.load(f)

    # Extract relevant data from history_data
    model_name, train_loss, train_accuracy, val_loss, val_accuracy = (
        history_data[k]
        for k in [
            "model_name",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
    )

    # Load y_pred and y_test from .npy files
    y_pred = np.load(y_pred_file)
    y_test = np.load(y_test_file)

    # Print shapes to ensure correct loading
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Calculate metrics after voting
    metrics = {
        "acc": round(accuracy_score(y_test, y_pred), 4),
        "prec": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "g_mean": round(geometric_mean_score(y_test, y_pred, average="weighted"), 4),
    }

    # Create directories for saving results
    file_name = model_name.replace(' ', '-')
    save_dir = Path(f'metrics/{file_name}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the computed metrics
    save_metrics(metrics, save_dir / f"metrics_{file_name}.json")

    # Plot and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name, file_name)

    # Plot and save the loss and accuracy curves
    plot_metric(train_loss, val_loss, "Loss", "Loss", save_dir / f"loss_curves_{file_name}.png", model_name)
    plot_metric(train_accuracy, val_accuracy, "Accuracy", "Accuracy", save_dir / f"acc_curves_{file_name}.png", model_name)

if __name__ == "__main__":
    main()

