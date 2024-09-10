from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import sys
import warnings
import json

warnings.filterwarnings("ignore")

# conda install -c conda-forge imbalanced-learn


def plot_confusion_matrix(cm, model_name, file_name):
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix of the {model_name}")
    plt.savefig(f"metrics/{file_name}/cm_{file_name}.png")
    plt.close()

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

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f)

def main():
    file = sys.argv[1]
    print(f"Processing file: {file}")
    
    with open(file, "r") as f:
        history_data = json.load(f)

    model_name, train_loss, train_accuracy, val_loss, val_accuracy, y_pred, y_test = (
        history_data[k]
        for k in [
            "model_name",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "y_pred",
            "y_test",
        ]
    )

    metrics = {
        "acc": round(accuracy_score(y_test, y_pred), 4),
        "prec": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall": round(recall_score(y_test, y_pred, average="macro"), 4),
        "f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "g_mean": round(geometric_mean_score(y_test, y_pred, average="macro"), 4),
    }

    file_name = model_name.replace(' ', '-')
    save_dir = Path(f'metrics/{file_name}')
    save_dir.mkdir(parents=True, exist_ok=True)

    
    save_metrics(metrics, save_dir / f"metrics_{file_name}.json")
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), model_name, file_name)
    plot_metric(train_loss, val_loss, "Loss", "Loss", save_dir / f"loss_curves_{file_name}.png", model_name)
    plot_metric(train_accuracy, val_accuracy, "Accuracy", "Accuracy", save_dir / f"acc_curves_{file_name}.png", model_name)


if __name__ == "__main__":
    main()

