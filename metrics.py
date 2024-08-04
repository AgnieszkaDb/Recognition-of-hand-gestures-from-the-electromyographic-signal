import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import seaborn as sns

# conda install -c conda-forge imbalanced-learn

y_pred = np.load('history/y_pred.npy')
y_test = np.load('history/y_test.npy')

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
g_mean = geometric_mean_score(y_test, y_pred, average='weighted')

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("G-Mean:", g_mean)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of the baseline model')
    plt.savefig('conf_matrix_baseline.png')

plot_confusion_matrix(conf_matrix)