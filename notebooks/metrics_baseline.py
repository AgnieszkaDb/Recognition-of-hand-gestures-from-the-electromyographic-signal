import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt

def geometric_mean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_sensitivity = cm.diagonal() / cm.sum(axis=1)
    g_mean = np.prod(per_class_sensitivity)**(1./len(per_class_sensitivity))
    return g_mean

data_path = 'processed_NinaDB1.csv'  
data = pd.read_csv(data_path)

# Prepare features and labels
X = data.drop('class', axis=1).values
y = data['class'].values - 1  # adjust labels from 1-52 to 0-51

# Split and scale the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert labels to categorical
y_test = to_categorical(y_test, num_classes=52)

# Load the saved model
model = load_model(r'..\models\baseline_model.h5')

# Predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# Evaluate using custom G-mean metric
g_mean = geometric_mean_score(y_true, y_pred)
print(f'G-mean: {g_mean:.3f}')

# Generate and visualize confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of the baseline model')
plt.savefig('..\plots\conf_matrix_basline.png')
plt.close()

# Display per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
# for i, accuracy in enumerate(per_class_accuracy, start=1):
#     print(f'Accuracy for class {i}: {accuracy:.3f}')

plt.figure(figsize=(10, 8))
plt.plot(range(1, 53), per_class_accuracy * 100)
plt.title('Accuracy of the baseline model')
plt.xlabel('Class')
plt.ylabel('Accuracy per class (%)')
plt.xlim(0, 52)
plt.ylim(0, 100)
plt.grid()
plt.savefig(r'..\plots\accuracy_baseline.png')