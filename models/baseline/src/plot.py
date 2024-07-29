import numpy as np
import matplotlib.pyplot as plt

# Load the metrics
train_loss = np.load('../history/train_loss.npy')
train_accuracy = np.load('../history/train_accuracy.npy')
val_loss = np.load('../history/val_loss.npy')
val_accuracy = np.load('../history/val_accuracy.npy')

# Plot loss
plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.savefig('../plots/loss_curves.png')
plt.show()

# Plot accuracy
plt.figure()
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.savefig('../plots/accuracy_curves.png')
plt.show()
