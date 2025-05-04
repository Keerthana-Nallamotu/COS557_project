from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

# load feature extraction tables
X_train = np.load("updated_X_train_features.npy")
y_train = np.load("updated_y_train_labels.npy")
X_test = np.load("updated_X_test_features.npy")
y_test = np.load("updated_y_test_labels.npy")

similarity_matrix = cosine_similarity(X_test[:100], X_train[:100])
print(f"Max similarity: {np.max(similarity_matrix)}")

print(f"Train feature shape: {X_train.shape}")
print(f"Test feature shape: {X_test.shape}")
print(f"Train labels: {np.bincount(y_train.astype(int))}")
print(f"Test labels: {np.bincount(y_test.astype(int))}")

# MLP
clf = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Compute individual metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print all metrics
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# More detailed per-class report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Precision recall curve generation
plot = PrecisionRecallDisplay.frompredictions(y_test, y_pred)
plot.plot()
plt.show()
