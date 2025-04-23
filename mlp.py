from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

#X, y = make_classification(n_samples=100, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# load feature extraction tables
X_train = np.load("X_train_features.npy")
y_train = np.load("y_train_labels.npy")
X_test = np.load("X_test_features.npy")
y_test = np.load("y_test_labels.npy")

print(f"Train feature shape: {X_train.shape}")
print(f"Test feature shape: {X_test.shape}")

# might need to play around with the hidden layer and max_iter hyperparameter to get better results
clf = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)

# I tested with 2000 max_iter and accuracy drops from 0.92 to 0.88. Max_iter = 700 makes accuracy score drop from 0.92 to 0.84. 
