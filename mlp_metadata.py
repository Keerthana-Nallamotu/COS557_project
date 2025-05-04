from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import os
import torch

# Load metadata
metadata = "/data/home/cos557/data/rothman/TAR_Sheet_fo_stats_SGP_7_9_24_output4.csv"
df_metadata = pd.read_csv(metadata)

## extract metadata of interset
metadata_age = dict(zip(df_metadata["ID"], df_metadata["Age"]))
#metadata_race = dict(zip(df_metadata["ID"], df_metadata["Race"]))
### ID, age, race. age is just the number, race (0 is white, 1 is black, asian is 2, hispanic is 3, multirace is 4, other/NA is 5)

# Load revision labels
revision = "/data/home/cos557/data/rothman/parsed_xray_files_log.csv"
df_revision = pd.read_csv(revision)
df_revision["patient_id"] = df_revision["patient_id"].astype(str)
revision_labels = dict(zip(df_revision["patient_id"], df_revision["revision_status"]))

# Filter metadata for only patients who have revision data 
df_metadata = df_metadata[df_metadata["ID"].isin(df_revision["patient_id"])]

# train test split 
X_train, X_test, y_train, y_test = train_test_split(
    revision_labels, metadata_age, test_size=0.2, random_state=42)

# load feature extraction tables
#X_train = np.load("X_train_features.npy")
#y_train = np.load("y_train_labels.npy")
#X_test = np.load("X_test_features.npy")
#y_test = np.load("y_test_labels.npy")

print(f"Train feature shape: {X_train.shape}")
print(f"Test feature shape: {X_test.shape}")

# might need to play around with the hidden layer and max_iter hyperparameter to get better results
clf = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)
