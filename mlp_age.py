import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#### LOADING DATA ####

# Load metadata
metadata_path = "/data/home/cos557/data/rothman/TAR_Sheet_fo_stats_SGP_7_9_24_output4.csv"
df_metadata = pd.read_csv(metadata_path)

# Load revision labels
revisions_path = "/data/home/cos557/data/rothman/parsed_xray_files_log.csv"
df_revisions = pd.read_csv(revisions_path)
df_revisions["patient_id"] = df_revisions["patient_id"].astype(int)
df_revisions = df_revisions.sort_values(by=["patient_id", "postop_days"])
df_revisions = df_revisions.drop_duplicates(subset=["patient_id"], keep="last")
revision_labels = dict(zip(df_revisions["patient_id"], df_revisions["revision_status"]))
revision_labels = {pid: 1 if status > 0 else 0 for pid, status in revision_labels.items()}

# Filter metadata for patients who have revision data 
df_metadata = df_metadata[df_metadata["ID"].isin(df_revisions["patient_id"])]

# Training dataframe based on age
df = df_metadata[["ID", "Age"]].copy()
df["revision_status"] = df["ID"].map(revision_labels)

X = df[["Age"]].astype(float)
y = df["revision_status"].astype(int)

#### TRAIN/TEST SPLIT ####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### MLP CLASSIFIER ####
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
