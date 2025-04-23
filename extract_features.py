import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #ideally would use gpu but given the project guidelines, will use cpu here
print("Using device:", device)

# Define paths
image_folder = "/data/home/cos557/data/rothman/images"
label_csv = "/data/home/cos557/data/rothman/parsed_xray_files_log.csv"

# Load image filenames
all_image_files = sorted(os.listdir(image_folder))
print(f"Found {len(all_image_files)} images.")

# Load revision labels
df_labels = pd.read_csv(label_csv)
df_labels["patient_id"] = df_labels["patient_id"].astype(str)
filename_to_label = dict(zip(df_labels["patient_id"], df_labels["revision_status"]))

# Group images by patient_id
patient_to_images = defaultdict(list)
for img_file in all_image_files:
    patient_id = os.path.splitext(img_file)[0].split("_")[0]
    patient_to_images[patient_id].append(img_file)

# Patient-level train/test split
all_patients = sorted(patient_to_images.keys())
np.random.seed(42)
np.random.shuffle(all_patients)

split = int(len(all_patients) * 0.8)
train_patients = set(all_patients[:split])
test_patients = set(all_patients[split:])

train_files = [img for pid in train_patients for img in patient_to_images[pid]]
test_files = [img for pid in test_patients for img in patient_to_images[pid]]

# Custom PyTorch Dataset
class XrayDataset(Dataset):
    def __init__(self, image_folder, image_files, filename_to_label, transform=None):
        self.image_folder = image_folder
        self.image_files = image_files
        self.filename_to_label = filename_to_label
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_file)
        patient_id = os.path.splitext(img_file)[0].split("_")[0]

        label = self.filename_to_label.get(patient_id, "No Revision Needed")
        label = 0 if label == "No Revision Needed" else 1

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets and loaders
train_dataset = XrayDataset(image_folder, train_files, filename_to_label, transform)
test_dataset = XrayDataset(image_folder, test_files, filename_to_label, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load pretrained VGG16
vgg = models.vgg16(pretrained=True).features.to(device)
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False

# Feature extraction function
def extract_features(dataloader):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            features = vgg(images)
            features = torch.flatten(features, start_dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_features, all_labels

# Extract and save
X_train_features, y_train_labels = extract_features(train_loader)
X_test_features, y_test_labels = extract_features(test_loader)

np.save("X_train_features.npy", X_train_features.numpy())
np.save("y_train_labels.npy", y_train_labels.numpy())
np.save("X_test_features.npy", X_test_features.numpy())
np.save("y_test_labels.npy", y_test_labels.numpy())

print("Feature extraction complete and saved.")
print(f"Train feature shape: {X_train_features.shape}")
print(f"Test feature shape: {X_test_features.shape}")
