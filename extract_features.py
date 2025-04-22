import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #ideally would use gpu but given the project guidelines, will use cpu here
print("Using device:", device)

# Define project paths
image_folder = "/data/home/cos557/data/rothman/images"
label_csv = "/data/home/cos557/data/rothman/parsed_xray_files_log.csv"

# Load image filenames
all_image_files = sorted(os.listdir(image_folder))
print(f"Found {len(all_image_files)} images.")

# Load labels
df_labels = pd.read_csv(label_csv)
df_labels["patient_id"] = df_labels["patient_id"].astype(str)

# Map patient_id to revision_status
filename_to_label = dict(zip(df_labels["patient_id"], df_labels["revision_status"]))

# Define custom Dataset
class XrayDataset(Dataset):
    def __init__(self, image_folder, all_image_files, filename_to_label, transform=None):
        self.image_folder = image_folder
        self.image_files = all_image_files
        self.filename_to_label = filename_to_label
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_file)

        img_id = os.path.splitext(img_file)[0]
        patient_id = img_id.split("_")[0]

        if patient_id not in self.filename_to_label:
            label = 0  # default label
        else:
            label = self.filename_to_label[patient_id]
            if label == "No Revision Needed":
                label = 0
            else:
                label = 1

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the full dataset
full_dataset = XrayDataset(image_folder, all_image_files, filename_to_label, transform=transform)

# Train-test split (on indices)
n = len(full_dataset)
indices = list(range(n))
np.random.seed(42)
np.random.shuffle(indices)

split = int(np.floor(0.2 * n))
train_indices, test_indices = indices[split:], indices[:split]

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load pretrained VGG feature extractor
vgg = models.vgg16(pretrained=True).features.to(device)
vgg.eval()

for param in vgg.parameters():
    param.requires_grad = False

# Define feature extraction function
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

# Run feature extraction
X_train_features, y_train_labels = extract_features(train_loader)
X_test_features, y_test_labels = extract_features(test_loader)

print(f"Train feature shape: {X_train_features.shape}")
print(f"Test feature shape: {X_test_features.shape}")

# Save features
np.save("X_train_features.npy", X_train_features.numpy())
np.save("y_train_labels.npy", y_train_labels.numpy())
np.save("X_test_features.npy", X_test_features.numpy())
np.save("y_test_labels.npy", y_test_labels.numpy())

print("Feature extraction complete and saved.")
