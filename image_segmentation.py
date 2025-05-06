import pandas as pd
import os
import re
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch import nn, optim
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance
import cv2
from matplotlib.colors import ListedColormap
import torch.nn.functional as F


labels_df = pd.read_csv("LERA_Dataset/labels.csv", names = ["patient_ID", "image_type", "label"])

ankle_df = labels_df[labels_df["image_type"]=="XR ANKLE"]
ankle_patients = list(ankle_df["patient_ID"])

image_paths = []

for patient in ankle_patients:
    dir = f"LERA_Dataset/{str(patient)}/ST-1"
    image_paths.extend(os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(".png"))

class AnkleXrayDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert("L").resize((256, 256))
        
        image = ToTensor()(image)
        
        return image

def generate_pseudo_labels(image):
    image_np = image.squeeze().cpu().numpy()
    
    image_np = (image_np * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(image_np)
    
    img_filtered = cv2.bilateralFilter(img_enhanced, 5, 50, 50)
    
    _, bone_mask = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    if np.mean(bone_mask == 255) > 0.5: 
        bone_mask = cv2.bitwise_not(bone_mask)
    
    kernel = np.ones((3, 3), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel)
    
    y_coords, x_coords = np.mgrid[0:(img_filtered.shape[0]), 0:(img_filtered.shape[1])]
    
    x_norm = x_coords/(img_filtered.shape[1])
    y_norm = y_coords/(img_filtered.shape[0])
    intensity_norm = img_filtered/255.0
    
    bone_indices = np.where(bone_mask == 255)
    
    features = np.column_stack((
        x_norm[bone_indices]*1.0,         
        y_norm[bone_indices]*2.0,     
        intensity_norm[bone_indices]*1.5    
    ))
    
    segmentation = np.zeros(((img_filtered.shape[0]), (img_filtered.shape[1])), dtype=np.int32)
    bone_cluster_cnt = 4
    
    kmeans = KMeans(
        n_clusters=bone_cluster_cnt, 
        random_state=42, 
        n_init=10,
        algorithm='elkan' 
    )
    bone_labels = kmeans.fit_predict(features)
    
    segmentation[bone_indices] = bone_labels + 1
    
    segmentation_tensor = torch.tensor(segmentation).float().unsqueeze(0)
    
    return segmentation_tensor

def segment():
    all_labels = []
    
    for images in dataloader:
        pseudo_label = [generate_pseudo_labels(img.cpu()) for img in images]
        all_labels.append(pseudo_label)

    return all_labels


full_dataset = AnkleXrayDataset(image_paths)
dataloader = DataLoader(full_dataset, batch_size=1)

segmented_images = segment()

np.save("segmented_images.npy", np.array(segmented_images))