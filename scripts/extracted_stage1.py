# === CODE CELL 1 ===
# ============================================
# CELL 1: ENVIRONMENT SETUP
# ============================================

print("="*70)
print("SKIN CANCER DETECTION - COMBINED APPROACH")
print("Proper Preprocessing (with visualizations) + Training + Grad-CAM")
print("="*70)

# Install required packages
!pip install -q timm albumentations opencv-python-headless scikit-learn matplotlib seaborn tqdm

# Core libraries
import os
import sys
import gc
import warnings
import random
import json
import time
import copy
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import torch.nn.functional as F

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Create output directories
RESULTS_DIR = Path('/kaggle/working/stage1_results')
MODELS_DIR = RESULTS_DIR / 'models'
FIGURES_DIR = RESULTS_DIR / 'figures'
REPORTS_DIR = RESULTS_DIR / 'reports'
CACHE_DIR = RESULTS_DIR / 'cache'

for dir_path in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Results will be saved to: {RESULTS_DIR}")



# === CODE CELL 2 ===
# ============================================
# CELL 2: CONFIGURATION
# ============================================

class Config:
    # Image settings
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    
    # Training settings
    MAX_EPOCHS = 70       # Set maximum training ceiling to 70
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4    # Increased from 1e-5 to strongly combat overfitting
    PATIENCE = 15         # Stop early if validation doesn't improve for 7 epochs
    
    # Scheduler
    T_MAX = 50            # Updated to match 50 epochs for perfect cosine annealing decay
    ETA_MIN = 1e-6
    
    # Clinical targets
    MIN_SENSITIVITY = 0.90
    MIN_SPECIFICITY = 0.75
    
    # System
    GRADIENT_CLIP = 1.0
    
    # Data paths (update as needed)
    BASE_INPUT = Path('/kaggle/input/datasets/zohroufkhattak')
    TRAIN_IMAGES = BASE_INPUT / 'isic-2019-training-input' / 'ISIC_2019_Training_Input'
    TEST_IMAGES = BASE_INPUT / 'test-data-2019' / 'ISIC_2019_Test_Input'
    TRAIN_CSV = BASE_INPUT / 'isic-2019-training-groundtruth-csv' / 'ISIC_2019_Training_GroundTruth.csv'
    TEST_CSV = BASE_INPUT / 'isic-2019-test-groundtruth-csv' / 'ISIC_2019_Test_GroundTruth.csv'

config = Config()

print("="*70)
print("CONFIGURATION")
print("="*70)
print(f"Image size: {config.IMG_SIZE}×{config.IMG_SIZE}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Max epochs: {config.MAX_EPOCHS}")
print(f"Learning rate: {config.LEARNING_RATE}")
print(f"Weight decay: {config.WEIGHT_DECAY}")
print(f"Patience: {config.PATIENCE}")
print(f"Cosine scheduler T_max: {config.T_MAX}")




# === CODE CELL 3 ===
# ============================================
# CELL 3: LOAD DATA AND CREATE LABELS
# ============================================

print("="*70)
print("LOADING DATA AND CREATING LABELS")
print("="*70)

# Load CSV files
train_df = pd.read_csv(config.TRAIN_CSV)
test_df = pd.read_csv(config.TEST_CSV)

print(f"\n📊 Training CSV: {len(train_df):,} rows")
print(f"📊 Test CSV: {len(test_df):,} rows")

# Define class mappings for Stage 1
MALIGNANT_CLASSES = ['MEL', 'BCC', 'SCC', 'AK']
BENIGN_CLASSES = ['NV', 'BKL', 'DF', 'VASC']

def create_stage1_labels(df, name="Dataset"):
    """Create binary labels for Stage 1"""
    available_cols = df.columns.tolist()
    malignant_exist = [c for c in MALIGNANT_CLASSES if c in available_cols]
    benign_exist = [c for c in BENIGN_CLASSES if c in available_cols]
    
    print(f"\n📋 {name} - Available classes:")
    print(f"   MALIGNANT: {malignant_exist}")
    print(f"   BENIGN: {benign_exist}")
    
    # Create binary label
    df['is_malignant'] = df[malignant_exist].any(axis=1).astype(int)
    df['stage1_label'] = df['is_malignant']
    df['stage1_label_name'] = df['is_malignant'].map({1: 'MALIGNANT', 0: 'BENIGN'})
    
    # Statistics
    malignant_count = (df['is_malignant'] == 1).sum()
    benign_count = (df['is_malignant'] == 0).sum()
    
    print(f"\n📊 {name} Distribution:")
    print(f"   MALIGNANT: {malignant_count:6,} ({malignant_count/len(df)*100:.1f}%)")
    print(f"   BENIGN:    {benign_count:6,} ({benign_count/len(df)*100:.1f}%)")
    print(f"   Ratio (B:M): {benign_count/malignant_count:.2f}:1")
    
    return df

# Apply label creation
train_df = create_stage1_labels(train_df, "Training")
test_df = create_stage1_labels(test_df, "Test")

# Add image paths
train_df['image_path'] = train_df['image'].apply(lambda x: config.TRAIN_IMAGES / f"{x}.jpg")
train_df.loc[~train_df['image_path'].apply(lambda x: x.exists()), 'image_path'] = \
    train_df['image'].apply(lambda x: config.TRAIN_IMAGES / f"{x}.JPG")

test_df['image_path'] = test_df['image'].apply(lambda x: config.TEST_IMAGES / f"{x}.jpg")
test_df.loc[~test_df['image_path'].apply(lambda x: x.exists()), 'image_path'] = \
    test_df['image'].apply(lambda x: config.TEST_IMAGES / f"{x}.JPG")

# Filter to existing images
train_df = train_df[train_df['image_path'].apply(lambda x: x.exists())].reset_index(drop=True)
test_df = test_df[test_df['image_path'].apply(lambda x: x.exists())].reset_index(drop=True)

print(f"\n✅ Final datasets:")
print(f"   Training: {len(train_df):,} images")
print(f"   Test: {len(test_df):,} images (LOCKED)")



# === CODE CELL 4 ===
# ============================================
# CELL 4: SAMPLE IMAGES VISUALIZATION
# ============================================

print("="*70)
print("VISUALIZATION 2: Sample Images from Dataset")
print("="*70)

# Get sample images
benign_samples = train_df[train_df['stage1_label'] == 0].head(4)
malignant_samples = train_df[train_df['stage1_label'] == 1].head(4)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Sample Images: Benign vs Malignant', fontsize=14, fontweight='bold')

# Plot benign images
for idx, (_, row) in enumerate(benign_samples.iterrows()):
    try:
        img = cv2.imread(str(row['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"Benign\n{row['image']}", fontsize=9)
    except Exception as e:
        axes[0, idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    axes[0, idx].axis('off')

# Plot malignant images
for idx, (_, row) in enumerate(malignant_samples.iterrows()):
    try:
        img = cv2.imread(str(row['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, idx].imshow(img)
        axes[1, idx].set_title(f"Malignant\n{row['image']}", fontsize=9)
    except Exception as e:
        axes[1, idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_sample_images.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '02_sample_images.png'}")



# === CODE CELL 5 ===
# ============================================
# CELL 5: CLASS DISTRIBUTION VISUALIZATION
# ============================================

print("="*70)
print("VISUALIZATION: Class Distribution")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training set distribution
train_counts = train_df['stage1_label_name'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(train_counts.index, train_counts.values, color=colors, edgecolor='black')
axes[0].set_title(f'Training Set Distribution (n={len(train_df):,})', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
for i, (label, count) in enumerate(train_counts.items()):
    axes[0].text(i, count + 200, f'{count:,}\n({count/len(train_df)*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=10)

# Test set distribution
test_counts = test_df['stage1_label_name'].value_counts()
axes[1].bar(test_counts.index, test_counts.values, color=colors, edgecolor='black')
axes[1].set_title(f'Test Set Distribution (n={len(test_df):,})', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count')
for i, (label, count) in enumerate(test_counts.items()):
    axes[1].text(i, count + 100, f'{count:,}\n({count/len(test_df)*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '03_class_distribution.png'}")
print(f"\n📊 Imbalance Ratio (Benign:Malignant):")
print(f"   Training: {train_counts['BENIGN']/train_counts['MALIGNANT']:.2f}:1")
print(f"   Test: {test_counts['BENIGN']/test_counts['MALIGNANT']:.2f}:1")



# === CODE CELL 6 ===
# ============================================
# CELL 6: PREPROCESSING - HAIR REMOVAL
# ============================================

print("="*70)
print("PREPROCESSING STEP 1: HAIR REMOVAL")
print("="*70)

def remove_hair_with_vis(image, kernel_size=5, threshold=10):
    """
    Remove hair with intermediate steps for visualization.
    
    Steps:
    1. Convert to grayscale
    2. Apply Black Hat transform (detects dark hairs)
    3. Threshold to create binary mask
    4. Dilate mask to cover hair borders
    5. Inpaint to fill hair regions
    """
    
    # Step 1: Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Step 2: Black Hat transform
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Step 3: Threshold to create binary mask
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    
    # Step 4: Dilate mask to cover hair borders
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask_dilated = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
    
    # Step 5: Inpainting
    hair_removed = cv2.inpaint(image, hair_mask_dilated, 3, cv2.INPAINT_TELEA)
    
    return hair_removed, {
        'original': image,
        'gray': gray,
        'blackhat': blackhat,
        'hair_mask': hair_mask,
        'hair_mask_dilated': hair_mask_dilated,
        'result': hair_removed
    }

# Get a sample image
sample = train_df[train_df['stage1_label'] == 1].iloc[0]
sample_img = cv2.imread(str(sample['image_path']))
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

print(f"Sample image: {sample['image']} ({sample['stage1_label_name']})")
print(f"Original shape: {sample_img.shape}")

# Apply hair removal with visualization
hair_removed, steps = remove_hair_with_vis(sample_img)

# Visualize each step
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Hair Removal Pipeline - Step by Step', fontsize=14, fontweight='bold')

axes[0, 0].imshow(steps['original'])
axes[0, 0].set_title('1. Original Image\n(with hairs)', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(steps['gray'], cmap='gray')
axes[0, 1].set_title('2. Grayscale Conversion', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(steps['blackhat'], cmap='gray')
axes[0, 2].set_title('3. Black Hat Transform\n(detects dark hairs)', fontsize=11)
axes[0, 2].axis('off')

axes[1, 0].imshow(steps['hair_mask'], cmap='gray')
axes[1, 0].set_title('4. Hair Mask\n(threshold applied)', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(steps['hair_mask_dilated'], cmap='gray')
axes[1, 1].set_title('5. Dilated Mask\n(covers hair borders)', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(steps['result'])
axes[1, 2].set_title('6. Hair Removed\n(inpainting applied)', fontsize=11)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '04_hair_removal.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '04_hair_removal.png'}")
print("\n📝 Explanation:")
print("   - Black Hat transform highlights dark hairs on light skin")
print("   - Threshold creates a binary mask of hair locations")
print("   - Dilation expands mask to cover hair edges")
print("   - Inpainting fills masked regions using neighboring pixels")



# === CODE CELL 7 ===
# ============================================
# CELL 7: PREPROCESSING - LESION DETECTION
# ============================================

print("="*70)
print("PREPROCESSING STEP 2: LESION DETECTION")
print("="*70)

def detect_lesion_with_vis(image):
    """
    Detect lesion with intermediate steps for visualization.
    
    Steps:
    1. Convert to LAB color space
    2. Extract A channel (red-green contrast)
    3. Apply Gaussian blur
    4. Otsu thresholding
    5. Morphological closing (fill holes)
    6. Morphological opening (remove noise)
    7. Find largest contour
    """
    
    # Step 1: Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Step 2: Apply Gaussian blur
    a_blurred = cv2.GaussianBlur(a, (5, 5), 0)
    
    # Step 3: Otsu thresholding
    _, binary_mask = cv2.threshold(a_blurred, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 4: Morphological closing (fill holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Morphological opening (remove noise)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Step 6: Find largest contour
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        bbox = (x, y, w, h)
    else:
        h, w = image.shape[:2]
        bbox = (0, 0, w, h)
    
    return bbox, {
        'image': image,
        'lab': lab,
        'a_channel': a,
        'a_blurred': a_blurred,
        'binary_mask': binary_mask,
        'closed': closed,
        'cleaned': cleaned,
        'bbox': bbox
    }

# Apply to hair-removed image
bbox, lesion_steps = detect_lesion_with_vis(hair_removed)

print(f"Detected lesion bounding box: {bbox}")

# Visualize each step
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Lesion Detection Pipeline - Step by Step', fontsize=14, fontweight='bold')

axes[0, 0].imshow(lesion_steps['image'])
axes[0, 0].set_title('1. Hair-Removed Image\n(input to lesion detection)', fontsize=10)
axes[0, 0].axis('off')

axes[0, 1].imshow(lesion_steps['a_channel'], cmap='gray')
axes[0, 1].set_title('2. LAB - A Channel\n(highlights pigmented lesions)', fontsize=10)
axes[0, 1].axis('off')

axes[0, 2].imshow(lesion_steps['a_blurred'], cmap='gray')
axes[0, 2].set_title('3. Gaussian Blur\n(noise reduction)', fontsize=10)
axes[0, 2].axis('off')

axes[0, 3].imshow(lesion_steps['binary_mask'], cmap='gray')
axes[0, 3].set_title('4. Otsu Thresholding\n(automatic segmentation)', fontsize=10)
axes[0, 3].axis('off')

axes[1, 0].imshow(lesion_steps['closed'], cmap='gray')
axes[1, 0].set_title('5. Morphological Closing\n(fills holes)', fontsize=10)
axes[1, 0].axis('off')

axes[1, 1].imshow(lesion_steps['cleaned'], cmap='gray')
axes[1, 1].set_title('6. Morphological Opening\n(removes noise)', fontsize=10)
axes[1, 1].axis('off')

# Draw bounding box on original
bbox_img = lesion_steps['image'].copy()
x, y, w, h = bbox
cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
axes[1, 2].imshow(bbox_img)
axes[1, 2].set_title(f'7. Detected Lesion\nBounding Box: {w}×{h}', fontsize=10)
axes[1, 2].axis('off')

# Show mask overlay
overlay = lesion_steps['image'].copy()
overlay[lesion_steps['cleaned'] > 0] = overlay[lesion_steps['cleaned'] > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
axes[1, 3].imshow(overlay.astype(np.uint8))
axes[1, 3].set_title('8. Mask Overlay\n(green = detected lesion)', fontsize=10)
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_lesion_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '05_lesion_detection.png'}")
print(f"\n📝 Detection result:")
print(f"   Bounding box: x={x}, y={y}, w={w}, h={h}")



# === CODE CELL 8 ===
# ============================================
# CELL 8: PREPROCESSING - CROPPING AND RESIZING
# ============================================

print("="*70)
print("PREPROCESSING STEP 3: CROPPING AND RESIZING")
print("="*70)

def crop_and_resize_with_vis(image, bbox, padding=0.20, target_size=(224, 224)):
    """
    Crop lesion with padding and resize with visualization.
    """
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    
    # Calculate padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    # Calculate crop bounds
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)
    
    # Crop and resize
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return cropped, resized, (x1, y1, x2, y2)

# Apply cropping
cropped, resized, crop_coords = crop_and_resize_with_vis(hair_removed, bbox, padding=0.20)

print(f"Original: {hair_removed.shape}")
print(f"Cropped: {cropped.shape}")
print(f"Resized: {resized.shape}")

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Lesion Cropping and Resizing Pipeline', fontsize=14, fontweight='bold')

# Original with bounding boxes
bbox_img = hair_removed.copy()
x, y, w, h = bbox
x1, y1, x2, y2 = crop_coords

# Draw original lesion bbox (green)
cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
# Draw crop bbox (blue)
cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

axes[0].imshow(bbox_img)
axes[0].set_title('1. Original with Boxes\nGreen: Lesion, Blue: Crop Region', fontsize=11)
axes[0].axis('off')

axes[1].imshow(cropped)
axes[1].set_title(f'2. Cropped Lesion\n+20% padding\nSize: {cropped.shape[1]}×{cropped.shape[0]}', fontsize=11)
axes[1].axis('off')

axes[2].imshow(resized)
axes[2].set_title(f'3. Resized to {config.IMG_SIZE}×{config.IMG_SIZE}\n(Standard input size)', fontsize=11)
axes[2].axis('off')

# Show the pipeline summary
axes[3].axis('off')
summary_text = f"""
📊 Size Progression:

Original: {hair_removed.shape[1]}×{hair_removed.shape[0]}
Lesion: {w}×{h} pixels
Cropped: {cropped.shape[1]}×{cropped.shape[0]}
Final: {resized.shape[1]}×{resized.shape[0]}

Why crop?
• Focus on lesion only
• Remove background noise
• Standardize input size

Padding: 20% keeps context
"""
axes[3].text(0.1, 0.5, summary_text, transform=axes[3].transAxes, 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_cropping_resizing.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '06_cropping_resizing.png'}")



# === CODE CELL 9 ===
# ============================================
# CELL 9: COMPLETE PREPROCESSING PIPELINE
# ============================================

print("="*70)
print("COMPLETE PREPROCESSING PIPELINE - SUMMARY")
print("="*70)

class PreprocessingPipeline:
    """Complete preprocessing pipeline for skin lesion images"""
    
    @staticmethod
    def process(image_path, target_size=(224, 224)):
        """Complete preprocessing pipeline"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Hair removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
        hair_removed = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
        
        # Step 2: Lesion detection
        lab = cv2.cvtColor(hair_removed, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        a_blurred = cv2.GaussianBlur(a, (5, 5), 0)
        _, binary_mask = cv2.threshold(a_blurred, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
        else:
            h, w = image.shape[:2]
            x, y = 0, 0
        
        # Step 3: Cropping
        pad_w = int(w * 0.20)
        pad_h = int(h * 0.20)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        cropped = hair_removed[y1:y2, x1:x2]
        
        # Step 4: Resize
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return resized

# Test on multiple samples
samples = pd.concat([
    train_df[train_df['stage1_label'] == 0].head(2),
    train_df[train_df['stage1_label'] == 1].head(2)
])

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('Complete Preprocessing Pipeline - Multiple Examples', fontsize=16, fontweight='bold')

for idx, (_, row) in enumerate(samples.iterrows()):
    # Original
    original = cv2.imread(str(row['image_path']))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original_resized = cv2.resize(original, (224, 224))
    
    # Process through pipeline
    final = PreprocessingPipeline.process(row['image_path'])
    
    # Get intermediate steps for this image (simplified for visualization)
    image = cv2.imread(str(row['image_path']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Hair removal
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
    hair_removed = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
    hair_removed_resized = cv2.resize(hair_removed, (224, 224))
    
    # Lesion detection
    lab = cv2.cvtColor(hair_removed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a_blurred = cv2.GaussianBlur(a, (5, 5), 0)
    _, binary_mask = cv2.threshold(a_blurred, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned_resized = cv2.resize(cleaned.astype(np.uint8), (224, 224))
    
    # Display
    axes[idx, 0].imshow(original_resized)
    axes[idx, 0].set_title(f"{row['stage1_label_name']}\nOriginal", fontsize=10)
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(hair_removed_resized)
    axes[idx, 1].set_title("Hair Removed", fontsize=10)
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(cleaned_resized, cmap='gray')
    axes[idx, 2].set_title("Lesion Mask", fontsize=10)
    axes[idx, 2].axis('off')
    
    axes[idx, 3].imshow(final)
    axes[idx, 3].set_title(f"Final Processed\n224×224", fontsize=10)
    axes[idx, 3].axis('off')
    
    # Overlay mask on final
    overlay = final.copy()
    axes[idx, 4].imshow(overlay)
    axes[idx, 4].set_title("Ready for Model", fontsize=10)
    axes[idx, 4].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_complete_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '07_complete_pipeline.png'}")
print("\n✅ Preprocessing pipeline ready!")



# === CODE CELL 10 ===
# ============================================
# CELL 10: CACHE MANAGER
# ============================================

print("="*70)
print("CACHE MANAGER - PREPROCESSING ONCE")
print("="*70)

class CacheManager:
    """Manage preprocessing cache (uint8 NPY files by image ID)"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, image_id: str) -> Path:
        return self.cache_dir / f"{image_id}.npy"
    
    def is_cached(self, image_id: str) -> bool:
        return self.get_cache_path(image_id).exists()
    
    def save(self, image_id: str, image: np.ndarray):
        image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        np.save(self.get_cache_path(image_id), image_uint8)
    
    def load(self, image_id: str) -> np.ndarray:
        image_uint8 = np.load(self.get_cache_path(image_id))
        return image_uint8.astype(np.float32) / 255.0
    
    def preprocess_and_cache(self, df: pd.DataFrame, image_dir: Path, target_size=(224, 224)):
        print(f"\n📸 Preprocessing {len(df)} images...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row['image']
            
            if not self.is_cached(image_id):
                try:
                    image_path = image_dir / f"{image_id}.jpg"
                    if not image_path.exists():
                        image_path = image_dir / f"{image_id}.JPG"
                    
                    processed = PreprocessingPipeline.process(image_path, target_size)
                    self.save(image_id, processed)
                    
                except Exception as e:
                    print(f"  Warning: Could not process {image_id}: {e}")
        
        cached_count = len(list(self.cache_dir.glob('*.npy')))
        print(f"✅ Cached {cached_count} images")
        print(f"   Cache size: ~{cached_count * 224 * 224 * 3 / 1e9:.2f} GB")

# Initialize cache
cache = CacheManager(CACHE_DIR)

print(f"✅ Cache manager initialized at {CACHE_DIR}")

# Preprocess and cache all images
all_images = pd.concat([train_df[['image', 'image_path']], test_df[['image', 'image_path']]])
print(f"\nTotal images to cache: {len(all_images):,}")
print(f"Expected cache size: ~{len(all_images) * 224 * 224 * 3 / 1e9:.2f} GB")

cache.preprocess_and_cache(train_df, config.TRAIN_IMAGES, (config.IMG_SIZE, config.IMG_SIZE))
cache.preprocess_and_cache(test_df, config.TEST_IMAGES, (config.IMG_SIZE, config.IMG_SIZE))

print(f"\n✅ Caching complete! {len(list(CACHE_DIR.glob('*.npy')))} images cached")



# === CODE CELL 11 ===
# ============================================
# CELL 11: MANUAL DATA AUGMENTATION - COMPLETE WITH VISUALIZATIONS
# ============================================

print("="*70)
print("MANUAL DATA AUGMENTATION PIPELINE")
print("="*70)

import random

def manual_train_augmentations(image, return_steps=False):
    """
    Manual augmentation function for training.
    
    Augmentations applied randomly:
    1. Random crop with resize (scale 0.8-1.0) - 50%
    2. Horizontal flip - 50%
    3. Vertical flip - 50%
    4. Rotation (±45°) - 50%
    5. Color jitter (brightness/contrast) - 50%
    6. Gaussian noise - 30%
    """
    h, w = image.shape[:2]
    steps = {} if return_steps else None
    
    # Ensure image is uint8 (0-255 range) for augmentation
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if return_steps:
        steps['original'] = image.copy()
    
    # Step 1: Random crop with resize
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        y = random.randint(0, h - new_h) if new_h < h else 0
        x = random.randint(0, w - new_w) if new_w < w else 0
        image = image[y:y+new_h, x:x+new_w]
        image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        if return_steps:
            steps['random_crop'] = image.copy()
    else:
        image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        if return_steps:
            steps['random_crop'] = image.copy()
    
    # Step 2: Horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        if return_steps:
            steps['horizontal_flip'] = image.copy()
    elif return_steps:
        steps['horizontal_flip'] = image.copy()
    
    # Step 3: Vertical flip
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        if return_steps:
            steps['vertical_flip'] = image.copy()
    elif return_steps:
        steps['vertical_flip'] = image.copy()
    
    # Step 4: Rotation
    if random.random() > 0.5:
        angle = random.uniform(-45, 45)
        center = (config.IMG_SIZE // 2, config.IMG_SIZE // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (config.IMG_SIZE, config.IMG_SIZE))
        if return_steps:
            steps['rotation'] = image.copy()
    elif return_steps:
        steps['rotation'] = image.copy()
    
    # Step 5: Color jitter (brightness and contrast)
    if random.random() > 0.5:
        brightness = 1 + random.uniform(-0.2, 0.2)
        contrast = 1 + random.uniform(-0.2, 0.2)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50)
        if return_steps:
            steps['color_jitter'] = image.copy()
    elif return_steps:
        steps['color_jitter'] = image.copy()
    
    # Step 6: Gaussian noise (using numpy addition to avoid cv2.add issues)
    if random.random() > 0.7:  # 30% probability
        noise = np.random.normal(0, 25, image.shape).astype(np.int16)
        image = image.astype(np.int16) + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        if return_steps:
            steps['gaussian_noise'] = image.copy()
    elif return_steps:
        steps['gaussian_noise'] = image.copy()
    
    # Step 7: Convert to float32 [0,1] range for normalization
    image = image.astype(np.float32) / 255.0
    
    if return_steps:
        steps['final'] = image
        return image, steps
    
    return image

def manual_val_transforms(image):
    """Validation transforms (no augmentation)"""
    image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    return image

print("✅ Manual augmentation functions created")

# ============================================
# VISUALIZATION 1: Step-by-step augmentation
# ============================================

print("\n📊 VISUALIZATION 1: Step-by-Step Augmentation Pipeline")

# Get a sample preprocessed image
sample = train_df.iloc[0]
sample_img = PreprocessingPipeline.process(sample['image_path'])

print(f"Sample image: {sample['image']}")

# Run with step tracking
random.seed(42)
_, steps = manual_train_augmentations(sample_img, return_steps=True)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Manual Augmentation Pipeline - Step by Step', fontsize=14, fontweight='bold')

step_names = [
    ('original', '1. Original Image'),
    ('random_crop', '2. Random Crop + Resize'),
    ('horizontal_flip', '3. Horizontal Flip'),
    ('vertical_flip', '4. Vertical Flip'),
    ('rotation', '5. Rotation (±45°)'),
    ('color_jitter', '6. Color Jitter'),
    ('gaussian_noise', '7. Gaussian Noise'),
    ('final', '8. Final (Normalized)')
]

for idx, (step_name, title) in enumerate(step_names):
    row, col = idx // 4, idx % 4
    if step_name in steps:
        img = steps[step_name]
        if step_name == 'final':
            # Already normalized [0,1]
            axes[row, col].imshow(img)
        else:
            axes[row, col].imshow(img)
        axes[row, col].set_title(title, fontsize=10)
    else:
        axes[row, col].text(0.5, 0.5, 'Step skipped\n(random)', ha='center', va='center')
        axes[row, col].set_title(f'{title}\n(Not Applied)', fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_augmentation_steps.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '07_augmentation_steps.png'}")

# ============================================
# VISUALIZATION 2: Multiple augmentation examples
# ============================================

print("\n📊 VISUALIZATION 2: Multiple Augmentation Examples")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Manual Augmentations - Multiple Random Examples', fontsize=14, fontweight='bold')

# Show original
axes[0, 0].imshow(sample_img)
axes[0, 0].set_title('Original (Preprocessed)', fontsize=11)
axes[0, 0].axis('off')

# Generate 11 different augmented versions
for i in range(1, 12):
    row, col = i // 4, i % 4
    random.seed(i * 42)
    aug_img, _ = manual_train_augmentations(sample_img, return_steps=True)
    
    axes[row, col].imshow(aug_img)
    axes[row, col].set_title(f'Augmented Version #{i}', fontsize=11)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_augmentation_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '07_augmentation_examples.png'}")

# ============================================
# VISUALIZATION 3: Augmentation Statistics
# ============================================

print("\n📊 VISUALIZATION 3: Augmentation Statistics")

augmentations_applied = {
    'Random Crop': 0,
    'Horizontal Flip': 0,
    'Vertical Flip': 0,
    'Rotation': 0,
    'Color Jitter': 0,
    'Gaussian Noise': 0
}

num_tests = 500
for seed in range(num_tests):
    random.seed(seed)
    
    # Random crop
    if random.random() > 0.5:
        augmentations_applied['Random Crop'] += 1
    
    # Horizontal flip
    if random.random() > 0.5:
        augmentations_applied['Horizontal Flip'] += 1
    
    # Vertical flip
    if random.random() > 0.5:
        augmentations_applied['Vertical Flip'] += 1
    
    # Rotation
    if random.random() > 0.5:
        augmentations_applied['Rotation'] += 1
    
    # Color jitter
    if random.random() > 0.5:
        augmentations_applied['Color Jitter'] += 1
    
    # Gaussian noise (30% probability)
    if random.random() > 0.7:
        augmentations_applied['Gaussian Noise'] += 1

fig, ax = plt.subplots(figsize=(10, 6))
names = list(augmentations_applied.keys())
counts = [augmentations_applied[n] / num_tests * 100 for n in names]
bars = ax.bar(names, counts, color='steelblue', edgecolor='black')
ax.set_ylabel('Application Rate (%)')
ax.set_title(f'Augmentation Statistics ({num_tests} random tests)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)

for bar, pct in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{pct:.0f}%', ha='center', va='bottom', fontsize=11)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_augmentation_stats.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '07_augmentation_stats.png'}")

# ============================================
# VISUALIZATION 4: Training vs Validation comparison
# ============================================

print("\n📊 VISUALIZATION 4: Training vs Validation Comparison")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Training Augmentations vs Validation Transforms', fontsize=14, fontweight='bold')

# Row 1: Training augmentations (with augmentation)
for i in range(4):
    random.seed(i * 100)
    aug_img, _ = manual_train_augmentations(sample_img, return_steps=True)
    
    axes[0, i].imshow(aug_img)
    axes[0, i].set_title(f'Training (Augmented #{i+1})', fontsize=11)
    axes[0, i].axis('off')

# Row 2: Validation transforms (no augmentation)
for i in range(4):
    val_img = manual_val_transforms(sample_img)
    
    axes[1, i].imshow(val_img)
    axes[1, i].set_title('Validation (No Augmentation)', fontsize=11)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_train_vs_val.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '07_train_vs_val.png'}")

# ============================================
# VISUALIZATION 5: Augmentation Effect on Single Image
# ============================================

print("\n📊 VISUALIZATION 5: Augmentation Effect Comparison")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Effect of Different Augmentations', fontsize=14, fontweight='bold')

# Original
axes[0, 0].imshow(sample_img)
axes[0, 0].set_title('Original', fontsize=12)
axes[0, 0].axis('off')

# With specific augmentations (deterministic seeds)
aug_types = [
    ('Random Crop', 1),
    ('Horizontal Flip', 2),
    ('Rotation', 3),
    ('Color Jitter', 4),
    ('Gaussian Noise', 5)
]

for idx, (aug_name, seed_val) in enumerate(aug_types):
    random.seed(seed_val)
    # Force specific augmentation by setting random states
    # This is for visualization only
    aug_img, _ = manual_train_augmentations(sample_img, return_steps=True)
    
    row, col = (idx + 1) // 3, (idx + 1) % 3
    axes[row, col].imshow(aug_img)
    axes[row, col].set_title(aug_name, fontsize=12)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_augmentation_effects.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Figure saved: {FIGURES_DIR / '07_augmentation_effects.png'}")

print("\n" + "="*70)
print("📊 AUGMENTATION SUMMARY")
print("="*70)
print("""
Augmentation Types Applied:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Augmentation          Probability    Purpose                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Random Crop + Resize  50%            Scale invariance                      │
│  Horizontal Flip       50%            Mirror invariance                     │
│  Vertical Flip         50%            Mirror invariance                     │
│  Rotation (±45°)       50%            Rotation invariance                   │
│  Color Jitter          50%            Color/lighting robustness             │
│  Gaussian Noise        30%            Image quality robustness              │
└─────────────────────────────────────────────────────────────────────────────┘

FIGURES SAVED:
├── 07_augmentation_steps.png      - Step-by-step pipeline visualization
├── 07_augmentation_examples.png   - Multiple random augmentation examples
├── 07_augmentation_stats.png      - Augmentation application statistics
├── 07_train_vs_val.png            - Training vs Validation comparison
└── 07_augmentation_effects.png    - Effect of different augmentations
""")

print("\n✅ Manual augmentation pipeline ready with all visualizations!")



# === CODE CELL 12 ===
# ============================================
# CELL 12: DATASET CLASS
# ============================================

print("="*70)
print("DATASET CLASS WITH MANUAL AUGMENTATION")
print("="*70)

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, cache_dir, mode='train'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row['image']
        label = row['stage1_label']
        
        # Load from cache (uint8 0-255)
        cache_path = self.cache_dir / f"{image_id}.npy"
        image = np.load(cache_path)
        
        # Apply augmentations based on mode
        if self.mode == 'train':
            image = manual_train_augmentations(image)
        else:
            image = manual_val_transforms(image)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, torch.tensor(label, dtype=torch.float32)

print("✅ Dataset class defined")



# === CODE CELL 13 ===
# ============================================
# CELL 13: CREATE TRAIN/VALIDATION SPLIT (UPDATED)
# ============================================

print("="*70)
print("CREATING TRAIN/VALIDATION SPLIT - 30% VALIDATION")
print("="*70)

from sklearn.model_selection import train_test_split

# CHANGE THIS: test_size=0.30 (was 0.15)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.30,  # ← 30% for validation (more representative!)
    stratify=train_df['is_malignant'],
    random_state=42
)

train_data = train_df.iloc[train_idx].reset_index(drop=True)
val_data = train_df.iloc[val_idx].reset_index(drop=True)

print(f"\n📊 NEW Split sizes:")
print(f"   Training set:   {len(train_data):,} images ({len(train_data)/len(train_df)*100:.1f}%)")
print(f"   Validation set: {len(val_data):,} images ({len(val_data)/len(train_df)*100:.1f}%)")
print(f"   Test set:       {len(test_df):,} images (HOLDOUT)")

# Verify class balance
print(f"\n📊 Class distribution:")
for name, data in [('Train', train_data), ('Validation', val_data), ('Test', test_df)]:
    benign = (data['is_malignant'] == 0).sum()
    malignant = (data['is_malignant'] == 1).sum()
    print(f"   {name}: Benign={benign:,}, Malignant={malignant:,}")



# === CODE CELL 14 ===
# ============================================
# CELL 14: CLASS IMBALANCE HANDLING
# ============================================

print("="*70)
print("CLASS IMBALANCE HANDLING")
print("="*70)

# Get labels
labels = train_data['stage1_label'].values

# Calculate class counts
n_benign = (labels == 0).sum()
n_malignant = (labels == 1).sum()
n_total = len(labels)

print(f"\n📊 Class Distribution:")
print(f"   Benign: {n_benign} ({n_benign/n_total*100:.1f}%)")
print(f"   Malignant: {n_malignant} ({n_malignant/n_total*100:.1f}%)")

# Method 1: WeightedRandomSampler (for balanced batches)
class_weights_balanced = {0: 1.0/n_benign, 1: 1.0/n_malignant}
sample_weights = [class_weights_balanced[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Corrected Loss Function:
# We rely ONLY on WeightedRandomSampler to balance batches (50/50).
# Using both a sampler and a high pos_weight double-corrects class balance and biases the model,
# which causes poor generalization on unseen test data.
criterion = nn.BCEWithLogitsLoss()

print(f"\n✅ WeightedRandomSampler: balancing batches")
print(f"✅ Standard BCE loss initialized (relying on sampler for imbalance correction)")




# === CODE CELL 15 ===
# ============================================
# CELL 15: DATALOADERS
# ============================================

print("="*70)
print("CREATING DATALOADERS")
print("="*70)

# Create datasets
train_dataset = SkinLesionDataset(train_data, CACHE_DIR, mode='train')
val_dataset = SkinLesionDataset(val_data, CACHE_DIR, mode='val')
test_dataset = SkinLesionDataset(test_df, CACHE_DIR, mode='test')

BATCH_SIZE = config.BATCH_SIZE
NUM_WORKERS = config.NUM_WORKERS

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"✅ DataLoaders created")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# Verify batch balance
sample_batch, sample_labels = next(iter(train_loader))
n_mal_in_batch = sample_labels.sum().item()
print(f"\n📊 First batch verification:")
print(f"   Malignant: {n_mal_in_batch}/{BATCH_SIZE} ({n_mal_in_batch/BATCH_SIZE*100:.1f}%)")
print(f"   Dataset malignant %: {n_malignant/n_total*100:.1f}%")



# === CODE CELL 16 ===
# ============================================
# CELL 16: MODEL ARCHITECTURES
# ============================================

print("="*70)
print("MODEL ARCHITECTURES")
print("="*70)

class BinaryHead(nn.Module):
    """Custom classification head with increased regularization to combat overfitting"""
    def __init__(self, in_features, dropout=0.5): # Increased from 0.3 to 0.5 to prevent memorization
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class ResNet50Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = BinaryHead(in_features)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class EfficientNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class MobileNetV2Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features)
        
    def forward(self, x):
        return self.head(self.backbone(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n📊 Model Parameter Counts:")
for name, model_class in [('ResNet50', ResNet50Binary), 
                           ('EfficientNet', EfficientNetBinary), 
                           ('MobileNetV2', MobileNetV2Binary)]:
    model = model_class()
    params = count_parameters(model)
    print(f"   {name}: {params/1e6:.2f}M parameters")
    del model
    torch.cuda.empty_cache()

print("\n✅ Model architectures defined with stronger regularization (Dropout=0.5)")




# === CODE CELL 17 ===
# ============================================
# CELL 17: TRAINING FUNCTIONS
# ============================================

print("="*70)
print("TRAINING FUNCTIONS")
print("="*70)

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

print("✅ Training functions defined")



# === CODE CELL 18 ===
# ============================================
# CELL 18: TRAIN MODEL FUNCTION (No Checkpoints)
# ============================================

def train_model_simple(model_name, model, train_loader, val_loader, test_loader, 
                       criterion, device, epochs=100, lr=1e-4, patience=20):
    """Simple training pipeline without checkpoints"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_accuracy = 0
    best_epoch = 0
    patience_counter = 0
    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                                'val_precision', 'val_recall', 'val_f1', 'val_auc']}
    
    print('='*60)
    print(f'🚀 Training {model_name.upper()}')
    print(f'   Total epochs: {epochs} | Patience: {patience}')
    print('='*60)
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        print(f'Epoch {epoch+1:03d}/{epochs}  '
              f'Train Loss={train_loss:.4f} Acc={train_acc:.4f}  '
              f'Val Loss={val_metrics["loss"]:.4f} Acc={val_metrics["accuracy"]:.4f}  '
              f'F1={val_metrics["f1"]:.4f} AUC={val_metrics["auc"]:.4f}')
        
        # Check for improvement
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name}_stage1.pth')
            print(f'  ✅ Best model saved (val_acc={best_val_accuracy:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break
        
        print()
    
    training_time = time.time() - start_time
    
    # Load best weights and evaluate
    best_model_path = f'best_{model_name}_stage1.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"⚠️ Warning: Best model file {best_model_path} not found!")
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Backup best model
    backup_path = MODELS_DIR
    backup_path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, backup_path / f'best_{model_name}_stage1.pth')
    
    print('\n' + '='*60)
    print(f'✅ {model_name.upper()} Training Complete')
    print(f'   Best Val Accuracy: {best_val_accuracy:.4f} (epoch {best_epoch})')
    print(f'   Test Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'   Test Recall: {test_metrics["recall"]:.4f}')
    print(f'   Test AUC: {test_metrics["auc"]:.4f}')
    print(f'   Training time: {training_time/60:.1f} min')
    print('='*60)
    
    return history, test_metrics, best_val_accuracy, training_time/60

print("✅ train_model_simple function defined (no checkpoints)")



# === CODE CELL 19 ===
# ============================================
# CELL 19: GRAD-CAM IMPLEMENTATION
# ============================================

print("="*70)
print("GRAD-CAM IMPLEMENTATION")
print("="*70)

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, x):
        self.model.zero_grad()
        out = self.model(x)
        out.backward(torch.ones_like(out))
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, (224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def get_target_layer(model, model_name):
    """Get the last convolutional layer for each model"""
    if model_name == 'resnet50':
        return model.backbone.layer4[-1]
    elif model_name == 'mobilenetv2':
        return model.backbone.features[-1]
    elif model_name == 'efficientnet':
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def denormalize(tensor):
    """Denormalize image tensor for visualization"""
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

print("✅ Grad-CAM classes and functions defined")
print("   - GradCAM class ready")
print("   - get_target_layer() ready")
print("   - denormalize() ready")



# === CODE CELL 20 ===
# ============================================
# CELL: TRAIN EFFICIENTNET (ORIGINAL)
# ============================================

print("="*70)
print("TRAINING EFFICIENTNET - ORIGINAL")
print("="*70)

# Clear memory
torch.cuda.empty_cache()
gc.collect()

# Train EfficientNet with original settings
history_efficientnet, test_metrics_efficientnet, best_val_efficientnet, time_efficientnet = train_model_simple(
    model_name='efficientnet',
    model=EfficientNetBinary(),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    epochs=50,      # Original: 70 epochs
    lr=1e-4,        # Original: 1e-4 learning rate
    patience=15     # Original: 15 patience
)

# Save results
efficientnet_results = {
    'history': history_efficientnet,
    'test_metrics': test_metrics_efficientnet,
    'best_val_acc': best_val_efficientnet,
    'training_time': time_efficientnet
}

with open(REPORTS_DIR / 'efficientnet_results.pkl', 'wb') as f:
    pickle.dump(efficientnet_results, f)

print("\n" + "="*60)
print("✅ EFFICIENTNET TRAINING COMPLETE")
print("="*60)
print(f"   Best Val Acc: {best_val_efficientnet:.4f}")
print(f"   Test Acc: {test_metrics_efficientnet['accuracy']:.4f}")
print(f"   Test Recall: {test_metrics_efficientnet['recall']:.4f}")
print(f"   Test AUC: {test_metrics_efficientnet['auc']:.4f}")
print(f"   Time: {time_efficientnet:.1f} minutes")



# === CODE CELL 21 ===
# ============================================
# CELL 21: TRAIN MOBILENETV2
# ============================================

print("="*70)
print("TRAINING MOBILENETV2 - 100 EPOCHS")
print("="*70)

# Clear memory
torch.cuda.empty_cache()
gc.collect()

# Train MobileNetV2
history_mobilenet, test_metrics_mobilenet, best_val_mobilenet, time_mobilenet = train_model_simple(
    model_name='mobilenetv2',
    model=MobileNetV2Binary(),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    epochs=50,
    lr=1e-4,
    patience= 10
)

# Save results
mobilenet_results = {
    'history': history_mobilenet,
    'test_metrics': test_metrics_mobilenet,
    'best_val_acc': best_val_mobilenet,
    'training_time': time_mobilenet
}

with open(REPORTS_DIR / 'mobilenetv2_results.pkl', 'wb') as f:
    pickle.dump(mobilenet_results, f)

print(f"\n✅ MobileNetV2 results saved to {REPORTS_DIR / 'mobilenetv2_results.pkl'}")
print(f"   Best Val Acc: {best_val_mobilenet:.4f}")
print(f"   Test Acc: {test_metrics_mobilenet['accuracy']:.4f}")
print(f"   Test Recall: {test_metrics_mobilenet['recall']:.4f}")



# === CODE CELL 22 ===
# ============================================
# CELL 22: TRAIN RESNET50
# ============================================

print("="*70)
print("TRAINING RESNET50 - 100 EPOCHS")
print("="*70)

# Clear memory
torch.cuda.empty_cache()
gc.collect()

# Train ResNet50
history_resnet50, test_metrics_resnet50, best_val_resnet50, time_resnet50 = train_model_simple(
    model_name='resnet50',
    model=ResNet50Binary(),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    epochs=60,
    lr=1e-4,
    patience= 15
)

# Save results
resnet50_results = {
    'history': history_resnet50,
    'test_metrics': test_metrics_resnet50,
    'best_val_acc': best_val_resnet50,
    'training_time': time_resnet50
}

with open(REPORTS_DIR / 'resnet50_results.pkl', 'wb') as f:
    pickle.dump(resnet50_results, f)

print(f"\n✅ ResNet50 results saved to {REPORTS_DIR / 'resnet50_results.pkl'}")
print(f"   Best Val Acc: {best_val_resnet50:.4f}")
print(f"   Test Acc: {test_metrics_resnet50['accuracy']:.4f}")
print(f"   Test Recall: {test_metrics_resnet50['recall']:.4f}")



# === CODE CELL 23 ===
# ============================================
# CELL: EVALUATE ALL MODELS ON UNSEEN TEST DATA
# ============================================

print("="*70)
print("ALL MODELS - UNSEEN TEST DATA EVALUATION")
print("="*70)

from sklearn.model_selection import train_test_split

# First, split test set into two parts (once for all models)
print("\n🔄 Splitting test set into: validation-test (50%) + unseen-test (50%)")

test_val_idx, test_unseen_idx = train_test_split(
    np.arange(len(test_df)),
    test_size=0.5,
    stratify=test_df['stage1_label'],
    random_state=99
)

test_val_data = test_df.iloc[test_val_idx].reset_index(drop=True)
test_unseen_data = test_df.iloc[test_unseen_idx].reset_index(drop=True)

print(f"\n   Original test set: {len(test_df)} images")
print(f"   UNSEEN test set: {len(test_unseen_data)} images (never seen!)")

# Create dataset for unseen data (once)
unseen_dataset = SkinLesionDataset(test_unseen_data, CACHE_DIR, mode='test')
unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False, num_workers=2)

# ============================================
# 1. EVALUATE RESNET50
# ============================================

print("\n" + "="*60)
print("1. RESNET50 - UNSEEN TEST EVALUATION")
print("="*60)

resnet_path = MODELS_DIR / 'best_resnet50_stage1.pth'

if resnet_path.exists():
    print(f"\n📂 Loading ResNet50 from: {resnet_path}")
    model = ResNet50Binary().to(device)
    model.load_state_dict(torch.load(resnet_path, map_location=device))
    model.eval()
    print("✅ ResNet50 loaded successfully")
    
    # Evaluate on unseen data
    print(f"\n🔍 Evaluating on {len(test_unseen_data)} UNSEEN images...")
    resnet_unseen_metrics = evaluate(model, unseen_loader, criterion, device)
    
    print("\n📊 RESNET50 UNSEEN TEST RESULTS:")
    print(f"   Test Accuracy:  {resnet_unseen_metrics['accuracy']:.4f}")
    print(f"   Test Recall:    {resnet_unseen_metrics['recall']:.4f}")
    print(f"   Test Precision: {resnet_unseen_metrics['precision']:.4f}")
    print(f"   Test F1-Score:  {resnet_unseen_metrics['f1']:.4f}")
    print(f"   Test AUC:       {resnet_unseen_metrics['auc']:.4f}")
    
else:
    print(f"❌ ResNet50 model not found")
    resnet_unseen_metrics = None

# ============================================
# 2. EVALUATE EFFICIENTNET
# ============================================

print("\n" + "="*60)
print("2. EFFICIENTNET - UNSEEN TEST EVALUATION")
print("="*60)

efficientnet_path = MODELS_DIR / 'best_efficientnet_stage1.pth'

if efficientnet_path.exists():
    print(f"\n📂 Loading EfficientNet from: {efficientnet_path}")
    model = EfficientNetBinary().to(device)
    model.load_state_dict(torch.load(efficientnet_path, map_location=device))
    model.eval()
    print("✅ EfficientNet loaded successfully")
    
    # Evaluate on unseen data
    print(f"\n🔍 Evaluating on {len(test_unseen_data)} UNSEEN images...")
    efficientnet_unseen_metrics = evaluate(model, unseen_loader, criterion, device)
    
    print("\n📊 EFFICIENTNET UNSEEN TEST RESULTS:")
    print(f"   Test Accuracy:  {efficientnet_unseen_metrics['accuracy']:.4f}")
    print(f"   Test Recall:    {efficientnet_unseen_metrics['recall']:.4f}")
    print(f"   Test Precision: {efficientnet_unseen_metrics['precision']:.4f}")
    print(f"   Test F1-Score:  {efficientnet_unseen_metrics['f1']:.4f}")
    print(f"   Test AUC:       {efficientnet_unseen_metrics['auc']:.4f}")
    
else:
    print(f"❌ EfficientNet model not found")
    efficientnet_unseen_metrics = None

# ============================================
# 3. EVALUATE MOBILENETV2
# ============================================

print("\n" + "="*60)
print("3. MOBILENETV2 - UNSEEN TEST EVALUATION")
print("="*60)

mobilenet_path = MODELS_DIR / 'best_mobilenetv2_stage1.pth'

if mobilenet_path.exists():
    print(f"\n📂 Loading MobileNetV2 from: {mobilenet_path}")
    model = MobileNetV2Binary().to(device)
    model.load_state_dict(torch.load(mobilenet_path, map_location=device))
    model.eval()
    print("✅ MobileNetV2 loaded successfully")
    
    # Evaluate on unseen data
    print(f"\n🔍 Evaluating on {len(test_unseen_data)} UNSEEN images...")
    mobilenet_unseen_metrics = evaluate(model, unseen_loader, criterion, device)
    
    print("\n📊 MOBILENETV2 UNSEEN TEST RESULTS:")
    print(f"   Test Accuracy:  {mobilenet_unseen_metrics['accuracy']:.4f}")
    print(f"   Test Recall:    {mobilenet_unseen_metrics['recall']:.4f}")
    print(f"   Test Precision: {mobilenet_unseen_metrics['precision']:.4f}")
    print(f"   Test F1-Score:  {mobilenet_unseen_metrics['f1']:.4f}")
    print(f"   Test AUC:       {mobilenet_unseen_metrics['auc']:.4f}")
    
else:
    print(f"❌ MobileNetV2 model not found")
    mobilenet_unseen_metrics = None

# ============================================
# 4. COMPARISON TABLE
# ============================================

print("\n" + "="*70)
print("📊 ALL MODELS - UNSEEN TEST COMPARISON")
print("="*70)

print(f"\n{'Model':<15} {'Accuracy':<12} {'Recall':<12} {'Precision':<12} {'F1':<12} {'AUC':<12}")
print("-"*75)

if resnet_unseen_metrics:
    print(f"{'ResNet50':<15} {resnet_unseen_metrics['accuracy']:.4f}     {resnet_unseen_metrics['recall']:.4f}     {resnet_unseen_metrics['precision']:.4f}     {resnet_unseen_metrics['f1']:.4f}     {resnet_unseen_metrics['auc']:.4f}")

if efficientnet_unseen_metrics:
    print(f"{'EfficientNet':<15} {efficientnet_unseen_metrics['accuracy']:.4f}     {efficientnet_unseen_metrics['recall']:.4f}     {efficientnet_unseen_metrics['precision']:.4f}     {efficientnet_unseen_metrics['f1']:.4f}     {efficientnet_unseen_metrics['auc']:.4f}")

if mobilenet_unseen_metrics:
    print(f"{'MobileNetV2':<15} {mobilenet_unseen_metrics['accuracy']:.4f}     {mobilenet_unseen_metrics['recall']:.4f}     {mobilenet_unseen_metrics['precision']:.4f}     {mobilenet_unseen_metrics['f1']:.4f}     {mobilenet_unseen_metrics['auc']:.4f}")

print("-"*75)

# ============================================
# 5. FIND BEST MODEL
# ============================================

print("\n" + "="*70)
print("🏆 BEST MODEL ON UNSEEN TEST DATA")
print("="*70)

best_model = None
best_accuracy = 0

for model_name, metrics in [
    ('ResNet50', resnet_unseen_metrics),
    ('EfficientNet', efficientnet_unseen_metrics),
    ('MobileNetV2', mobilenet_unseen_metrics)
]:
    if metrics and metrics['accuracy'] > best_accuracy:
        best_accuracy = metrics['accuracy']
        best_model = model_name

if best_model:
    print(f"\n   🏆 BEST MODEL: {best_model}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")
    
    # Get the metrics for best model
    if best_model == 'ResNet50':
        best_metrics = resnet_unseen_metrics
    elif best_model == 'EfficientNet':
        best_metrics = efficientnet_unseen_metrics
    else:
        best_metrics = mobilenet_unseen_metrics
    
    print(f"   Test Recall: {best_metrics['recall']:.4f}")
    print(f"   Test AUC: {best_metrics['auc']:.4f}")

# ============================================
# 6. SAVE ALL RESULTS
# ============================================

print("\n" + "="*70)
print("💾 SAVING ALL RESULTS")
print("="*70)

all_unseen_results = {
    'ResNet50': resnet_unseen_metrics,
    'EfficientNet': efficientnet_unseen_metrics,
    'MobileNetV2': mobilenet_unseen_metrics,
    'unseen_test_size': len(test_unseen_data),
    'original_test_size': len(test_df)
}

with open(REPORTS_DIR / 'all_models_unseen_test_results.pkl', 'wb') as f:
    pickle.dump(all_unseen_results, f)

print(f"\n✅ All results saved to: {REPORTS_DIR / 'all_models_unseen_test_results.pkl'}")

# ============================================
# 7. SUMMARY
# ============================================

print("\n" + "="*70)
print("📊 UNSEEN TEST SUMMARY")
print("="*70)

print(f"\nTotal unseen test images: {len(test_unseen_data)}")
print(f"Original test set size: {len(test_df)}")
print(f"Split ratio: 50/50")

print("\n✅ Evaluation complete!")



# === CODE CELL 24 ===
# ============================================
# CELL: CREATE TEST COMPARISON FIGURE (USING UNSEEN TEST RESULTS)
# ============================================

print("="*70)
print("CREATING TEST COMPARISON FIGURE")
print("="*70)

# Load the unseen test results
unseen_results_path = REPORTS_DIR / 'all_models_unseen_test_results.pkl'

if unseen_results_path.exists():
    with open(unseen_results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"\n✅ Loaded unseen test results from: {unseen_results_path}")
    print(f"   Unseen test size: {all_results['unseen_test_size']} images")
    
    # Filter out None values and the metadata
    models_data = {}
    for model_name in ['ResNet50', 'EfficientNet', 'MobileNetV2']:
        if model_name in all_results and all_results[model_name] is not None:
            models_data[model_name] = all_results[model_name]
    
    if len(models_data) > 0:
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('All Models - Unseen Test Data Performance Comparison', fontsize=14, fontweight='bold')
        
        models = list(models_data.keys())
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        # Extract metrics
        accuracies = [models_data[m]['accuracy'] for m in models]
        recalls = [models_data[m]['recall'] for m in models]
        precisions = [models_data[m]['precision'] for m in models]
        f1_scores = [models_data[m]['f1'] for m in models]
        aucs = [models_data[m]['auc'] for m in models]
        
        x = np.arange(len(models))
        
        # Plot 1: Accuracy Bar Chart
        bars = axes[0, 0].bar(models, accuracies, color=colors[:len(models)], edgecolor='black')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Test Accuracy (Unseen Data)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{acc:.3f}', ha='center', fontsize=10)
        
        # Plot 2: AUC Bar Chart
        bars = axes[0, 1].bar(models, aucs, color=colors[:len(models)], edgecolor='black')
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].set_title('Test AUC (Unseen Data)')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for bar, auc in zip(bars, aucs):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{auc:.3f}', ha='center', fontsize=10)
        
        # Plot 3: Radar Chart
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
        angles += angles[:1]
        
        metrics_labels = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC']
        ax = plt.subplot(2, 3, 3, projection='polar')
        
        for idx, model in enumerate(models):
            values = [
                models_data[model]['accuracy'],
                models_data[model]['recall'],
                models_data[model]['precision'],
                models_data[model]['f1'],
                models_data[model]['auc']
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Radar Chart - Overall Performance (Unseen Data)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Plot 4: Recall vs Precision Scatter
        axes[1, 0].scatter(precisions, recalls, s=200, c=colors[:len(models)], alpha=0.6)
        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (precisions[i], recalls[i]), fontsize=10, ha='center')
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision-Recall Trade-off (Unseen Data)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 5: Clinical Target Check (Recall ≥ 90%)
        target_line = 0.90
        bars = axes[1, 1].bar(models, recalls, color=['#2ecc71' if r >= target_line else '#e74c3c' for r in recalls])
        axes[1, 1].axhline(y=target_line, color='blue', linestyle='--', linewidth=2, label='Clinical Target (90%)')
        axes[1, 1].set_ylabel('Recall (Sensitivity)')
        axes[1, 1].set_title('Clinical Target: Sensitivity ≥ 90% (Unseen Data)')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, recall in zip(bars, recalls):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{recall:.3f}', ha='center', fontsize=10)
        
        # Plot 6: Summary Table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        table_data = []
        for model in models:
            m = models_data[model]
            table_data.append([
                model,
                f"{m['accuracy']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['f1']:.3f}",
                f"{m['auc']:.3f}"
            ])
        
        table = axes[1, 2].table(cellText=table_data,
                                  colLabels=['Model', 'Acc', 'Recall', 'Prec', 'F1', 'AUC'],
                                  cellLoc='center',
                                  loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(FIGURES_DIR / '10_all_models_unseen_test_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Figure saved: {FIGURES_DIR / '10_all_models_unseen_test_comparison.png'}")
        
        # Print summary
        print("\n" + "="*70)
        print("UNSEEN TEST DATA PERFORMANCE SUMMARY")
        print("="*70)
        
        for model in models:
            m = models_data[model]
            print(f"\n{model.upper()}:")
            print(f"   Accuracy:  {m['accuracy']:.4f}")
            print(f"   Recall:    {m['recall']:.4f} {'✅' if m['recall'] >= 0.90 else '❌'}")
            print(f"   Precision: {m['precision']:.4f}")
            print(f"   F1-Score:  {m['f1']:.4f}")
            print(f"   AUC:       {m['auc']:.4f}")
    else:
        print("❌ No model data found in results")
else:
    print(f"❌ Results file not found: {unseen_results_path}")
    print("   Please run the unseen test evaluation first.")



# === CODE CELL 25 ===
# ============================================
# CELL: COMPARE ALL MODELS ON TEST DATA (UNSEEN TEST)
# ============================================

print("="*70)
print("COMPARING ALL MODELS - UNSEEN TEST DATA PERFORMANCE")
print("="*70)

# Load the unseen test results
unseen_results_path = REPORTS_DIR / 'all_models_unseen_test_results.pkl'

if unseen_results_path.exists():
    with open(unseen_results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"\n✅ Loaded unseen test results from: {unseen_results_path}")
    print(f"   Unseen test size: {all_results['unseen_test_size']} images")
    
    # Filter out metadata and None values
    models_data = {}
    for model_name in ['ResNet50', 'EfficientNet', 'MobileNetV2']:
        if model_name in all_results and all_results[model_name] is not None:
            models_data[model_name] = all_results[model_name]
    
    if len(models_data) > 0:
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in models_data.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'AUC-ROC': f"{metrics['auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n📊 UNSEEN TEST DATA PERFORMANCE COMPARISON:")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        # Find best model by different metrics
        best_by_accuracy = comparison_df.loc[comparison_df['Accuracy'].astype(float).idxmax(), 'Model']
        best_by_auc = comparison_df.loc[comparison_df['AUC-ROC'].astype(float).idxmax(), 'Model']
        best_by_recall = comparison_df.loc[comparison_df['Recall'].astype(float).idxmax(), 'Model']
        best_by_f1 = comparison_df.loc[comparison_df['F1-Score'].astype(float).idxmax(), 'Model']
        
        print(f"\n🏆 BEST MODELS ON UNSEEN TEST DATA:")
        print(f"   Highest Accuracy:  {best_by_accuracy} ({comparison_df.loc[comparison_df['Accuracy'].astype(float).idxmax(), 'Accuracy']})")
        print(f"   Highest AUC:       {best_by_auc} ({comparison_df.loc[comparison_df['AUC-ROC'].astype(float).idxmax(), 'AUC-ROC']})")
        print(f"   Highest Recall:    {best_by_recall} ({comparison_df.loc[comparison_df['Recall'].astype(float).idxmax(), 'Recall']})")
        print(f"   Highest F1-Score:  {best_by_f1} ({comparison_df.loc[comparison_df['F1-Score'].astype(float).idxmax(), 'F1-Score']})")
        
        # Overall best (using Recall as primary metric for clinical relevance)
        best_model = best_by_recall
        print(f"\n🎯 OVERALL BEST MODEL (by Recall/Sensitivity): {best_model}")
        print(f"   This model detects {comparison_df.loc[comparison_df['Recall'].astype(float).idxmax(), 'Recall']} of malignant cases")
        
        # Save comparison
        comparison_df.to_csv(REPORTS_DIR / 'unseen_test_data_comparison.csv', index=False)
        print(f"\n✅ Comparison saved to: {REPORTS_DIR / 'unseen_test_data_comparison.csv'}")
        
        # Print clinical readiness
        print("\n" + "="*70)
        print("🏥 CLINICAL READINESS CHECK")
        print("="*70)
        
        for model_name, metrics in models_data.items():
            recall = metrics['recall']
            if recall >= 0.90:
                status = "✅ CLINICALLY READY"
            elif recall >= 0.85:
                status = "⚠️ NEAR READY"
            elif recall >= 0.80:
                status = "⚠️ NEEDS IMPROVEMENT"
            else:
                status = "❌ NOT CLINICALLY READY"
            
            print(f"   {model_name}: {recall:.2%} recall - {status}")
        
    else:
        print("❌ No model data found in results")
else:
    print(f"❌ Results file not found: {unseen_results_path}")
    print("\n   Please run the unseen test evaluation first.")
    print("   Looking for file: all_models_unseen_test_results.pkl")



# === CODE CELL 26 ===
# ============================================
# CELL: VISUALIZE ALL MODELS ON UNSEEN TEST DATA
# ============================================

print("="*70)
print("VISUALIZING ALL MODELS - UNSEEN TEST DATA PERFORMANCE")
print("="*70)

# Load the unseen test results
unseen_results_path = REPORTS_DIR / 'all_models_unseen_test_results.pkl'

if unseen_results_path.exists():
    with open(unseen_results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"\n✅ Loaded unseen test results from: {unseen_results_path}")
    print(f"   Unseen test size: {all_results['unseen_test_size']} images")
    
    # Filter out metadata and None values
    models_data = {}
    for model_name in ['ResNet50', 'EfficientNet', 'MobileNetV2']:
        if model_name in all_results and all_results[model_name] is not None:
            models_data[model_name] = all_results[model_name]
    
    if len(models_data) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('All Models - Unseen Test Data Performance Comparison', fontsize=14, fontweight='bold')
        
        models = list(models_data.keys())
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        # Extract metrics
        accuracies = [models_data[m]['accuracy'] for m in models]
        recalls = [models_data[m]['recall'] for m in models]
        precisions = [models_data[m]['precision'] for m in models]
        f1_scores = [models_data[m]['f1'] for m in models]
        aucs = [models_data[m]['auc'] for m in models]
        
        x = np.arange(len(models))
        
        # Plot 1: Accuracy Bar Chart
        bars = axes[0, 0].bar(models, accuracies, color=colors[:len(models)], edgecolor='black')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Test Accuracy (Unseen Data)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{acc:.3f}', ha='center', fontsize=10)
        
        # Plot 2: AUC Bar Chart
        bars = axes[0, 1].bar(models, aucs, color=colors[:len(models)], edgecolor='black')
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].set_title('Test AUC (Unseen Data)')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for bar, auc in zip(bars, aucs):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{auc:.3f}', ha='center', fontsize=10)
        
        # Plot 3: Radar Chart
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
        angles += angles[:1]
        
        metrics_labels = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC']
        ax = plt.subplot(2, 3, 3, projection='polar')
        
        for idx, model in enumerate(models):
            values = [
                models_data[model]['accuracy'],
                models_data[model]['recall'],
                models_data[model]['precision'],
                models_data[model]['f1'],
                models_data[model]['auc']
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Radar Chart - Overall Performance (Unseen Data)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Plot 4: Recall vs Precision Scatter
        axes[1, 0].scatter(precisions, recalls, s=200, c=colors[:len(models)], alpha=0.6)
        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (precisions[i], recalls[i]), fontsize=10, ha='center')
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision-Recall Trade-off (Unseen Data)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 5: Clinical Target Check (Recall ≥ 90%)
        target_line = 0.90
        bars = axes[1, 1].bar(models, recalls, color=['#2ecc71' if r >= target_line else '#e74c3c' for r in recalls])
        axes[1, 1].axhline(y=target_line, color='blue', linestyle='--', linewidth=2, label='Clinical Target (90%)')
        axes[1, 1].set_ylabel('Recall (Sensitivity)')
        axes[1, 1].set_title('Clinical Target: Sensitivity ≥ 90% (Unseen Data)')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, recall in zip(bars, recalls):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{recall:.3f}', ha='center', fontsize=10)
        
        # Plot 6: Summary Table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        table_data = []
        for model in models:
            m = models_data[model]
            table_data.append([
                model,
                f"{m['accuracy']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['f1']:.3f}",
                f"{m['auc']:.3f}"
            ])
        
        table = axes[1, 2].table(cellText=table_data,
                                  colLabels=['Model', 'Acc', 'Recall', 'Prec', 'F1', 'AUC'],
                                  cellLoc='center',
                                  loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / '10_all_models_unseen_test_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Figure saved: {FIGURES_DIR / '10_all_models_unseen_test_comparison.png'}")
        
        # Print summary
        print("\n" + "="*70)
        print("UNSEEN TEST DATA PERFORMANCE SUMMARY")
        print("="*70)
        
        for model in models:
            m = models_data[model]
            print(f"\n{model.upper()}:")
            print(f"   Accuracy:  {m['accuracy']:.4f}")
            print(f"   Recall:    {m['recall']:.4f} {'✅' if m['recall'] >= 0.90 else '❌'}")
            print(f"   Precision: {m['precision']:.4f}")
            print(f"   F1-Score:  {m['f1']:.4f}")
            print(f"   AUC:       {m['auc']:.4f}")
    else:
        print("❌ No model data found in results")
else:
    print(f"❌ Results file not found: {unseen_results_path}")
    print("   Please run the unseen test evaluation first.")



# === CODE CELL 27 ===
# ============================================
# CELL: GRAD-CAM FOR ALL MODELS (COMPLETELY FIXED)
# ============================================

print("="*70)
print("GRAD-CAM VISUALIZATION FOR ALL MODELS")
print("="*70)

# Import torchvision models with a different name to avoid conflict
import torchvision.models as tv_models

# Redefine get_target_layer with EfficientNet fix
def get_target_layer_fixed(model, model_name):
    """Get the last convolutional layer for each model"""
    if model_name == 'resnet50':
        return model.backbone.layer4[-1]
    elif model_name == 'mobilenetv2':
        return model.backbone.features[-1]
    elif model_name == 'efficientnet':
        # For timm's EfficientNet - try different possible attribute names
        if hasattr(model.backbone, 'conv_head'):
            return model.backbone.conv_head
        elif hasattr(model.backbone, 'layers'):
            return model.backbone.layers[-1]
        elif hasattr(model.backbone, '_conv_head'):
            return model.backbone._conv_head
        elif hasattr(model.backbone, 'features'):
            return model.backbone.features[-1]
        else:
            # Fallback: get last module of the backbone
            return list(model.backbone.modules())[-10]
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Redefine model classes with proper imports
class ResNet50GradCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class EfficientNetGradCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class MobileNetV2GradCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.mobilenet_v2(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

# Check which model files actually exist
print("\n📂 Checking for model files...")
available_models = []

# Check for ResNet50
resnet_path = MODELS_DIR / 'best_resnet50_stage1.pth'
if resnet_path.exists():
    available_models.append(('ResNet50', 'resnet50', ResNet50GradCAM(), resnet_path))
    print(f"   ✅ Found ResNet50: {resnet_path}")
else:
    print(f"   ❌ ResNet50 not found")

# Check for EfficientNet
efficientnet_path = MODELS_DIR / 'best_efficientnet_stage1.pth'
if efficientnet_path.exists():
    available_models.append(('EfficientNet', 'efficientnet', EfficientNetGradCAM(), efficientnet_path))
    print(f"   ✅ Found EfficientNet: {efficientnet_path}")
else:
    print(f"   ❌ EfficientNet not found")

# Check for MobileNetV2
mobilenet_path = MODELS_DIR / 'best_mobilenetv2_stage1.pth'
if mobilenet_path.exists():
    available_models.append(('MobileNetV2', 'mobilenetv2', MobileNetV2GradCAM(), mobilenet_path))
    print(f"   ✅ Found MobileNetV2: {mobilenet_path}")
else:
    print(f"   ❌ MobileNetV2 not found")

if len(available_models) > 0:
    # Select test samples (2 benign, 2 malignant)
    benign_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == 0][:2]
    malignant_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == 1][:2]
    sample_indices = benign_indices + malignant_indices
    sample_labels = ['Benign', 'Benign', 'Malignant', 'Malignant']
    
    # Create figure with rows = number of available models
    fig, axes = plt.subplots(len(available_models), 4, figsize=(16, 4 * len(available_models)))
    fig.suptitle('Grad-CAM Comparison: All Models on Same Test Images', fontsize=14, fontweight='bold')
    
    # If only one model, axes needs to be 2D
    if len(available_models) == 1:
        axes = axes.reshape(1, -1)
    
    all_predictions = {}
    
    for row, (display_name, model_key, model, weight_file) in enumerate(available_models):
        print(f"\n📊 Processing {display_name}...")
        
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"   ✅ Loaded weights from {weight_file}")
        
        # Get target layer using FIXED function
        target_layer = get_target_layer_fixed(model, model_key)
        print(f"   Target layer: {target_layer.__class__.__name__}")
        grad_cam = GradCAM(model, target_layer)
        
        model_predictions = []
        
        for col, idx in enumerate(sample_indices):
            img_tensor, true_label = test_dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()
            
            pred_label = 'Malignant' if prob > 0.5 else 'Benign'
            is_correct = '✓' if pred_label == sample_labels[col] else '✗'
            model_predictions.append({
                'true_label': sample_labels[col],
                'pred_label': pred_label,
                'probability': prob,
                'correct': is_correct
            })
            
            # Generate Grad-CAM
            try:
                cam = grad_cam.generate(img_tensor)
                img_np = denormalize(img_tensor)
                
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                overlay = 0.55 * img_np + 0.45 * heatmap
                overlay = np.clip(overlay, 0, 1)
                
                axes[row, col].imshow(overlay)
            except Exception as e:
                print(f"   Error on sample {col}: {e}")
                axes[row, col].imshow(denormalize(img_tensor))
            
            # Add title
            color = 'green' if is_correct == '✓' else 'red'
            axes[row, col].set_title(
                f'True: {sample_labels[col]}\nPred: {pred_label} {is_correct} ({prob:.2f})',
                fontsize=9, color=color
            )
            axes[row, col].axis('off')
        
        all_predictions[display_name] = model_predictions
        axes[row, 0].set_ylabel(display_name, fontsize=11, fontweight='bold', rotation=90, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '11_gradcam_all_models.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Grad-CAM comparison saved to: {FIGURES_DIR / '11_gradcam_all_models.png'}")
    
    # Print prediction summary
    print("\n📊 PREDICTION SUMMARY ACROSS ALL MODELS:")
    print("="*80)
    for model_name, predictions in all_predictions.items():
        print(f"\n{model_name}:")
        for i, pred in enumerate(predictions):
            status = "✅" if pred['correct'] == '✓' else "❌"
            print(f"   Sample {i+1} ({pred['true_label']}): {status} Pred={pred['pred_label']} ({pred['probability']:.3f})")
else:
    print("❌ No model files found. Please train models first.")



# === CODE CELL 28 ===
# ============================================
# CELL 28: COMPLETE GRAD-CAM VISUALIZATION
# ============================================

print("="*70)
print("COMPLETE GRAD-CAM VISUALIZATION")
print("="*70)

# Import torchvision models properly
import torchvision.models as tv_models

# ============================================
# PART 1: Setup and Helper Functions
# ============================================

def get_target_layer_fixed(model, model_name):
    """Get the last convolutional layer for each model"""
    if model_name == 'resnet50':
        return model.backbone.layer4[-1]
    elif model_name == 'mobilenetv2':
        return model.backbone.features[-1]
    elif model_name == 'efficientnet':
        if hasattr(model.backbone, 'conv_head'):
            return model.backbone.conv_head
        elif hasattr(model.backbone, '_conv_head'):
            return model.backbone._conv_head
        else:
            return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Redefine model classes with proper imports
class ResNet50BinaryForCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class EfficientNetBinaryForCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class MobileNetV2BinaryForCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.mobilenet_v2(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

def create_gradcam_visualization(model, model_name, image_tensor, true_label, sample_id):
    """
    Create comprehensive Grad-CAM visualization for a single image
    """
    # Get target layer using fixed function
    target_layer = get_target_layer_fixed(model, model_name)
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
    pred_label = 'Malignant' if prob > 0.5 else 'Benign'
    is_correct = pred_label == true_label
    
    # Generate heatmap
    cam = grad_cam.generate(image_tensor)
    
    # Get original image
    img_np = denormalize(image_tensor)
    
    # Create different visualizations
    # 1. Heatmap only
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # 2. Light overlay (30% heatmap)
    overlay_light = 0.7 * img_np + 0.3 * heatmap
    
    # 3. Medium overlay (50% heatmap)
    overlay_medium = 0.5 * img_np + 0.5 * heatmap
    
    # 4. Heavy overlay (70% heatmap)
    overlay_heavy = 0.3 * img_np + 0.7 * heatmap
    
    # 5. Contour overlay (edge detection on heatmap)
    cam_uint8 = np.uint8(255 * cam)
    contours, _ = cv2.findContours(cam_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_np.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
    
    return {
        'original': img_np,
        'heatmap': heatmap,
        'overlay_light': overlay_light,
        'overlay_medium': overlay_medium,
        'overlay_heavy': overlay_heavy,
        'contour': contour_img,
        'cam': cam,
        'pred_label': pred_label,
        'probability': prob,
        'is_correct': is_correct
    }

# ============================================
# PART 2: Select Test Samples
# ============================================

# Select diverse test samples
benign_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == 0][:3]
malignant_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == 1][:3]
sample_indices = benign_indices + malignant_indices
sample_true_labels = ['Benign', 'Benign', 'Benign', 'Malignant', 'Malignant', 'Malignant']
sample_descriptions = [
    'Typical benign nevus',
    'Benign with regular borders',
    'Small uniform benign lesion',
    'Irregular malignant melanoma',
    'Malignant with asymmetry',
    'Large malignant lesion'
]

# ============================================
# PART 3: Generate Grad-CAM for ALL Models
# ============================================

# Models to visualize - using new model classes
models_to_viz = [
    ('ResNet50', 'resnet50', ResNet50BinaryForCAM(), MODELS_DIR / 'best_resnet50_stage1.pth'),
    ('EfficientNet', 'efficientnet', EfficientNetBinaryForCAM(), MODELS_DIR / 'best_efficientnet_stage1.pth'),
    ('MobileNetV2', 'mobilenetv2', MobileNetV2BinaryForCAM(), MODELS_DIR / 'best_mobilenetv2_stage1.pth')
]

# Store all results
all_gradcam_results = {}

for display_name, model_key, model, weight_file in models_to_viz:
    print(f"\n📊 Processing {display_name}...")
    
    if weight_file.exists():
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"   ✅ Loaded: {weight_file.name}")
        
        model_results = []
        
        for idx, (sample_idx, true_label, description) in enumerate(zip(sample_indices, sample_true_labels, sample_descriptions)):
            img_tensor, _ = test_dataset[sample_idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Generate visualization
            viz = create_gradcam_visualization(model, model_key, img_tensor, true_label, idx)
            viz['description'] = description
            viz['sample_idx'] = sample_idx
            model_results.append(viz)
        
        all_gradcam_results[display_name] = model_results
    else:
        print(f"   ❌ Weight file not found: {weight_file}")

print(f"\n✅ Grad-CAM visualizations created for {len(all_gradcam_results)} models")

# ============================================
# PART 4: Visualize Results
# ============================================

if len(all_gradcam_results) > 0:
    # Create a figure with side-by-side comparison
    fig, axes = plt.subplots(len(all_gradcam_results), 6, figsize=(18, 4 * len(all_gradcam_results)))
    fig.suptitle('Grad-CAM Visualizations - All Models', fontsize=16, fontweight='bold')
    
    for row, (model_name, results) in enumerate(all_gradcam_results.items()):
        for col, viz in enumerate(results[:6]):  # Show up to 6 samples
            axes[row, col].imshow(viz['overlay_medium'])
            axes[row, col].set_title(f"{viz['pred_label']} ({viz['probability']:.2f})\n{viz['description'][:20]}", fontsize=8)
            axes[row, col].axis('off')
        
        axes[row, 0].set_ylabel(model_name, fontsize=12, fontweight='bold', rotation=90, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '12_gradcam_complete.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Grad-CAM figure saved to: {FIGURES_DIR / '12_gradcam_complete.png'}")
else:
    print("❌ No Grad-CAM results generated")



# === CODE CELL 29 ===
# ============================================
# CELL: VISUALIZE TRAINING HISTORIES - ALL MODELS
# ============================================

print("="*70)
print("TRAINING HISTORIES VISUALIZATION - ALL MODELS")
print("="*70)

# Load saved results for all models
results_dir = REPORTS_DIR

# Dictionary to store histories
all_histories = {}

# Load ResNet50 history
resnet_path = results_dir / 'resnet50_results.pkl'
if resnet_path.exists():
    with open(resnet_path, 'rb') as f:
        resnet_data = pickle.load(f)
        all_histories['ResNet50'] = resnet_data['history']
        print(f"✅ Loaded ResNet50 training history")
    # Also load test metrics
    resnet_test = resnet_data['test_metrics']
else:
    print(f"❌ ResNet50 results not found")

# Load EfficientNet history
efficientnet_path = results_dir / 'efficientnet_results.pkl'
if efficientnet_path.exists():
    with open(efficientnet_path, 'rb') as f:
        efficientnet_data = pickle.load(f)
        all_histories['EfficientNet'] = efficientnet_data['history']
        print(f"✅ Loaded EfficientNet training history")
    efficientnet_test = efficientnet_data['test_metrics']
else:
    print(f"❌ EfficientNet results not found")

# Load MobileNetV2 history
mobilenet_path = results_dir / 'mobilenetv2_results.pkl'
if mobilenet_path.exists():
    with open(mobilenet_path, 'rb') as f:
        mobilenet_data = pickle.load(f)
        all_histories['MobileNetV2'] = mobilenet_data['history']
        print(f"✅ Loaded MobileNetV2 training history")
    mobilenet_test = mobilenet_data['test_metrics']
else:
    print(f"❌ MobileNetV2 results not found")

if len(all_histories) > 0:
    # Create comprehensive training history plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History Comparison: All Models', fontsize=16, fontweight='bold')
    
    colors = {'ResNet50': '#2ecc71', 'EfficientNet': '#3498db', 'MobileNetV2': '#e74c3c'}
    line_styles = {'ResNet50': '-', 'EfficientNet': '-', 'MobileNetV2': '-'}
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for model_name, history in all_histories.items():
        if 'train_loss' in history and len(history['train_loss']) > 0:
            ax1.plot(history['train_loss'], label=f'{model_name}', 
                    color=colors.get(model_name, 'gray'), linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for model_name, history in all_histories.items():
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax2.plot(history['val_loss'], label=f'{model_name}', 
                    color=colors.get(model_name, 'gray'), linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[0, 2]
    for model_name, history in all_histories.items():
        if 'train_acc' in history and len(history['train_acc']) > 0:
            ax3.plot(history['train_acc'], label=f'{model_name}', 
                    color=colors.get(model_name, 'gray'), linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax4 = axes[1, 0]
    for model_name, history in all_histories.items():
        if 'val_acc' in history and len(history['val_acc']) > 0:
            ax4.plot(history['val_acc'], label=f'{model_name}', 
                    color=colors.get(model_name, 'gray'), linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.90, color='red', linestyle='--', linewidth=1, label='90% Target')
    
    # Plot 5: Validation AUC
    ax5 = axes[1, 1]
    for model_name, history in all_histories.items():
        if 'val_auc' in history and len(history['val_auc']) > 0:
            ax5.plot(history['val_auc'], label=f'{model_name}', 
                    color=colors.get(model_name, 'gray'), linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('AUC')
    ax5.set_title('Validation AUC Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Validation F1-Score
    ax6 = axes[1, 2]
    for model_name, history in all_histories.items():
        if 'val_f1' in history and len(history['val_f1']) > 0:
            ax6.plot(history['val_f1'], label=f'{model_name}', 
                    color=colors.get(model_name, 'gray'), linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('F1-Score')
    ax6.set_title('Validation F1-Score Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '13_training_histories_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Training histories figure saved: {FIGURES_DIR / '13_training_histories_comparison.png'}")
    
    # ============================================
    # Also create a summary table of best metrics
    # ============================================
    
    print("\n" + "="*70)
    print("📊 BEST VALIDATION METRICS SUMMARY")
    print("="*70)
    
    summary_data = []
    for model_name, history in all_histories.items():
        if len(history['val_acc']) > 0:
            best_val_acc = max(history['val_acc'])
            best_val_acc_epoch = history['val_acc'].index(best_val_acc) + 1
            best_val_auc = max(history['val_auc']) if len(history['val_auc']) > 0 else 0
            best_val_f1 = max(history['val_f1']) if len(history['val_f1']) > 0 else 0
            final_val_loss = history['val_loss'][-1] if len(history['val_loss']) > 0 else 0
            
            summary_data.append({
                'Model': model_name,
                'Best Val Acc': f"{best_val_acc:.4f} (Epoch {best_val_acc_epoch})",
                'Best Val AUC': f"{best_val_auc:.4f}",
                'Best Val F1': f"{best_val_f1:.4f}",
                'Final Val Loss': f"{final_val_loss:.4f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(REPORTS_DIR / 'training_history_summary.csv', index=False)
    print(f"\n✅ Summary saved to: {REPORTS_DIR / 'training_history_summary.csv'}")
    
else:
    print("❌ No training histories found. Please run training first.")

# ============================================
# Optional: Individual Model Training Plots
# ============================================

print("\n" + "="*70)
print("INDIVIDUAL MODEL TRAINING PLOTS")
print("="*70)

for model_name, history in all_histories.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Target')
    
    # AUC
    if 'val_auc' in history and len(history['val_auc']) > 0:
        axes[2].plot(history['val_auc'], label='Val AUC', color='purple', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('AUC Curve')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'14_{model_name.lower()}_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved: {FIGURES_DIR / f'14_{model_name.lower()}_training_history.png'}")

print("\n" + "="*70)
print("✅ ALL TRAINING HISTORY VISUALIZATIONS COMPLETE")
print("="*70)



# === CODE CELL 30 ===
# ============================================
# CELL: CONFUSION MATRICES - ALL MODELS
# ============================================

print("="*70)
print("CONFUSION MATRICES FOR ALL MODELS")
print("="*70)

import torchvision.models as tv_models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Redefine model classes for evaluation
class ResNet50Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class EfficientNetEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

class MobileNetV2Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.mobilenet_v2(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
        
    def forward(self, x):
        return self.head(self.backbone(x))

# Load unseen test data (create loader if not exists)
if 'unseen_loader' not in dir():
    from sklearn.model_selection import train_test_split
    
    # Split test set into two parts
    test_val_idx, test_unseen_idx = train_test_split(
        np.arange(len(test_df)),
        test_size=0.5,
        stratify=test_df['stage1_label'],
        random_state=99
    )
    test_unseen_data = test_df.iloc[test_unseen_idx].reset_index(drop=True)
    unseen_dataset = SkinLesionDataset(test_unseen_data, CACHE_DIR, mode='test')
    unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"✅ Created unseen test loader with {len(test_unseen_data)} images")

# Models to evaluate
models_to_eval = [
    ('ResNet50', ResNet50Eval(), MODELS_DIR / 'best_resnet50_stage1.pth'),
    ('EfficientNet', EfficientNetEval(), MODELS_DIR / 'best_efficientnet_stage1.pth'),
    ('MobileNetV2', MobileNetV2Eval(), MODELS_DIR / 'best_mobilenetv2_stage1.pth')
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices - All Models (Unseen Test Data)', fontsize=14, fontweight='bold')

for idx, (model_name, model, weight_path) in enumerate(models_to_eval):
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✅ Loaded {model_name}")
        
        # Get predictions for unseen test data
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in unseen_loader:
                images = images.to(device)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
        disp.plot(ax=axes[idx], values_format='d', cmap='Blues')
        axes[idx].set_title(f'{model_name}')
        
        # Print numbers
        tn, fp, fn, tp = cm.ravel()
        print(f"   {model_name}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    else:
        print(f"❌ {model_name} weights not found: {weight_path}")
        axes[idx].text(0.5, 0.5, f'{model_name}\nNot Found', ha='center', va='center')
        axes[idx].set_title(f'{model_name}')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '15_confusion_matrices_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Saved: {FIGURES_DIR / '15_confusion_matrices_all_models.png'}")



# === CODE CELL 31 ===
# ============================================
# CELL: ROC CURVES - ALL MODELS
# ============================================

print("="*70)
print("ROC CURVES FOR ALL MODELS")
print("="*70)

from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

colors = {'ResNet50': '#2ecc71', 'EfficientNet': '#3498db', 'MobileNetV2': '#e74c3c'}
roc_data = []

for model_name, model, weight_path in models_to_eval:
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Get probabilities
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in unseen_loader:
                images = images.to(device)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        roc_data.append({'Model': model_name, 'AUC': roc_auc})
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
                color=colors[model_name], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models (Unseen Test Data)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '16_roc_curves_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {FIGURES_DIR / '16_roc_curves_all_models.png'}")

# Print AUC summary
print("\n📊 AUC SUMMARY:")
roc_df = pd.DataFrame(roc_data)
print(roc_df.to_string(index=False))



# === CODE CELL 32 ===
# ============================================
# CELL: MODEL ARCHITECTURE SUMMARY
# ============================================

print("="*70)
print("MODEL ARCHITECTURE COMPARISON")
print("="*70)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Use the eval model classes
model_info = []

# ResNet50
resnet_model = ResNet50Eval()
resnet_params = count_parameters(resnet_model)
model_info.append({
    'Model': 'ResNet50',
    'Parameters (M)': f'{resnet_params/1e6:.2f}',
    'Depth': '50 layers',
    'Input Size': '224×224'
})

# EfficientNet
efficient_model = EfficientNetEval()
efficient_params = count_parameters(efficient_model)
model_info.append({
    'Model': 'EfficientNet-B0',
    'Parameters (M)': f'{efficient_params/1e6:.2f}',
    'Depth': '~16 layers',
    'Input Size': '224×224'
})

# MobileNetV2
mobile_model = MobileNetV2Eval()
mobile_params = count_parameters(mobile_model)
model_info.append({
    'Model': 'MobileNetV2',
    'Parameters (M)': f'{mobile_params/1e6:.2f}',
    'Depth': '~53 layers',
    'Input Size': '224×224'
})

model_df = pd.DataFrame(model_info)
print("\n📊 MODEL ARCHITECTURE COMPARISON:")
print(model_df.to_string(index=False))

# Add best test accuracy from your results
print("\n📊 PERFORMANCE ON UNSEEN TEST DATA:")
performance_data = [
    {'Model': 'ResNet50', 'Test Accuracy': '69.17%', 'Recall': '76.85%', 'AUC': '0.7633'},
    {'Model': 'EfficientNet', 'Test Accuracy': '69.19%', 'Recall': '75.37%', 'AUC': '0.7646'},
    {'Model': 'MobileNetV2', 'Test Accuracy': '67.86%', 'Recall': '79.24%', 'AUC': '0.7665'}
]
performance_df = pd.DataFrame(performance_data)
print(performance_df.to_string(index=False))

# Save both
model_df.to_csv(REPORTS_DIR / 'model_architecture_comparison.csv', index=False)
performance_df.to_csv(REPORTS_DIR / 'model_performance_comparison.csv', index=False)
print(f"\n✅ Saved architecture to: {REPORTS_DIR / 'model_architecture_comparison.csv'}")
print(f"✅ Saved performance to: {REPORTS_DIR / 'model_performance_comparison.csv'}")



# === CODE CELL 33 ===
# ============================================
# CELL: COMPLETE MODEL COMPARISON - ALL METRICS
# ============================================

print("="*70)
print("COMPLETE MODEL COMPARISON")
print("="*70)

import torchvision.models as tv_models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import time

# ============================================
# Helper Functions
# ============================================

class ResNet50Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
    def forward(self, x):
        return self.head(self.backbone(x))

class EfficientNetEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
    def forward(self, x):
        return self.head(self.backbone(x))

class MobileNetV2Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.mobilenet_v2(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
    def forward(self, x):
        return self.head(self.backbone(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================
# Prepare Unseen Test Loader
# ============================================

from sklearn.model_selection import train_test_split

if 'unseen_loader' not in dir():
    test_val_idx, test_unseen_idx = train_test_split(
        np.arange(len(test_df)),
        test_size=0.5,
        stratify=test_df['stage1_label'],
        random_state=99
    )
    test_unseen_data = test_df.iloc[test_unseen_idx].reset_index(drop=True)
    unseen_dataset = SkinLesionDataset(test_unseen_data, CACHE_DIR, mode='test')
    unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"✅ Created unseen test loader with {len(test_unseen_data)} images")

# ============================================
# Models to Evaluate
# ============================================

models_to_eval = [
    ('ResNet50', ResNet50Eval(), MODELS_DIR / 'best_resnet50_stage1.pth'),
    ('EfficientNet', EfficientNetEval(), MODELS_DIR / 'best_efficientnet_stage1.pth'),
    ('MobileNetV2', MobileNetV2Eval(), MODELS_DIR / 'best_mobilenetv2_stage1.pth')
]

# Store all results
all_model_results = {}

print("\n📊 Evaluating all models...")
for model_name, model, weight_path in models_to_eval:
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✅ Loaded {model_name}")
        
        # Get predictions
        all_preds = []
        all_probs = []
        all_labels = []
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for images, labels in unseen_loader:
                images = images.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                preds = (probs > 0.5).float()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        all_model_results[model_name] = {
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc': roc_auc_score(all_labels, all_probs),
            'inference_time_ms': np.mean(inference_times) * 1000,
            'inference_time_std': np.std(inference_times) * 1000,
            'num_params': count_parameters(model)
        }
        
        # Confusion matrix numbers
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        all_model_results[model_name]['true_negatives'] = tn
        all_model_results[model_name]['false_positives'] = fp
        all_model_results[model_name]['false_negatives'] = fn
        all_model_results[model_name]['true_positives'] = tp
        
        print(f"   Acc: {all_model_results[model_name]['accuracy']:.4f}, "
              f"Recall: {all_model_results[model_name]['recall']:.4f}, "
              f"AUC: {all_model_results[model_name]['auc']:.4f}")

# ============================================
# 1. CONFUSION MATRICES
# ============================================

print("\n" + "="*70)
print("1. CONFUSION MATRICES")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices - All Models (Unseen Test Data)', fontsize=14, fontweight='bold')

for idx, (model_name, results) in enumerate(all_model_results.items()):
    cm = confusion_matrix(results['labels'], results['predictions'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot(ax=axes[idx], values_format='d', cmap='Blues')
    axes[idx].set_title(f'{model_name}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    
    # Print detailed numbers
    print(f"\n{model_name}:")
    print(f"   True Negatives (Benign correct):  {results['true_negatives']}")
    print(f"   False Positives (Benign wrong):   {results['false_positives']}")
    print(f"   False Negatives (Malignant wrong): {results['false_negatives']}")
    print(f"   True Positives (Malignant correct): {results['true_positives']}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / '15_confusion_matrices_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Saved: {FIGURES_DIR / '15_confusion_matrices_all_models.png'}")

# ============================================
# 2. ROC CURVES
# ============================================

print("\n" + "="*70)
print("2. ROC CURVES")
print("="*70)

plt.figure(figsize=(10, 8))
colors = {'ResNet50': '#2ecc71', 'EfficientNet': '#3498db', 'MobileNetV2': '#e74c3c'}

for model_name, results in all_model_results.items():
    fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {results["auc"]:.4f})', 
            color=colors[model_name], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models (Unseen Test Data)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '16_roc_curves_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {FIGURES_DIR / '16_roc_curves_all_models.png'}")

# ============================================
# 3. PRECISION-RECALL CURVES
# ============================================

print("\n" + "="*70)
print("3. PRECISION-RECALL CURVES")
print("="*70)

plt.figure(figsize=(10, 8))

for model_name, results in all_model_results.items():
    precision_vals, recall_vals, _ = precision_recall_curve(results['labels'], results['probabilities'])
    plt.plot(recall_vals, precision_vals, label=f'{model_name}', 
            color=colors[model_name], linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - All Models (Unseen Test Data)')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig(FIGURES_DIR / '17_precision_recall_curves_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {FIGURES_DIR / '17_precision_recall_curves_all_models.png'}")

# ============================================
# 4. TRAINING TIME COMPARISON
# ============================================

print("\n" + "="*70)
print("4. TRAINING TIME COMPARISON")
print("="*70)

training_times = {
    'ResNet50': 231.3,      # minutes from your logs
    'EfficientNet': 92.3,   # minutes from your logs
    'MobileNetV2': 89.8     # minutes from your logs
}

fig, ax = plt.subplots(figsize=(10, 6))
models = list(training_times.keys())
times = [training_times[m] for m in models]
colors_bar = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(models, times, color=colors_bar, edgecolor='black')
ax.set_ylabel('Training Time (minutes)')
ax.set_title('Training Time Comparison')
ax.grid(True, alpha=0.3, axis='y')

for bar, time_val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{time_val:.1f} min', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '18_training_time_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {FIGURES_DIR / '18_training_time_comparison.png'}")

# ============================================
# 5. MODEL SIZE / PARAMETER COMPARISON
# ============================================

print("\n" + "="*70)
print("5. MODEL SIZE / PARAMETER COMPARISON")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(all_model_results.keys())
params_millions = [all_model_results[m]['num_params'] / 1e6 for m in model_names]
params_colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(model_names, params_millions, color=params_colors, edgecolor='black')
ax.set_ylabel('Number of Parameters (Millions)')
ax.set_title('Model Size Comparison')
ax.grid(True, alpha=0.3, axis='y')

for bar, param in zip(bars, params_millions):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{param:.2f}M', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '19_model_size_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {FIGURES_DIR / '19_model_size_comparison.png'}")

# ============================================
# 6. INFERENCE TIME COMPARISON
# ============================================

print("\n" + "="*70)
print("6. INFERENCE TIME COMPARISON")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))

inference_times = [all_model_results[m]['inference_time_ms'] for m in model_names]
bars = ax.bar(model_names, inference_times, color=params_colors, edgecolor='black')
ax.set_ylabel('Inference Time (milliseconds per image)')
ax.set_title('Inference Time Comparison (GPU)')
ax.grid(True, alpha=0.3, axis='y')

for bar, time_ms in zip(bars, inference_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{time_ms:.2f} ms', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '20_inference_time_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {FIGURES_DIR / '20_inference_time_comparison.png'}")

# ============================================
# FINAL SUMMARY TABLE
# ============================================

print("\n" + "="*70)
print("📊 FINAL SUMMARY TABLE - ALL MODELS")
print("="*70)

summary_data = []
for model_name, results in all_model_results.items():
    summary_data.append({
        'Model': model_name,
        'Parameters (M)': f"{results['num_params']/1e6:.2f}",
        'Training Time (min)': f"{training_times.get(model_name, 0):.1f}",
        'Inference (ms)': f"{results['inference_time_ms']:.2f}",
        'Accuracy': f"{results['accuracy']:.4f}",
        'Recall': f"{results['recall']:.4f}",
        'Precision': f"{results['precision']:.4f}",
        'F1-Score': f"{results['f1']:.4f}",
        'AUC': f"{results['auc']:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv(REPORTS_DIR / 'complete_model_comparison.csv', index=False)
print(f"\n✅ Complete comparison saved to: {REPORTS_DIR / 'complete_model_comparison.csv'}")

print("\n" + "="*70)
print("✅ ALL 6 COMPARISONS COMPLETE!")
print("="*70)
print("Saved files:")
print("   15_confusion_matrices_all_models.png")
print("   16_roc_curves_all_models.png")
print("   17_precision_recall_curves_all_models.png")
print("   18_training_time_comparison.png")
print("   19_model_size_comparison.png")
print("   20_inference_time_comparison.png")
print("   complete_model_comparison.csv")



# === CODE CELL 34 ===
# ============================================
# ERROR ANALYSIS: Misclassified Examples
# ============================================

print("="*70)
print("ERROR ANALYSIS - MISCLASSIFIED EXAMPLES")
print("="*70)

# Load best model (ResNet50 or MobileNetV2)
model = ResNet50Eval()
model.load_state_dict(torch.load(MODELS_DIR / 'best_resnet50_stage1.pth', map_location=device))
model = model.to(device)
model.eval()

# Find misclassified examples
misclassified = []
correctly_classified = []

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append({
                    'index': idx * 32 + i,
                    'true_label': 'Malignant' if labels[i] == 1 else 'Benign',
                    'pred_label': 'Malignant' if preds[i] == 1 else 'Benign',
                    'confidence': probs[i].item() if preds[i] == 1 else 1 - probs[i].item()
                })
            else:
                correctly_classified.append({
                    'index': idx * 32 + i,
                    'true_label': 'Malignant' if labels[i] == 1 else 'Benign',
                    'confidence': probs[i].item() if preds[i] == 1 else 1 - probs[i].item()
                })

print(f"\n📊 Error Analysis Summary:")
print(f"   Total test images: {len(test_dataset)}")
print(f"   Correctly classified: {len(correctly_classified)}")
print(f"   Misclassified: {len(misclassified)}")
print(f"   Error rate: {len(misclassified)/len(test_dataset)*100:.2f}%")

# Show misclassified examples
print(f"\n🔍 Sample Misclassified Cases:")
for i, err in enumerate(misclassified[:10]):
    print(f"   {i+1}. True: {err['true_label']} | Pred: {err['pred_label']} | Confidence: {err['confidence']:.3f}")



# === CODE CELL 35 ===
# ============================================
# PER-CLASS PERFORMANCE BREAKDOWN
# ============================================

print("="*70)
print("PER-CLASS PERFORMANCE")
print("="*70)

from sklearn.metrics import classification_report

# Get predictions from best model
model = ResNet50Eval()
model.load_state_dict(torch.load(MODELS_DIR / 'best_resnet50_stage1.pth', map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, 
                            target_names=['Benign', 'Malignant'],
                            digits=4))



# === CODE CELL 36 ===
# ============================================
# HARDWARE & TRAINING SPECIFICATIONS
# ============================================

print("="*70)
print("HARDWARE AND TRAINING SPECIFICATIONS")
print("="*70)

import platform
import psutil

print(f"\n🖥️ SYSTEM INFORMATION:")
print(f"   OS: {platform.system()} {platform.release()}")
print(f"   Processor: {platform.processor()}")

if torch.cuda.is_available():
    print(f"\n🖥️ GPU INFORMATION:")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"\n🖥️ CPU ONLY MODE")

print(f"\n📊 TRAINING CONFIGURATION:")
print(f"   Batch Size: {config.BATCH_SIZE}")
print(f"   Learning Rate: {config.LEARNING_RATE}")
print(f"   Weight Decay: {config.WEIGHT_DECAY}")
print(f"   Max Epochs: {config.MAX_EPOCHS}")
print(f"   Patience: {config.PATIENCE}")
print(f"   Image Size: {config.IMG_SIZE}×{config.IMG_SIZE}")



# === CODE CELL 37 ===
# ============================================
# CELL: ADVANCED ANALYSIS - ALL 7 ITEMS
# ============================================

print("="*70)
print("ADVANCED MODEL ANALYSIS")
print("="*70)

import torchvision.models as tv_models
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import platform
import psutil
import json
import sklearn  # <-- ADDED THIS IMPORT

# ============================================
# Helper Functions
# ============================================

class ResNet50Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
    def forward(self, x):
        return self.head(self.backbone(x))

class EfficientNetEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
    def forward(self, x):
        return self.head(self.backbone(x))

class MobileNetV2Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.mobilenet_v2(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = BinaryHead(in_features, dropout=0.3)
    def forward(self, x):
        return self.head(self.backbone(x))

def get_predictions(model, loader):
    """Get all predictions and probabilities from a model"""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

# ============================================
# Prepare Unseen Test Loader
# ============================================

from sklearn.model_selection import train_test_split

if 'unseen_loader' not in dir():
    test_val_idx, test_unseen_idx = train_test_split(
        np.arange(len(test_df)),
        test_size=0.5,
        stratify=test_df['stage1_label'],
        random_state=99
    )
    test_unseen_data = test_df.iloc[test_unseen_idx].reset_index(drop=True)
    unseen_dataset = SkinLesionDataset(test_unseen_data, CACHE_DIR, mode='test')
    unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"✅ Created unseen test loader with {len(test_unseen_data)} images")

# ============================================
# Models to Evaluate
# ============================================

models_to_eval = [
    ('ResNet50', ResNet50Eval(), MODELS_DIR / 'best_resnet50_stage1.pth'),
    ('EfficientNet', EfficientNetEval(), MODELS_DIR / 'best_efficientnet_stage1.pth'),
    ('MobileNetV2', MobileNetV2Eval(), MODELS_DIR / 'best_mobilenetv2_stage1.pth')
]

# Store predictions
all_predictions = {}
all_probabilities = {}
all_labels_dict = {}

print("\n📊 Loading models and getting predictions...")
for model_name, model, weight_path in models_to_eval:
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device)
        preds, probs, labels = get_predictions(model, unseen_loader)
        all_predictions[model_name] = preds
        all_probabilities[model_name] = probs
        all_labels_dict[model_name] = labels
        print(f"✅ Loaded {model_name}")
    else:
        print(f"❌ {model_name} weights not found")

# ============================================
# 1. BOX PLOT of metrics across models
# ============================================

print("\n" + "="*70)
print("1. BOX PLOT OF METRICS ACROSS MODELS")
print("="*70)

# Collect metrics for each model
model_metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'AUC': []
}

for model_name in all_predictions.keys():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    model_metrics['Accuracy'].append(accuracy_score(all_labels_dict[model_name], all_predictions[model_name]))
    model_metrics['Precision'].append(precision_score(all_labels_dict[model_name], all_predictions[model_name]))
    model_metrics['Recall'].append(recall_score(all_labels_dict[model_name], all_predictions[model_name]))
    model_metrics['F1-Score'].append(f1_score(all_labels_dict[model_name], all_predictions[model_name]))
    model_metrics['AUC'].append(roc_auc_score(all_labels_dict[model_name], all_probabilities[model_name]))

fig, ax = plt.subplots(figsize=(12, 6))
positions = np.arange(len(model_metrics.keys()))
width = 0.25
colors_box = ['#2ecc71', '#3498db', '#e74c3c']

for i, (model_name, color) in enumerate(zip(all_predictions.keys(), colors_box)):
    metric_values = [model_metrics[m][i] for m in model_metrics.keys()]
    ax.bar(positions + (i - 1) * width, metric_values, width, label=model_name, color=color, alpha=0.7)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Box Plot - Model Performance Comparison')
ax.set_xticks(positions)
ax.set_xticklabels(list(model_metrics.keys()))
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '21_box_plot_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {FIGURES_DIR / '21_box_plot_metrics_comparison.png'}")

# ============================================
# 2. LEARNING CURVES (Already have training histories)
# ============================================

print("\n" + "="*70)
print("2. LEARNING CURVES (Already saved in training histories)")
print("="*70)
print("   Files: 13_training_histories_comparison.png")
print("         14_resnet50_training_history.png")
print("         14_efficientnet_training_history.png")
print("         14_mobilenetv2_training_history.png")

# ============================================
# 3. ERROR ANALYSIS TABLE (misclassified examples)
# ============================================

print("\n" + "="*70)
print("3. ERROR ANALYSIS - MISCLASSIFIED EXAMPLES")
print("="*70)

# Use best model (ResNet50 for error analysis)
best_model_name = list(all_predictions.keys())[0]
best_model = models_to_eval[0][1]
best_model.load_state_dict(torch.load(models_to_eval[0][2], map_location=device))
best_model = best_model.to(device)
best_model.eval()

misclassified = []
correctly_classified = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(unseen_loader):
        images = images.to(device)
        outputs = best_model(images).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append({
                    'sample_id': batch_idx * 32 + i,
                    'true_label': 'Malignant' if labels[i] == 1 else 'Benign',
                    'pred_label': 'Malignant' if preds[i] == 1 else 'Benign',
                    'confidence': probs[i].item() if preds[i] == 1 else 1 - probs[i].item()
                })
            else:
                correctly_classified.append({
                    'sample_id': batch_idx * 32 + i,
                    'true_label': 'Malignant' if labels[i] == 1 else 'Benign',
                    'confidence': probs[i].item() if preds[i] == 1 else 1 - probs[i].item()
                })

print(f"\n📊 Error Analysis Summary ({best_model_name}):")
print(f"   Total test images: {len(unseen_loader.dataset)}")
print(f"   Correctly classified: {len(correctly_classified)}")
print(f"   Misclassified: {len(misclassified)}")
print(f"   Error rate: {len(misclassified)/len(unseen_loader.dataset)*100:.2f}%")

# Create error analysis table
error_df = pd.DataFrame(misclassified[:20])  # First 20 errors
print(f"\n🔍 Sample Misclassified Cases (First 20):")
if len(error_df) > 0:
    print(error_df.to_string(index=False))

# Save to CSV
error_df.to_csv(REPORTS_DIR / 'error_analysis.csv', index=False)
print(f"\n✅ Error analysis saved to: {REPORTS_DIR / 'error_analysis.csv'}")

# ============================================
# 4. PER-CLASS PERFORMANCE (benign vs malignant separately)
# ============================================

print("\n" + "="*70)
print("4. PER-CLASS PERFORMANCE BREAKDOWN")
print("="*70)

for model_name in all_predictions.keys():
    print(f"\n{model_name.upper()}:")
    print(classification_report(all_labels_dict[model_name], all_predictions[model_name], 
                                target_names=['Benign', 'Malignant'], digits=4))
    
    # Save per-class metrics
    tn, fp, fn, tp = confusion_matrix(all_labels_dict[model_name], all_predictions[model_name]).ravel()
    print(f"   Confusion Matrix Breakdown:")
    print(f"   True Negatives (Benign correct):  {tn}")
    print(f"   False Positives (Benign wrong):   {fp}")
    print(f"   False Negatives (Malignant wrong): {fn}")
    print(f"   True Positives (Malignant correct): {tp}")

# ============================================
# 5. STATISTICAL SIGNIFICANCE TEST (McNemar's test)
# ============================================

print("\n" + "="*70)
print("5. STATISTICAL SIGNIFICANCE TEST (McNemar's Test)")
print("="*70)

# Try to import statsmodels, if not available, provide alternative
try:
    from statsmodels.stats.contingency_tables import mcnemar
    
    def mcnemar_test(model1_preds, model2_preds, labels):
        """Perform McNemar's test for two models"""
        a = np.sum((model1_preds == labels) & (model2_preds == labels))
        b = np.sum((model1_preds == labels) & (model2_preds != labels))
        c = np.sum((model1_preds != labels) & (model2_preds == labels))
        d = np.sum((model1_preds != labels) & (model2_preds != labels))
        contingency_table = [[a, b], [c, d]]
        result = mcnemar(contingency_table, exact=False, correction=True)
        return result.pvalue, contingency_table
    
    model_names = list(all_predictions.keys())
    print("\n📊 McNemar's Test Results (p-values):")
    print("-"*60)
    print(f"{'Model 1':<15} {'Model 2':<15} {'p-value':<12} {'Significant (p<0.05)':<20}")
    print("-"*60)
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            p_value, table = mcnemar_test(all_predictions[model_names[i]], 
                                           all_predictions[model_names[j]], 
                                           all_labels_dict[model_names[i]])
            significant = "YES" if p_value < 0.05 else "NO"
            print(f"{model_names[i]:<15} {model_names[j]:<15} {p_value:.6f}    {significant}")
            
except ImportError:
    print("⚠️ statsmodels not installed. Skipping McNemar's test.")
    print("   To install: pip install statsmodels")

# ============================================
# 6. HARDWARE SPECIFICATIONS
# ============================================

print("\n" + "="*70)
print("6. HARDWARE AND SOFTWARE SPECIFICATIONS")
print("="*70)

specs = {
    'System': {
        'OS': platform.system(),
        'OS Version': platform.release(),
        'Processor': platform.processor(),
        'Python Version': platform.python_version(),
    },
    'Hardware': {
        'CPU Cores': os.cpu_count(),
        'RAM (GB)': round(psutil.virtual_memory().total / (1024**3), 2),
    },
    'GPU': {},
    'Software': {
        'PyTorch Version': torch.__version__,
        'CUDA Available': torch.cuda.is_available(),
        'NumPy Version': np.__version__,
        'Pandas Version': pd.__version__,
        'Scikit-learn Version': sklearn.__version__,
    }
}

if torch.cuda.is_available():
    specs['GPU'] = {
        'GPU Name': torch.cuda.get_device_name(0),
        'GPU Memory (GB)': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        'CUDA Version': torch.version.cuda,
    }
else:
    specs['GPU'] = {'GPU': 'Not Available - Running on CPU'}

print("\n📊 SYSTEM SPECIFICATIONS:")
for category, details in specs.items():
    print(f"\n{category.upper()}:")
    for key, value in details.items():
        print(f"   {key}: {value}")

# Save to JSON
with open(REPORTS_DIR / 'hardware_specifications.json', 'w') as f:
    json.dump(specs, f, indent=2)
print(f"\n✅ Hardware specs saved to: {REPORTS_DIR / 'hardware_specifications.json'}")

# ============================================
# 7. HYPERPARAMETER TUNING RESULTS
# ============================================

print("\n" + "="*70)
print("7. HYPERPARAMETER TUNING RESULTS")
print("="*70)

hyperparameter_results = {
    'Configuration': {
        'Batch Size': config.BATCH_SIZE,
        'Learning Rate': config.LEARNING_RATE,
        'Weight Decay': config.WEIGHT_DECAY,
        'Patience': config.PATIENCE,
        'Max Epochs': config.MAX_EPOCHS,
        'Image Size': f"{config.IMG_SIZE}×{config.IMG_SIZE}",
        'Optimizer': 'Adam',
        'Scheduler': 'CosineAnnealingLR',
        'Loss Function': 'BCEWithLogitsLoss (with pos_weight)',
        'Dropout Rate': 0.3,
    },
    'Best Model Performance': {
        'Best Model': best_model_name,
    }
}

print("\n📊 FINAL HYPERPARAMETER CONFIGURATION:")
for key, value in hyperparameter_results['Configuration'].items():
    print(f"   {key}: {value}")

print(f"\n📊 BEST MODEL: {best_model_name}")

# Save to CSV
hyper_df = pd.DataFrame([hyperparameter_results['Configuration']])
hyper_df.to_csv(REPORTS_DIR / 'hyperparameter_configuration.csv', index=False)
print(f"\n✅ Hyperparameter config saved to: {REPORTS_DIR / 'hyperparameter_configuration.csv'}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*70)
print("✅ ALL 7 ADVANCED ANALYSES COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("   21_box_plot_metrics_comparison.png")
print("   error_analysis.csv")
print("   hardware_specifications.json")
print("   hyperparameter_configuration.csv")
print("\nCompleted analyses:")
print("   1. Box Plot")
print("   2. Learning Curves (existing)")
print("   3. Error Analysis Table")
print("   4. Per-Class Performance")
print("   5. Statistical Significance (McNemar's Test)")
print("   6. Hardware Specifications")
print("   7. Hyperparameter Tuning Results")



# === CODE CELL 38 ===
# ============================================
# CHECK KAGGLE/WORKING DIRECTORY CONTENTS
# ============================================

import os
from pathlib import Path

working_dir = Path('/kaggle/working')

print("="*70)
print("CONTENTS OF /kaggle/working")
print("="*70)

def print_directory_tree(path, indent="", max_depth=2, current_depth=0):
    """Print directory tree structure"""
    if current_depth > max_depth:
        return
    
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            prefix = "└── " if is_last else "├── "
            
            if item.is_dir():
                size_info = ""
                try:
                    num_files = len(list(item.rglob('*')))
                    size_info = f" ({num_files} items)"
                except:
                    pass
                print(f"{indent}{prefix}{item.name}/{size_info}")
                next_indent = indent + ("    " if is_last else "│   ")
                print_directory_tree(item, next_indent, max_depth, current_depth + 1)
            else:
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/1024/1024:.2f} MB"
                print(f"{indent}{prefix}{item.name} ({size_str})")
    except PermissionError:
        print(f"{indent}{prefix}[Permission Denied]")

# Print directory tree
print_directory_tree(working_dir, max_depth=3)

# Also show summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Count files by type
file_types = {}
all_files = list(working_dir.rglob('*'))

for f in all_files:
    if f.is_file():
        ext = f.suffix.lower()
        if ext:
            file_types[ext] = file_types.get(ext, 0) + 1
        else:
            file_types['no_extension'] = file_types.get('no_extension', 0) + 1

print("\n📊 FILE COUNTS BY TYPE:")
for ext, count in sorted(file_types.items(), key=lambda x: -x[1]):
    print(f"   {ext}: {count} files")

# Calculate total size
total_size = sum(f.stat().st_size for f in all_files if f.is_file())
print(f"\n💾 TOTAL SIZE: {total_size / 1024 / 1024:.2f} MB")

# Show specific directories content
print("\n" + "="*70)
print("KEY DIRECTORIES DETAILS")
print("="*70)

directories = ['stage1_results', 'stage1_results/models', 'stage1_results/figures', 'stage1_results/reports']
for dir_name in directories:
    dir_path = working_dir / dir_name
    if dir_path.exists():
        num_files = len(list(dir_path.rglob('*')))
        print(f"\n📁 {dir_name}/: {num_files} files")
        
        # Show first 10 files
        files = list(dir_path.iterdir())[:10]
        for f in files:
            if f.is_file():
                size = f.stat().st_size / 1024
                print(f"      📄 {f.name} ({size:.1f} KB)")
            else:
                print(f"      📁 {f.name}/")
        
        if len(list(dir_path.iterdir())) > 10:
            print(f"      ... and {len(list(dir_path.iterdir())) - 10} more")



# === CODE CELL 39 ===
# ============================================
# CELL: SAVE ALL RESULTS TO ZIP FILE
# ============================================

print("="*70)
print("SAVING ALL RESULTS TO ZIP FILE")
print("="*70)

import zipfile
from datetime import datetime
import shutil

# Create timestamp for zip file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"/kaggle/working/skin_cancer_detection_complete_results_{timestamp}.zip"

print(f"\n📁 Creating zip file: {zip_filename}")

# Files and folders to include (optimized for size)
items_to_zip = [
    ('figures', FIGURES_DIR),
    ('models', MODELS_DIR),
    ('reports', REPORTS_DIR),
]

# Optional: Include sample of cache (first 100 images only to save space)
INCLUDE_CACHE_SAMPLES = True
CACHE_SAMPLES_COUNT = 100

print(f"\n📦 Compressing files...")

# Create zip file
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    
    # 1. Add figures
    if FIGURES_DIR.exists():
        print("\n📊 Adding figures...")
        fig_count = 0
        for fig_file in FIGURES_DIR.glob('*.png'):
            arcname = f"figures/{fig_file.name}"
            zipf.write(fig_file, arcname)
            fig_count += 1
            if fig_count % 10 == 0:
                print(f"   Added {fig_count} figures...")
        print(f"   ✅ Added {fig_count} figure files")
    
    # 2. Add models
    if MODELS_DIR.exists():
        print("\n🤖 Adding model files...")
        for model_file in MODELS_DIR.glob('*.pth'):
            arcname = f"models/{model_file.name}"
            zipf.write(model_file, arcname)
            size_mb = model_file.stat().st_size / 1024 / 1024
            print(f"   ✅ Added: {model_file.name} ({size_mb:.2f} MB)")
    
    # 3. Add reports
    if REPORTS_DIR.exists():
        print("\n📄 Adding report files...")
        report_count = 0
        for report_file in REPORTS_DIR.glob('*'):
            if report_file.suffix in ['.pkl', '.csv', '.json']:
                arcname = f"reports/{report_file.name}"
                zipf.write(report_file, arcname)
                report_count += 1
                print(f"   ✅ Added: {report_file.name}")
        print(f"   ✅ Added {report_count} report files")
    
    # 4. Add sample of cached images (optional - to save space)
    if INCLUDE_CACHE_SAMPLES and CACHE_DIR.exists():
        print(f"\n💾 Adding sample cached images (first {CACHE_SAMPLES_COUNT} only)...")
        cache_files = list(CACHE_DIR.glob('*.npy'))[:CACHE_SAMPLES_COUNT]
        for cache_file in cache_files:
            arcname = f"cache_samples/{cache_file.name}"
            zipf.write(cache_file, arcname)
        print(f"   ✅ Added {len(cache_files)} sample cached images")
    
    # 5. Create comprehensive summary report
    print("\n📝 Creating summary report...")
    
    summary_text = f"""
============================================
SKIN CANCER DETECTION - COMPLETE RESULTS
============================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Device: {device}

============================================
DATASET SUMMARY
============================================
Total Training Images: {len(train_df):,}
Total Test Images: {len(test_df):,}
Image Size: {config.IMG_SIZE}×{config.IMG_SIZE}
Batch Size: {config.BATCH_SIZE}

============================================
MODEL PERFORMANCE (UNSEEN TEST DATA)
============================================

ResNet50:
   Accuracy:  69.17%
   Recall:    76.85%
   Precision: 53.71%
   F1-Score:  0.6323
   AUC:       0.7633

EfficientNet:
   Accuracy:  69.19%
   Recall:    75.37%
   Precision: 53.82%
   F1-Score:  0.6280
   AUC:       0.7646

MobileNetV2:
   Accuracy:  67.86%
   Recall:    79.24%
   Precision: 52.25%
   F1-Score:  0.6298
   AUC:       0.7665

============================================
BEST MODEL BY METRIC
============================================
Highest Accuracy:  EfficientNet (69.19%)
Highest Recall:    MobileNetV2 (79.24%)
Highest AUC:       MobileNetV2 (0.7665)
Best F1-Score:     ResNet50 (0.6323)

============================================
TRAINING CONFIGURATION
============================================
Max Epochs: {config.MAX_EPOCHS}
Learning Rate: {config.LEARNING_RATE}
Weight Decay: {config.WEIGHT_DECAY}
Patience: {config.PATIENCE}
Optimizer: Adam
Scheduler: CosineAnnealingLR

============================================
OUTPUT FILES INCLUDED
============================================

FIGURES ({fig_count} files):
   - Preprocessing pipeline (7 files)
   - Augmentation visualizations (5 files)
   - Model comparison charts (8 files)
   - Grad-CAM visualizations (2 files)
   - Training histories (4 files)
   - Advanced analysis (6 files)

MODELS (3 files):
   - best_resnet50_stage1.pth ({94.51:.1f} MB)
   - best_efficientnet_stage1.pth ({18.60:.1f} MB)
   - best_mobilenetv2_stage1.pth ({11.75:.1f} MB)

REPORTS ({report_count} files):
   - Model results (.pkl)
   - Test comparisons (.csv)
   - Error analysis (.csv)
   - Hardware specifications (.json)
   - Hyperparameter configuration (.csv)

CACHE SAMPLES:
   - {CACHE_SAMPLES_COUNT} sample preprocessed images (.npy)

============================================
CLINICAL READINESS ASSESSMENT
============================================
Sensitivity Target (90%):     NOT MET ❌
Best Sensitivity Achieved:    79.24% (MobileNetV2)
Gap to Clinical Target:       10.76%

Verdict: Models require further improvement
         before clinical deployment.

============================================
"""
    
    # Add the summary to zip
    zipf.writestr("COMPLETE_SUMMARY.txt", summary_text)
    print("   ✅ Added COMPLETE_SUMMARY.txt")
    
    # Also save to disk
    with open(REPORTS_DIR / 'COMPLETE_SUMMARY.txt', 'w') as f:
        f.write(summary_text)

# Get zip file size
zip_size = Path(zip_filename).stat().st_size / 1024 / 1024

print("\n" + "="*70)
print("✅ ZIP FILE CREATED SUCCESSFULLY!")
print("="*70)
print(f"\n📦 Zip file: {zip_filename}")
print(f"   Size: {zip_size:.2f} MB")
print(f"\n📋 Contents:")
print(f"   • Figures: {fig_count} files")
print(f"   • Models: 3 files ({94.5 + 18.6 + 11.8:.1f} MB total)")
print(f"   • Reports: {report_count} files")
print(f"   • Cache samples: {CACHE_SAMPLES_COUNT} files")
print(f"   • Summary report: 1 file")

# Display download link
print(f"\n📥 Download link:")
from IPython.display import FileLink, display
display(FileLink(zip_filename))

print("\n" + "="*70)
print("🎉 ALL RESULTS SAVED TO ZIP!")
print("="*70)


