# 🏥 Automated Skin Cancer Detection and Classification Using Deep Learning

**Final Year Project — BSc Computer Science (Session 2022-2026)**  
**University of Peshawar, Shaikh Zayed Islamic Centre**

## Team
- **Zohrouf Khattak** (Roll No: 1000)
- **Huma Zeb** (Roll No: 883)
- **Supervisor**: Dr Muhammad Ayaz

---

## Project Overview

A deep learning pipeline for automated skin cancer detection using dermoscopic images from the **ISIC 2019 dataset** (25,331 training + 8,238 test images):

1. **Stage 1 (Binary Classification)**: Classifies skin lesions as Benign or Malignant

## Folder Structure

```
├── data/                   Ground truth CSV files
├── notebooks/              Jupyter notebooks (all experiments)
│   └── stage1_binary_classification.ipynb
├── models/                 Trained model weights (.pth)
│   └── stage1/             3 models (ResNet50, EfficientNet-B0, MobileNetV2)
├── results/                All figures, reports, and visualizations
│   └── stage1/             Binary classification results
├── docs/                   Proposal, architecture docs, thesis
├── scripts/                Utility Python scripts
├── archive/                Dataset ZIPs and result archives
├── master_results.md       📊 Consolidated results (all experiments)
└── README.md               This file
```

## Key Results

| Stage | Best Model | Test Accuracy | Key Metric |
|-------|-----------|-------------|-----------|
| Stage 1 (Binary) | EfficientNet-B0 | 69.19% | AUC: 0.7665 |

## Dataset

- **ISIC 2019 Challenge** — International Skin Imaging Collaboration
- 8 diagnostic categories mapped to binary (Benign/Malignant) and 4-class malignant subtypes
- Preprocessing: Hair removal, lesion detection, cropping, 224×224 resize

## Technologies

- Python, PyTorch, torchvision, timm
- OpenCV (preprocessing), scikit-learn (metrics)
- Grad-CAM (model interpretability)
- Trained on Kaggle (NVIDIA Tesla T4 GPU)

## Project Roadmap (Finalized Architecture)

Based on recent project reviews, the final architecture will be a **Dual Ensemble Pipeline** with synthetic data augmentation. The timeline is as follows:

### 1. Stage 1 (Binary Classification: Benign vs. Malignant)
- **Goal**: Maximize binary classification accuracy before passing images to Stage 2.
- **Action**: Build a **5-Model Ensemble** using:
  1. ResNet50 (Trained)
  2. EfficientNet-B0 (Trained)
  3. MobileNetV2 (Trained)
  4. ConvNeXt (To be trained)
  5. Vision Transformer / ViT (To be trained)

### 2. Data Imbalance Resolution (In-Between Stage)
- **Goal**: Fix the severe class imbalance in the Malignant subtypes before training Stage 2.
- **Action**: Train **Conditional GANs** to generate highly realistic, synthetic images of the rare cancer classes (e.g., SCC, BCC). These synthetic images will be merged with the real dataset to achieve perfect class balance.

### 3. Stage 2 (Multi-Class Classification: Malignant Subtypes)
- **Goal**: Classify malignant tumors into specific subtypes using the correct `ISIC 2019` dataset from the `archive/` folder.
- **Action**: Build a **Second 5-Model Ensemble** (Dual Ensemble). We will train 5 brand new models (using the exact same 5 architectures from Stage 1) completely from scratch on the balanced, malignant-only dataset.
