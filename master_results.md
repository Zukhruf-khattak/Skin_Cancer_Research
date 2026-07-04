# Master Results Summary - Automated Skin Cancer Detection and Classification

This document serves as the single source of truth for the final metrics achieved across both stages of the dual-stage ensemble pipeline, reflecting all revisions made per the final reviewer feedback in June 2026.

---

## 1. Stage 1: Binary Classification (Benign vs. Malignant)

* **Task:** Triage classification mapping (Malignant: MEL, BCC, AK, SCC | Benign: NV, BKL, DF, VASC)
* **Dataset Size:** 25,331 images (original)
* **Splits:** Train (raw): 17,731 | Train (balanced): 12,822 | Val: 3,800 | Test: 3,800 images
* **Imbalance Mitigation:** Targeted undersampling of the Melanocytic Nevus (NV) class (8,978 to 4,238 images) in the training set
* **Training:** Fine-tuning ImageNet pre-trained CNNs and Vision Transformers (70 epochs, Adam optimizer, lr=1e-4)

### 1.1 Model Performance (Unseen Test Set - 3,800 images)

| Model | Accuracy | Precision | Sensitivity (Recall) | F1-Score | AUC |
|---|---|---|---|---|---|
| ResNet50 | 92.76% | 90.80% | 89.44% | 0.9011 | 0.9675 |
| EfficientNet-B0 | 92.87% | 88.38% | 92.86% | 0.9057 | 0.9713 |
| MobileNetV2 | 91.89% | 87.61% | 90.86% | 0.8921 | 0.9680 |
| ConvNeXt-Tiny | 93.18% | 87.71% | 94.79% | 0.9111 | 0.9746 |
| Swin-Tiny | 93.55% | 88.79% | 94.43% | 0.9153 | 0.9734 |
| **ENSEMBLE** | **94.34%** | **91.12%** | **93.79%** | **0.9244** | **0.9845** |

### 1.2 Stage 1 Clinical Confusion Matrix (3,800 images)

* **True Negatives (TN):** 2,271 (Correct Benign)
* **False Positives (FP):** 128 (Benign flagged as Malignant)
* **False Negatives (FN):** 87 (Malignant missed - FNR = 6.21%)
* **True Positives (TP):** 1,314 (Correct Malignant)

---

## 2. Stage 2: Multi-Class Malignant Subtype Classification

* **Task:** 4-class classification of malignant lesions (MEL, BCC, SCC, AK)
* **Splits:** Train: 3,846 | Val: 824 | Test: 825 images
* **Imbalance Mitigation:** Focal Loss (gamma=2, alpha=class frequencies) + MixUp (alpha=0.2) + Heavy Data Augmentation
* **Training:** From-scratch training (40-50 epochs, lr=1e-3, CosineAnnealingLR T_max=50)

### 2.1 Model Performance (Unseen Test Set - 825 images)

| Model | Accuracy | Macro F1 | Macro AUC | MEL Recall | BCC Recall | SCC Recall | AK Recall | Clinical Score |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | 80.48% | 0.7814 | 0.9480 | 81.67% | 77.33% | 91.57% | 76.92% | 0.8472 |
| EfficientNet-B0 | 83.39% | 0.8132 | 0.9615 | 86.00% | 82.00% | 88.42% | 76.92% | 0.8566 |
| MobileNetV2 | 80.24% | 0.7764 | 0.9506 | 82.33% | 79.00% | 87.37% | 73.08% | 0.8309 |
| ConvNeXt-Tiny | 80.36% | 0.7789 | 0.9549 | 80.67% | 77.67% | 91.57% | 77.69% | 0.8443 |
| Swin-Tiny | 82.91% | 0.8063 | 0.9617 | 83.00% | 80.00% | 91.57% | **83.08%** | 0.8614 |
| **ENSEMBLE** | **85.58%** | **0.8332** | **0.9738** | **85.00%** | **85.67%** | **95.79%** | 79.23% | **0.8881** |

---

## 3. Revision and Conversion Status

* **Chapter 3 (Methodology):** Fully revised to separate numerical results into Chapter 4, resolve split size arithmetic discrepancies, update CosineAnnealingLR T_max explanations, add Figure captions, and include ethical/clinical considerations.
* **Chapter 4 (Results):** Created as a dedicated chapter with precise split counts (Stage 1 test set: 3,800 images; Stage 2 test set: 825 images), corrected baseline comparison footnotes, and Clopper-Pearson exact confidence intervals for sensitivity ([92.2%, 95.2%]).
* **DOCX Conversion:** Both files (Methodology_Chapter.docx and Results_Chapter.docx) successfully compiled and verified.

*Last updated: 2026-06-13*
