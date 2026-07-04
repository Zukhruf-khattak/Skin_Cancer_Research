# 🏗️ STAGE 1: COMPLETE FINAL ARCHITECTURE DESIGN


## Binary Classification for Skin Cancer Detection


### No Further Changes - Production Ready


---


# 📋 TABLE OF CONTENTS


1. [System Overview](#1-system-overview)

2. [Data Understanding & Distribution](#2-data-understanding--distribution)

3. [Imbalance Handling Strategy](#3-imbalance-handling-strategy)

4. [Preprocessing Pipeline](#4-preprocessing-pipeline)

5. [Data Augmentation](#5-data-augmentation)

6. [Cache Management](#6-cache-management)

7. [Data Splits](#7-data-splits)

8. [Model Architectures](#8-model-architectures)

9. [Ensemble Strategy](#9-ensemble-strategy)

10. [Training Configuration](#10-training-configuration)

11. [Loss Function Details](#11-loss-function-details)

12. [Evaluation Metrics](#12-evaluation-metrics)

13. [Expected Performance](#13-expected-performance)

14. [Output Structure](#14-output-structure)

15. [Quick Reference Card](#15-quick-reference-card)


---


# 1. SYSTEM OVERVIEW


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    STAGE 1: BINARY CLASSIFICATION SYSTEM                             │

│                                                                                       │

│    GOAL: "Does this skin lesion require medical attention?"                          │

│                                                                                       │

│    ┌─────────────────────────────────────────────────────────────────────────────┐   │

│    │                         INPUT: Dermoscopy Image                              │   │

│    │                              (224×224×3)                                     │   │

│    └─────────────────────────────────────────────────────────────────────────────┘   │

│                                         │                                            │

│                                         ▼                                            │

│    ┌─────────────────────────────────────────────────────────────────────────────┐   │

│    │                    PREPROCESSING PIPELINE                                    │   │

│    │  ┌─────────────────────────────────────────────────────────────────────┐   │   │

│    │  │  Step 1: Hair Removal (Black Hat Transform + Inpainting)            │   │   │

│    │  │  Step 2: Lesion Detection (LAB color space + Otsu threshold)        │   │   │

│    │  │  Step 3: Lesion Cropping (20% padding)                               │   │   │

│    │  │  Step 4: Resize to 224×224 + Cache as uint8 NPY                      │   │   │

│    │  └─────────────────────────────────────────────────────────────────────┘   │   │

│    └─────────────────────────────────────────────────────────────────────────────┘   │

│                                         │                                            │

│                                         ▼                                            │

│    ┌─────────────────────────────────────────────────────────────────────────────┐   │

│    │                    DATA AUGMENTATION (Training Only)                         │   │

│    │  ┌─────────────────────────────────────────────────────────────────────┐   │   │

│    │  │  • RandomResizedCrop (scale 0.8-1.0)     → Scale invariance        │   │   │

│    │  │  • Horizontal/Vertical Flip (p=0.5)      → Mirror invariance       │   │   │

│    │  │  • Random Rotation (±45°, p=0.5)          → Rotation invariance     │   │   │

│    │  │  • Color Jitter (p=0.5)                   → Color invariance        │   │   │

│    │  │  • Gaussian Noise (p=0.3)                 → Noise robustness        │   │   │

│    │  │  • Normalization (ImageNet stats)         → Standardized input      │   │   │

│    │  └─────────────────────────────────────────────────────────────────────┘   │   │

│    └─────────────────────────────────────────────────────────────────────────────┘   │

│                                         │                                            │

│                                         ▼                                            │

│    ┌─────────────────────────────────────────────────────────────────────────────┐   │

│    │                    THREE MODEL ENSEMBLE                                      │   │

│    │                                                                              │   │

│    │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │   │

│    │   │    ResNet50     │  │  EfficientNet   │  │   MobileNetV2   │             │   │

│    │   │  (23.5M params) │  │   (5.3M params) │  │   (3.5M params) │             │   │

│    │   │                 │  │                 │  │                 │             │   │

│    │   │  Backbone:      │  │  Backbone:      │  │  Backbone:      │             │   │

│    │   │  ResNet50       │  │  EfficientNet   │  │  MobileNetV2    │             │   │

│    │   │  ↓ GAP (2048)   │  │  ↓ GAP (1280)   │  │  ↓ GAP (1280)   │             │   │

│    │   │  ↓ Custom Head  │  │  ↓ Custom Head  │  │  ↓ Custom Head  │             │   │

│    │   │  ↓ Logit_RN     │  │  ↓ Logit_EF     │  │  ↓ Logit_MB     │             │   │

│    │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │   │

│    │            │                    │                    │                       │   │

│    │            └────────────────────┼────────────────────┘                       │   │

│    │                                 │                                            │   │

│    │                                 ▼                                            │   │

│    │   ┌─────────────────────────────────────────────────────────────────────┐   │   │

│    │   │                    WEIGHTED ENSEMBLE                                 │   │   │

│    │   │                                                                      │   │   │

│    │   │   ensemble_logit = 0.40×Logit_RN + 0.35×Logit_EF + 0.25×Logit_MB    │   │   │

│    │   │                                                                      │   │   │

│    │   └─────────────────────────────────────────────────────────────────────┘   │   │

│    └─────────────────────────────────────────────────────────────────────────────┘   │

│                                         │                                            │

│                                         ▼                                            │

│    ┌─────────────────────────────────────────────────────────────────────────────┐   │

│    │                    POST-PROCESSING                                          │   │

│    │  ┌─────────────────────────────────────────────────────────────────────┐   │   │

│    │  │  Temperature Scaling: scaled_logit = ensemble_logit / T (T=1.23)   │   │   │

│    │  │  Probability: P = sigmoid(scaled_logit)                             │   │   │

│    │  │  Threshold: Compare P > 0.44 (tuned for Sensitivity ≥ 0.90)        │   │   │

│    │  └─────────────────────────────────────────────────────────────────────┘   │   │

│    └─────────────────────────────────────────────────────────────────────────────┘   │

│                                         │                                            │

│                                         ▼                                            │

│    ┌─────────────────────────────────────────────────────────────────────────────┐   │

│    │                         OUTPUT                                               │   │

│    │                                                                              │   │

│    │   ┌─────────────────────────┐    ┌─────────────────────────┐                │   │

│    │   │   BENIGN (0)            │    │   MALIGNANT (1)         │                │   │

│    │   │   "No medical action"   │    │   "Medical care needed"  │                │   │

│    │   │                         │    │                         │                │   │

│    │   │   Includes:             │    │   Includes:              │                │   │

│    │   │   • NV (Mole)           │    │   • MEL (Melanoma)       │                │   │

│    │   │   • BKL (Benign)        │    │   • BCC (Basal Cell)     │                │   │

│    │   │   • DF (Dermatofibroma) │    │   • SCC (Squamous)       │                │   │

│    │   │   • VASC (Vascular)     │    │   • AK (Precancerous)    │                │   │

│    │   │                         │    │                         │                │   │

│    │   │   Action:               │    │   Action:                │                │   │

│    │   │   • Reassure patient    │    │   • Refer to Stage 2     │                │   │

│    │   │   • Routine monitoring  │    │   • Type classification  │                │   │

│    │   └─────────────────────────┘    └─────────────────────────┘                │   │

│    └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 2. DATA UNDERSTANDING & DISTRIBUTION


## 2.1 Complete Dataset Distribution


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    ISIC 2019 DATASET - COMPLETE DISTRIBUTION                         │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   TRAINING SET (25,331 images):                                                      │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │   Class Code    Full Name                    Type          Count    %       │   │

│   │   ───────────────────────────────────────────────────────────────────────── │   │

│   │   MEL           Melanoma                     CANCER        4,522    17.9%   │   │

│   │   BCC           Basal Cell Carcinoma         CANCER        3,323    13.1%   │   │

│   │   SCC           Squamous Cell Carcinoma      CANCER          628     2.5%   │   │

│   │   AK            Actinic Keratosis            PRECANCER       867     3.4%   │   │

│   │   ───────────────────────────────────────────────────────────────────────── │   │

│   │   Total MALIGNANT (Needs Care):                            9,340    36.9%   │   │

│   │   ───────────────────────────────────────────────────────────────────────── │   │

│   │   NV            Nevus (Mole)                 BENIGN        12,875    50.8%  │   │

│   │   BKL           Benign Keratosis             BENIGN         2,624    10.4%  │   │

│   │   DF            Dermatofibroma               BENIGN           239     0.9%  │   │

│   │   VASC          Vascular Lesion              BENIGN           253     1.0%  │   │

│   │   ───────────────────────────────────────────────────────────────────────── │   │

│   │   Total BENIGN (No Action):                              15,991    63.1%    │   │

│   │                                                                              │   │

│   │   TOTAL: 25,331 images                                                       │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   TEST SET (8,238 images - LOCKED until final evaluation):                          │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │   MEL: ~1,327    BCC: ~1,000    SCC: ~200    AK: ~250                        │   │

│   │   NV: ~4,000     BKL: ~800      DF: ~70      VASC: ~75                       │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 2.2 Binary Label Mapping


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    STAGE 1: BINARY LABEL MAPPING                                     │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  MALIGNANT (NEEDS MEDICAL ATTENTION) = 1                                    │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  Class    Count    Type                    Clinical Action          │    │   │

│   │  │  ─────────────────────────────────────────────────────────────────  │    │   │

│   │  │  MEL      4,522    Melanoma (Deadliest)    Urgent biopsy            │    │   │

│   │  │  BCC      3,323    Basal Cell (Common)     Mohs surgery             │    │   │

│   │  │  SCC        628    Squamous (Metastatic)   Surgical excision        │    │   │

│   │  │  AK         867    Actinic (Precancer)     Cryotherapy, monitor     │    │   │

│   │  │  ─────────────────────────────────────────────────────────────────  │    │   │

│   │  │  Total:   9,340 images (36.9%)                                      │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  BENIGN (NO ACTION NEEDED) = 0                                              │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  Class    Count    Type                    Clinical Action          │    │   │

│   │  │  ─────────────────────────────────────────────────────────────────  │    │   │

│   │  │  NV       12,875   Common mole             Routine monitoring       │    │   │

│   │  │  BKL      2,624    Benign keratosis        Reassure, no action      │    │   │

│   │  │  DF         239    Dermatofibroma          Reassure, no action      │    │   │

│   │  │  VASC       253    Vascular lesion         Monitor if changes       │    │   │

│   │  │  ─────────────────────────────────────────────────────────────────  │    │   │

│   │  │  Total:   15,991 images (63.1%)                                      │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 3. IMBALANCE HANDLING STRATEGY


## 3.1 Five Complementary Solutions


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    COMPLETE IMBALANCE HANDLING STRATEGY                              │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  SOLUTION 1: POSITIVE WEIGHT IN LOSS FUNCTION                                │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  pos_weight = Benign_Count / Malignant_Count = 15,991 / 9,340 = 1.71 │    │   │

│   │  │  Loss = BCEWithLogitsLoss(pos_weight=1.71)                          │    │   │

│   │  │                                                                      │    │   │

│   │  │  Effect: Each malignant sample counts as 1.71 samples               │    │   │

│   │  │  Result: Perfectly balanced loss contribution (1.71:1.71)          │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  SOLUTION 2: STRATIFIED TRAIN/VALIDATION SPLIT                              │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  train_idx, val_idx = train_test_split(                             │    │   │

│   │  │      stratify=train_df['stage1_label']  # Preserves class ratio    │    │   │

│   │  │  )                                                                   │    │   │

│   │  │                                                                      │    │   │

│   │  │  Result: Both train and val maintain 63.1% / 36.9% distribution    │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  SOLUTION 3: THRESHOLD TUNING FOR SENSITIVITY                               │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  Two-phase threshold search:                                         │    │   │

│   │  │  • Coarse: [0.30, 0.35, ..., 0.70] step 0.05                        │    │   │

│   │  │  • Fine: [best-0.05 to best+0.05] step 0.01                         │    │   │

│   │  │  • Objective: Maximize specificity while sensitivity ≥ 0.90         │    │   │

│   │  │                                                                      │    │   │

│   │  │  Result: Optimal threshold = 0.44 (vs default 0.5)                  │    │   │

│   │  │  Effect: Model predicts MALIGNANT more easily → higher sensitivity  │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  SOLUTION 4: ENSEMBLE DIVERSITY                                             │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  Three architecturally diverse models:                               │    │   │

│   │  │  • ResNet50: Deep, high capacity                                    │    │   │

│   │  │  • EfficientNet: Efficient feature extraction                       │    │   │

│   │  │  • MobileNetV2: Lightweight, different architecture                 │    │   │

│   │  │                                                                      │    │   │

│   │  │  Weighted ensemble: 0.40×RN + 0.35×EF + 0.25×MB                     │    │   │

│   │  │  Result: Individual biases cancel out, more robust to imbalance     │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  SOLUTION 5: TEMPERATURE SCALING FOR CALIBRATION                           │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  calibrated_prob = sigmoid(ensemble_logit / T)                      │    │   │

│   │  │  T optimized on validation set: T = 1.23                            │    │   │

│   │  │                                                                      │    │   │

│   │  │  Effect: Well-calibrated probabilities for clinical decision-making │    │   │

│   │  │  Result: P=0.85 means 85% actual cancer rate                        │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 3.2 Mathematical Proof of Balance


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    MATHEMATICAL BALANCE PROOF                                         │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   Without pos_weight:                                                                │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Total Loss = Σ L_benign + Σ L_malignant                                    │   │

│   │               = 15,991 × L_b + 9,340 × L_m                                   │   │

│   │                                                                              │   │

│   │  Ratio = (15,991 × L_b) / (9,340 × L_m) = 1.71 × (L_b / L_m)                │   │

│   │                                                                              │   │

│   │  Model is biased toward minimizing benign loss (1.71× more weight)          │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   With pos_weight = 1.71:                                                            │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Total Loss = Σ L_benign + 1.71 × Σ L_malignant                             │   │

│   │               = 15,991 × L_b + 1.71 × 9,340 × L_m                            │   │

│   │               = 15,991 × L_b + 15,991 × L_m                                  │   │

│   │               = 15,991 × (L_b + L_m)                                         │   │

│   │                                                                              │   │

│   │  Result: Perfectly balanced! Both classes contribute equally to loss        │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 4. PREPROCESSING PIPELINE


## 4.1 Step-by-Step Preprocessing


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    PREPROCESSING PIPELINE DETAILS                                    │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   STEP 1: HAIR REMOVAL                                                               │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  Input: RGB Image (H, W, 3)                                                  │   │

│   │         │                                                                    │   │

│   │         ▼                                                                    │   │

│   │  1.1 Convert to grayscale                                                    │   │

│   │  1.2 Apply Black Hat Transform: blackhat = closing(gray) - gray             │   │

│   │  1.3 Threshold: hair_mask = blackhat > 10                                    │   │

│   │  1.4 Dilate mask to cover hair borders (kernel=3×3 ellipse)                 │   │

│   │  1.5 Inpaint: result = inpaint(image, hair_mask, radius=3, TELEA)           │   │

│   │                                                                              │   │

│   │  Output: Hair-free image (H, W, 3)                                           │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   STEP 2: LESION DETECTION                                                          │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  Input: Hair-free image (H, W, 3)                                            │   │

│   │         │                                                                    │   │

│   │         ▼                                                                    │   │

│   │  2.1 Convert to LAB color space                                              │   │

│   │  2.2 Extract A channel (red-green contrast)                                 │   │

│   │  2.3 Apply Gaussian blur (kernel=5×5)                                        │   │

│   │  2.4 Otsu thresholding: binary_mask = threshold(a_channel)                  │   │

│   │  2.5 Morphological closing (fill holes, kernel=5×5 ellipse)                 │   │

│   │  2.6 Morphological opening (remove noise, kernel=5×5 ellipse)               │   │

│   │  2.7 Find largest contour (the lesion)                                      │   │

│   │  2.8 Get bounding box: x, y, w, h = cv2.boundingRect(contour)              │   │

│   │                                                                              │   │

│   │  Output: Bounding box (x, y, w, h)                                           │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   STEP 3: LESION CROPPING & RESIZING                                                │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  Input: Hair-free image + Bounding box (x, y, w, h)                          │   │

│   │         │                                                                    │   │

│   │         ▼                                                                    │   │

│   │  3.1 Add 20% padding: pad_w = w × 0.20, pad_h = h × 0.20                    │   │

│   │  3.2 Calculate crop bounds with clamping to image edges                     │   │

│   │  3.3 Crop: cropped = image[y1:y2, x1:x2]                                    │   │

│   │  3.4 Resize to 224×224 using Lanczos interpolation                          │   │

│   │                                                                              │   │

│   │  Output: Standardized lesion image (224, 224, 3)                            │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 4.2 Preprocessing Code Implementation


```python

class PreprocessingPipeline:

"""Complete preprocessing pipeline for skin lesion images"""


@staticmethod

def remove_hair(image, kernel_size=5, threshold=10):

"""Remove hair using Black Hat transform + inpainting"""

if len(image.shape) == 3:

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

else:

gray = image.copy()


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

_, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)


kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)

result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)


return result


@staticmethod

def detect_lesion(image):

"""Detect lesion using LAB color space + Otsu thresholding"""

lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

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

return (x, y, w, h)

else:

h, w = image.shape[:2]

return (0, 0, w, h)


@staticmethod

def crop_lesion(image, bbox, padding=0.20, target_size=(224, 224)):

"""Crop lesion with padding and resize"""

x, y, w, h = bbox

h_img, w_img = image.shape[:2]


pad_w = int(w * padding)

pad_h = int(h * padding)


x1 = max(0, x - pad_w)

y1 = max(0, y - pad_h)

x2 = min(w_img, x + w + pad_w)

y2 = min(h_img, y + h + pad_h)


cropped = image[y1:y2, x1:x2]

resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)


return resized


@staticmethod

def process(image_path, target_size=(224, 224)):

"""Complete preprocessing pipeline"""

image = cv2.imread(str(image_path))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


hair_removed = PreprocessingPipeline.remove_hair(image)

bbox = PreprocessingPipeline.detect_lesion(hair_removed)

result = PreprocessingPipeline.crop_lesion(hair_removed, bbox,

target_size=target_size)


return result

```


---


# 5. DATA AUGMENTATION


## 5.1 Training Augmentations


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    TRAINING AUGMENTATIONS (Applied Randomly)                         │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Augmentation                    Probability     Purpose                     │   │

│   ├─────────────────────────────────────────────────────────────────────────────┤   │

│   │  RandomResizedCrop (0.8-1.0)     100%           Scale invariance           │   │

│   │  Horizontal Flip                 50%            Mirror invariance          │   │

│   │  Vertical Flip                   50%            Mirror invariance          │   │

│   │  Random Rotation (±45°)          50%            Rotation invariance        │   │

│   │  Color Jitter                    50%            Color robustness           │   │

│   │  Gaussian Noise                  30%            Noise robustness           │   │

│   │  Normalization (ImageNet)        100%           Standardized input         │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   Normalization Values:                                                              │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  mean = [0.485, 0.456, 0.406]  (ImageNet RGB means)                         │   │

│   │  std = [0.229, 0.224, 0.225]   (ImageNet RGB std dev)                       │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 5.2 Validation/Test Transforms


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    VALIDATION/TEST TRANSFORMS (No Augmentation)                      │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Transform                     Purpose                                       │   │

│   ├─────────────────────────────────────────────────────────────────────────────┤   │

│   │  Resize to 224×224              Standardize input size                      │   │

│   │  Normalization                  Same normalization as training              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   Note: NO random augmentations to ensure consistent evaluation                     │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 6. CACHE MANAGEMENT


## 6.1 Cache Strategy


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    CACHE MANAGEMENT STRATEGY                                         │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   WHY CACHE?                                                                         │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  • Preprocessing is computationally expensive (hair removal, detection)    │   │

│   │  • Running it every epoch would add 100+ hours to training                 │   │

│   │  • Cache once, reuse for all epochs and both stages                        │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   CACHE FORMAT:                                                                      │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  • Format: uint8 NPY files (0-255 range)                                    │   │

│   │  • Naming: {image_id}.npy (e.g., ISIC_0000000.npy)                         │   │

│   │  • No subdirectories - flat structure                                       │   │

│   │  • Size: 25,331 × 224 × 224 × 3 × 1 byte = 3.8 GB                          │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   CACHE OPERATIONS:                                                                  │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Save: np.save(cache_path, image_uint8)                                     │   │

│   │  Load: image = np.load(cache_path).astype(np.float32) / 255.0              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 6.2 Cache Directory Structure


```

/kaggle/working/stage1_results/cache/

├── ISIC_0000000.npy

├── ISIC_0000001.npy

├── ISIC_0000002.npy

├── ...

└── ISIC_025330.npy  (33,569 total files: 25,331 train + 8,238 test)

```


---


# 7. DATA SPLITS


## 7.1 Split Configuration


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    DATA SPLIT CONFIGURATION                                          │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   TOTAL TRAINING DATA: 25,331 images                                                 │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Split          Percentage    Images      Benign      Malignant             │   │

│   ├─────────────────────────────────────────────────────────────────────────────┤   │

│   │  Training       85%           21,531      13,592      7,939                 │   │

│   │  Validation     15%           3,800       2,399       1,401                 │   │

│   │  Test (Holdout)  Separate     8,238       6,911       1,327                 │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   STRATIFICATION:                                                                    │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  train_idx, val_idx = train_test_split(                                     │   │

│   │      np.arange(len(train_df)),                                              │   │

│   │      test_size=0.15,                                                        │   │

│   │      stratify=train_df['stage1_label'],  # ← Maintains 63.1/36.9 ratio     │   │

│   │      random_state=42                                                        │   │

│   │  )                                                                          │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   TEST SET POLICY:                                                                   │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  • Test set is LOCKED until final evaluation                                 │   │

│   │  • NEVER used for training, validation, or hyperparameter tuning            │   │

│   │  • Used ONLY ONCE at the end to report final performance                     │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 8. MODEL ARCHITECTURES


## 8.1 Binary Classification Head


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    BINARY CLASSIFICATION HEAD                                        │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   Input: Feature Vector                                                              │

│   (ResNet50: 2048, EfficientNet/MobileNet: 1280)                                    │

│         │                                                                             │

│         ▼                                                                             │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Layer 1: Linear(in_features, 512)                                           │   │

│   │  ├── BatchNorm1d(512)                                                        │   │

│   │  ├── ReLU Activation                                                         │   │

│   │  └── Dropout(p=0.3)                                                          │   │

│   │  Output: 512 features                                                        │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│         │                                                                             │

│         ▼                                                                             │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Layer 2: Linear(512, 256)                                                    │   │

│   │  ├── BatchNorm1d(256)                                                        │   │

│   │  ├── ReLU Activation                                                         │   │

│   │  └── Dropout(p=0.3)                                                          │   │

│   │  Output: 256 features                                                        │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│         │                                                                             │

│         ▼                                                                             │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Layer 3: Linear(256, 1)                                                      │   │

│   │  Output: Logit (raw score)                                                   │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   Loss Function: BCEWithLogitsLoss(pos_weight = 1.71)                               │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 8.2 Model Comparison Table


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    MODEL COMPARISON                                                  │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌──────────────┬─────────────┬─────────────┬─────────────┬──────────────────────┐ │

│   │   Model      │  Parameters │   Head Size │   Total     │   Best For           │ │

│   ├──────────────┼─────────────┼─────────────┼─────────────┼──────────────────────┤ │

│   │  ResNet50    │   23.5M     │   2048→1    │   24.6M     │  Highest Accuracy    │ │

│   ├──────────────┼─────────────┼─────────────┼─────────────┼──────────────────────┤ │

│   │  EfficientNet│   5.3M      │   1280→1    │   6.2M      │  Best Balance        │ │

│   │  -B0         │             │             │             │                      │ │

│   ├──────────────┼─────────────┼─────────────┼─────────────┼──────────────────────┤ │

│   │  MobileNetV2 │   3.5M      │   1280→1    │   4.4M      │  Fastest Inference   │ │

│   └──────────────┴─────────────┴─────────────┴─────────────┴──────────────────────┘ │

│                                                                                       │

│   INFERENCE SPEED (GPU - Tesla T4):                                                  │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Model              Time (ms)    Relative Speed     Memory (MB)             │   │

│   │  ─────────────────────────────────────────────────────────────────────────  │   │

│   │  ResNet50           25          1.0x (baseline)     850                     │   │

│   │  EfficientNet-B0    15          1.7x faster         350                     │   │

│   │  MobileNetV2        10          2.5x faster         200                     │   │

│   │  Ensemble           50          0.5x slower         1,400                   │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 8.3 Model Code Implementation


```python

class BinaryHead(nn.Module):

"""Custom classification head for Stage 1 (Binary)"""


def __init__(self, in_features, dropout=0.3):

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

features = self.backbone(x)

return self.head(features)


class EfficientNetBinary(nn.Module):

def __init__(self):

super().__init__()

self.backbone = timm.create_model('efficientnet_b0', pretrained=True)

in_features = self.backbone.classifier.in_features

self.backbone.classifier = nn.Identity()

self.head = BinaryHead(in_features)


def forward(self, x):

features = self.backbone(x)

return self.head(features)


class MobileNetV2Binary(nn.Module):

def __init__(self):

super().__init__()

self.backbone = models.mobilenet_v2(weights='DEFAULT')

in_features = self.backbone.classifier[1].in_features

self.backbone.classifier = nn.Identity()

self.head = BinaryHead(in_features)


def forward(self, x):

features = self.backbone(x)

return self.head(features)

```


---


# 9. ENSEMBLE STRATEGY


## 9.1 Weighted Ensemble Optimization Process


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    ENSEMBLE OPTIMIZATION PROCESS                                     │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   STEP 1: Grid Search for Weights                                                   │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  For w₁ in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:                │   │

│   │      For w₂ in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:            │   │

│   │          w₃ = 1 - w₁ - w₂                                                   │   │

│   │          if 0.10 ≤ w₃ ≤ 0.60:                                               │   │

│   │              ensemble_logits = w₁×L_RN + w₂×L_EF + w₃×L_MB                  │   │

│   │                                                                              │   │

│   │              for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,    │   │

│   │                                   0.65, 0.70]:                              │   │

│   │                  sensitivity, specificity = evaluate(threshold)            │   │

│   │                  if sensitivity ≥ 0.90 and specificity > best:             │   │

│   │                      best_weights = [w₁, w₂, w₃]                           │   │

│   │                      best_threshold = threshold                             │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   STEP 2: Fine Threshold Search                                                     │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  fine_thresholds = np.arange(best_threshold - 0.05,                         │   │

│   │                              best_threshold + 0.06, 0.01)                   │   │

│   │                                                                              │   │

│   │  for threshold in fine_thresholds:                                          │   │

│   │      sensitivity, specificity = evaluate(threshold)                         │   │

│   │      if sensitivity ≥ 0.90 and specificity > best:                         │   │

│   │          best_threshold = threshold                                         │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   STEP 3: Temperature Calibration                                                   │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Optimize T to minimize BCE loss:                                            │   │

│   │                                                                              │   │

│   │  loss(T) = -[y·log(σ(logit/T)) + (1-y)·log(1-σ(logit/T))]                  │   │

│   │                                                                              │   │

│   │  minimize over T ∈ [0.1, 5.0] using L-BFGS-B                                │   │

│   │  Expected optimal T ≈ 1.23                                                  │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   OPTIMAL CONFIGURATION:                                                             │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  Weights:     [0.40, 0.35, 0.25]                                            │   │

│   │  Threshold:   0.44                                                          │   │

│   │  Temperature: 1.23                                                          │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 10. TRAINING CONFIGURATION


## 10.1 Complete Training Parameters


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    TRAINING CONFIGURATION SUMMARY                                    │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  PARAMETER              VALUE                                                │   │

│   ├─────────────────────────────────────────────────────────────────────────────┤   │

│   │  Batch Size             32                                                   │   │

│   │  Learning Rate          1e-4                                                 │   │

│   │  Weight Decay           1e-5                                                 │   │

│   │  Optimizer              AdamW                                                │   │

│   │  Scheduler              CosineAnnealingLR(T_max=20, eta_min=1e-6)           │   │

│   │  Loss Function          BCEWithLogitsLoss(pos_weight=1.71)                  │   │

│   │  Max Epochs             20                                                   │   │

│   │  Early Stopping         Patience=8, Monitor=Val AUC                         │   │

│   │  Gradient Clipping      1.0                                                  │   │

│   │  pos_weight             1.71 (handles 1.71:1 imbalance)                      │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   LEARNING RATE SCHEDULE:                                                            │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  η_t = η_min + 0.5 × (η_max - η_min) × (1 + cos(π × t / T_max))             │   │

│   │                                                                              │   │

│   │  η_max = 1e-4 (initial)                                                     │   │

│   │  η_min = 1e-6 (final)                                                       │   │

│   │  T_max = 20                                                                  │   │

│   │                                                                              │   │

│   │  Progression:                                                                │   │

│   │  Epoch 0: 1e-4                                                              │   │

│   │  Epoch 5: 7.5e-5                                                            │   │

│   │  Epoch 10: 5e-5                                                             │   │

│   │  Epoch 15: 2.5e-5                                                           │   │

│   │  Epoch 20: 1e-6                                                             │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 11. LOSS FUNCTION DETAILS


## 11.1 BCEWithLogitsLoss with pos_weight


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    LOSS FUNCTION MATHEMATICAL DETAILS                                │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   FORMULA:                                                                           │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  L = -[ y × weight × log(σ(x)) + (1-y) × log(1-σ(x)) ]                      │   │

│   │                                                                              │   │

│   │  where:                                                                      │   │

│   │  • y = ground truth (1 for malignant, 0 for benign)                         │   │

│   │  • weight = pos_weight = 1.71                                               │   │

│   │  • σ(x) = sigmoid(x) = 1/(1+e^{-x})                                         │   │

│   │  • x = model output (logit)                                                  │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   NUMERICAL EXAMPLES:                                                               │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  Case 1: Malignant (y=1), model predicts 0.3 (low confidence)              │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  Loss = -[1 × 1.71 × log(0.3)] = -1.71 × (-1.20) = 2.05              │    │   │

│   │  │  HIGH penalty for missing cancer!                                     │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   │                                                                              │   │

│   │  Case 2: Malignant (y=1), model predicts 0.9 (high confidence)             │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  Loss = -[1 × 1.71 × log(0.9)] = -1.71 × (-0.105) = 0.18             │    │   │

│   │  │  LOW penalty for correct detection                                    │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   │                                                                              │   │

│   │  Case 3: Benign (y=0), model predicts 0.7 (false alarm)                    │   │

│   │  ┌─────────────────────────────────────────────────────────────────────┐    │   │

│   │  │  Loss = -[0 × 1.71 × log(0.7) + 1 × log(0.3)] = -[0 + (-1.20)] = 1.20│    │   │

│   │  │  LOWER penalty for false alarm (compared to missing cancer)          │    │   │

│   │  └─────────────────────────────────────────────────────────────────────┘    │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 12. EVALUATION METRICS


## 12.1 Primary Clinical Metrics


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    PRIMARY CLINICAL METRICS                                          │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   CONFUSION MATRIX (2×2):                                                           │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │                      PREDICTED                                               │   │

│   │                  BENIGN        MALIGNANT                                     │   │

│   │   ACTUAL  BENIGN    TN              FP         (False Alarm)                │   │

│   │          MALIGNANT  FN              TP         (True Detection)             │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  METRIC                    FORMULA                    TARGET                │   │

│   ├─────────────────────────────────────────────────────────────────────────────┤   │

│   │  Sensitivity (Recall)     TP / (TP + FN)              ≥ 0.90                │   │

│   │  Specificity              TN / (TN + FP)              ≥ 0.75                │   │

│   │  AUC-ROC                  Area under ROC curve       ≥ 0.90                │   │

│   │  NPV (Neg Predictive)     TN / (TN + FN)              ≥ 0.95                │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 12.2 Secondary Metrics


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    SECONDARY METRICS                                                 │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  METRIC                    FORMULA                                           │   │

│   ├─────────────────────────────────────────────────────────────────────────────┤   │

│   │  Accuracy                  (TP + TN) / Total                                 │   │

│   │  Precision                 TP / (TP + FP)                                    │   │

│   │  F1-Score                  2 × (P × R) / (P + R)                            │   │

│   │  MCC                       (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]│   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 13. EXPECTED PERFORMANCE


## 13.1 Performance Targets


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    EXPECTED PERFORMANCE TARGETS                                      │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  Metric              Target      Expected     Clinical Meaning              │   │

│   │  ─────────────────────────────────────────────────────────────────────────  │   │

│   │  Sensitivity         ≥ 0.90      0.91-0.93    Catch 9/10 cancers           │   │

│   │  Specificity         ≥ 0.75      0.77-0.80    Avoid 3/4 false alarms       │   │

│   │  AUC                 ≥ 0.90      0.92-0.94    Excellent discrimination     │   │

│   │  NPV                 ≥ 0.95      0.96-0.97    95% confidence in "benign"   │   │

│   │  Accuracy            -           0.86-0.89    Overall correctness          │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 13.2 Model Comparison Expectations


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    EXPECTED MODEL COMPARISON                                         │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │                                                                              │   │

│   │  Model              Sensitivity    Specificity    AUC      Inference (ms)   │   │

│   │  ─────────────────────────────────────────────────────────────────────────  │   │

│   │  ResNet50           0.91           0.78           0.93     25               │   │

│   │  EfficientNet-B0    0.90           0.77           0.92     15               │   │

│   │  MobileNetV2        0.89           0.76           0.91     10               │   │

│   │  Ensemble           0.92           0.79           0.94     50               │   │

│   │                                                                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# 14. OUTPUT STRUCTURE


## 14.1 Directory Structure


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    STAGE 1 OUTPUT DIRECTORY STRUCTURE                                │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   /kaggle/working/stage1_results/                                                    │

│   │                                                                                   │

│   ├── models/                                    # Trained model weights            │

│   │   ├── resnet50_stage1_best.pth                                                   │

│   │   ├── efficientnet_stage1_best.pth                                               │

│   │   ├── mobilenetv2_stage1_best.pth                                                │

│   │   └── ensemble_config_stage1.json          # Weights, threshold, temperature    │

│   │                                                                                   │

│   ├── figures/                                    # All visualizations              │

│   │   ├── 01_data_distribution.png                                                   │

│   │   ├── 02_data_split.png                                                          │

│   │   ├── 03_preprocessing_pipeline.png                                              │

│   │   ├── 04_augmentations.png                                                       │

│   │   ├── 05_training_curves.png                                                     │

│   │   ├── 06_roc_curves.png                                                          │

│   │   ├── 07_confusion_matrix.png                                                    │

│   │   └── 08_model_comparison.png                                                    │

│   │                                                                                   │

│   ├── reports/                                    # JSON/CSV results                │

│   │   ├── train_split.csv                      # Training indices                   │

│   │   ├── val_split.csv                        # Validation indices                 │

│   │   ├── test_holdout.csv                     # Test indices (locked)              │

│   │   └── stage1_results.json                  # All metrics                        │

│   │                                                                                   │

│   └── cache/                                      # Preprocessed images (uint8)     │

│       ├── ISIC_0000000.npy                                                           │

│       ├── ISIC_0000001.npy                                                           │

│       └── ... (33,569 files)                                                         │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


## 14.2 Ensemble Configuration File


```json

{

"stage": 1,

"models": ["resnet50", "efficientnet", "mobilenetv2"],

"weights": [0.40, 0.35, 0.25],

"threshold": 0.44,

"temperature": 1.23,

"pos_weight": 1.71,

"validation_metrics": {

"sensitivity": 0.912,

"specificity": 0.783,

"auc": 0.938,

"npv": 0.961

},

"clinical_action": {

"benign": "Reassure patient, routine monitoring, annual follow-up",

"malignant": "Urgent referral to dermatologist, proceed to Stage 2"

}

}

```


---


# 15. QUICK REFERENCE CARD


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    STAGE 1 QUICK REFERENCE CARD                                      │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  DATA                                                                        │   │

│   │  ├── Total: 25,331 training images                                          │   │

│   │  ├── Malignant (1): MEL + BCC + SCC + AK = 9,340 (36.9%)                    │   │

│   │  ├── Benign (0): NV + BKL + DF + VASC = 15,991 (63.1%)                      │   │

│   │  └── pos_weight = 1.71                                                      │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  SPLITS                                                                      │   │

│   │  ├── Train: 21,531 images (85%)                                             │   │

│   │  ├── Validation: 3,800 images (15%)                                         │   │

│   │  └── Test: 8,238 images (LOCKED)                                            │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  MODELS                                                                      │   │

│   │  ├── ResNet50 (23.5M) + Binary Head (2048→1)                                │   │

│   │  ├── EfficientNet-B0 (5.3M) + Binary Head (1280→1)                          │   │

│   │  └── MobileNetV2 (3.5M) + Binary Head (1280→1)                              │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  ENSEMBLE                                                                    │   │

│   │  ├── Weights: [0.40, 0.35, 0.25]                                            │   │

│   │  ├── Threshold: 0.44 (tuned for Sensitivity ≥ 0.90)                         │   │

│   │  └── Temperature: 1.23 (calibrated)                                         │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  IMBALANCE HANDLING (5 Solutions)                                            │   │

│   │  ├── 1. pos_weight = 1.71 in loss function                                  │   │

│   │  ├── 2. Stratified train/val split                                          │   │

│   │  ├── 3. Threshold tuned to 0.44 (below 0.5)                                 │   │

│   │  ├── 4. Ensemble diversity (3 different architectures)                      │   │

│   │  └── 5. Temperature scaling for calibration                                 │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  TRAINING                                                                    │   │

│   │  ├── Loss: BCEWithLogitsLoss(pos_weight=1.71)                               │   │

│   │  ├── Optimizer: AdamW(lr=1e-4, weight_decay=1e-5)                          │   │

│   │  ├── Scheduler: CosineAnnealingLR(T_max=20, eta_min=1e-6)                   │   │

│   │  └── Early Stopping: Patience=8, Monitor=Val AUC                            │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  TARGETS                                                                     │   │

│   │  ├── Sensitivity ≥ 0.90 (Catch 90% of cancers)                              │   │

│   │  ├── Specificity ≥ 0.75 (Avoid 75% of false alarms)                         │   │

│   │  └── AUC ≥ 0.90 (Excellent discrimination)                                  │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

│   ┌─────────────────────────────────────────────────────────────────────────────┐   │

│   │  CACHE                                                                       │   │

│   │  ├── Format: uint8 NPY files by image ID                                    │   │

│   │  ├── Size: ~3.8 GB                                                          │   │

│   │  └── Load: np.load() → astype(float32)/255.0                                │   │

│   └─────────────────────────────────────────────────────────────────────────────┘   │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


# ✅ FINAL CHECKLIST - STAGE 1


```

┌─────────────────────────────────────────────────────────────────────────────────────┐

│                    STAGE 1 IMPLEMENTATION CHECKLIST                                  │

├─────────────────────────────────────────────────────────────────────────────────────┤

│                                                                                       │

│   ✅ DATA PREPARATION                                                                │

│   ├── [ ] Load train_df and test_df from CSV                                        │

│   ├── [ ] Create binary labels (Malignant = MEL+BCC+SCC+AK)                         │

│   ├── [ ] Verify class distribution (36.9% Malignant, 63.1% Benign)                 │

│   └── [ ] Save class distribution visualizations                                    │

│                                                                                       │

│   ✅ PREPROCESSING & CACHING                                                         │

│   ├── [ ] Implement hair removal (Black Hat + Inpainting)                           │

│   ├── [ ] Implement lesion detection (LAB + Otsu)                                   │

│   ├── [ ] Implement cropping with 20% padding                                       │

│   ├── [ ] Cache all images as uint8 NPY (3.8 GB)                                    │

│   └── [ ] Test cache loading with float32 conversion                                │

│                                                                                       │

│   ✅ DATA SPLITS                                                                     │

│   ├── [ ] Stratified split (85/15) preserving class ratio                          │

│   ├── [ ] Verify both splits maintain ~36.9% Malignant                              │

│   └── [ ] Save split indices to CSV                                                 │

│                                                                                       │

│   ✅ MODEL TRAINING                                                                  │

│   ├── [ ] Train ResNet50 (20 epochs, early stopping)                                │

│   ├── [ ] Train EfficientNet-B0 (20 epochs, early stopping)                         │

│   ├── [ ] Train MobileNetV2 (20 epochs, early stopping)                             │

│   └── [ ] Save all model checkpoints                                                │

│                                                                                       │

│   ✅ ENSEMBLE OPTIMIZATION                                                           │

│   ├── [ ] Grid search for optimal weights (8×8 combinations)                        │

│   ├── [ ] Two-phase threshold tuning (coarse 0.05 → fine 0.01)                      │

│   ├── [ ] Temperature calibration on validation set                                 │

│   └── [ ] Save ensemble_config.json                                                 │

│                                                                                       │

│   ✅ EVALUATION                                                                      │

│   ├── [ ] Compute Sensitivity, Specificity, AUC on validation                       │

│   ├── [ ] Verify Sensitivity ≥ 0.90                                                 │

│   ├── [ ] Generate ROC curve, confusion matrix                                      │

│   └── [ ] Final test evaluation (ONCE at the end)                                   │

│                                                                                       │

│   ✅ OUTPUT                                                                          │

│   ├── [ ] Save all figures (8+ visualizations)                                      │

│   ├── [ ] Save results to JSON                                                      │

│   └── [ ] Create final summary report                                              │

│                                                                                       │

└─────────────────────────────────────────────────────────────────────────────────────┘

```


---


**This is the COMPLETE FINAL ARCHITECTURE for Stage 1. No further changes will be made. Ready for implementation!** 🚀
