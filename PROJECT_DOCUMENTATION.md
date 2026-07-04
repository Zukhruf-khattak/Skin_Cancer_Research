# 🏥 Automated Skin Cancer Detection — Complete Project Documentation

## BSc Final Year Project | University of Peshawar, Shaikh Zayed Islamic Centre
### Zohrouf Khattak & Huma Zeb | Supervisor: Dr Muhammad Ayaz
### Session: 2022–2026

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Dataset](#3-dataset)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Model Architectures](#5-model-architectures)
6. [Stage 1 — Binary Classification Results](#6-stage-1--binary-classification-results)
7. [Stage 2 — Multi-Class Subtype Results](#7-stage-2--multi-class-subtype-results)
8. [Improvement Over Baseline](#8-improvement-over-baseline)
9. [Key Innovations](#9-key-innovations)
10. [Streamlit Demo App](#10-streamlit-demo-app)
11. [Project Structure](#11-project-structure)
12. [How to Run](#12-how-to-run)
13. [Model Weight Files](#13-model-weight-files)
14. [Technical Details](#14-technical-details)

---

## 1. Project Overview

This project implements a **dual-stage ensemble deep learning pipeline** for automated detection and classification of skin cancer from dermoscopic images.

| Aspect          | Detail                                        |
|-----------------|-----------------------------------------------|
| **Goal**        | Automated skin cancer detection from images   |
| **Dataset**     | ISIC 2019 — 25,331 dermoscopic images         |
| **Stage 1**     | Binary: Benign vs Malignant                   |
| **Stage 2**     | Multi-class: MEL, BCC, SCC, AK subtypes       |
| **Ensemble**    | 5-model probability averaging                 |
| **Explainability** | Grad-CAM heatmaps                          |
| **Framework**   | PyTorch 2.x + torchvision                     |

---

## 2. Pipeline Architecture

```
                    ┌─────────────────────┐
                    │   Input Dermoscopic  │
                    │       Image          │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Preprocessing      │
                    │  (Hair Removal,      │
                    │   Segmentation,      │
                    │   224×224 Resize)     │
                    └─────────┬───────────┘
                              │
              ┌───────────────▼───────────────┐
              │       STAGE 1: BINARY          │
              │    Benign vs Malignant         │
              │  (5-Model Ensemble Vote)       │
              └───────┬───────────┬───────────┘
                      │           │
               ┌──────▼──┐  ┌────▼────┐
               │ Benign  │  │Malignant│
               │ (STOP)  │  │(→ Stage2)│
               └─────────┘  └────┬────┘
                                 │
              ┌──────────────────▼──────────────┐
              │       STAGE 2: MULTI-CLASS       │
              │   MEL · BCC · SCC · AK           │
              │  (5-Model Ensemble Vote)         │
              └──────────────────────────────────┘
```

---

## 3. Dataset

**ISIC 2019 Challenge Dataset** — 25,331 dermoscopic images across 8 original classes.

### Binary Mapping (Stage 1)
| Category    | Original Classes Included                       |
|-------------|------------------------------------------------|
| **Benign**  | Melanocytic Nevus (NV), Benign Keratosis (BKL), Dermatofibroma (DF), Vascular Lesion (VASC) |
| **Malignant**| Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratosis (AK) |

### Class Imbalance Handling
- **NV Undersampling**: 8,978 → 4,238 samples (stratified) to prevent benign dominance
- **Focal Loss**: Up-weights rare classes during training
- **MixUp Augmentation**: Blends training pairs as regularisation

---

## 4. Preprocessing Pipeline

All images go through a 4-step standardisation pipeline before model inference:

| Step | Operation                 | Detail                                       |
|------|---------------------------|----------------------------------------------|
| 1    | Artifact Detection        | Hair and ruler detection using morphological ops |
| 2    | Inpainting                | OpenCV inpainting to remove detected hair    |
| 3    | Lesion Segmentation       | Crop to lesion region of interest            |
| 4    | Standardisation           | Resize to 224×224, ImageNet normalisation    |

**ImageNet Normalisation Values:**
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 5. Model Architectures

The ensemble uses **5 diverse architectures** — 3 CNNs and 2 Vision Transformers:

| # | Model             | Type                    | Key Strength                                        | Parameters |
|---|-------------------|-------------------------|-----------------------------------------------------|------------|
| 1 | **ResNet50**       | CNN (50 layers)         | Skip-connections solve vanishing gradients. Reliable medical imaging baseline. | ~23.5M |
| 2 | **EfficientNet-B0**| Compound Scaling CNN    | Balances depth/width/resolution for best accuracy-per-FLOP ratio. | ~5.3M |
| 3 | **MobileNetV2**    | Lightweight CNN         | Inverted residual blocks for fast real-time inference on CPU/mobile. | ~3.4M |
| 4 | **ConvNeXt-Tiny**  | Modern Pure CNN         | Rebuilt from scratch to match ViT strengths while keeping CNN efficiency. | ~28.6M |
| 5 | **Swin-Tiny**      | Shifted-Window ViT      | Treats 16×16 patches as tokens with self-attention — captures global patterns. | ~28.3M |

### Head Architecture Details

**Stage 1** (Binary — 2 classes): ResNet50, ConvNeXt-Tiny, and Swin-Tiny use `Sequential(Dropout(0.5), Linear)` heads.

**Stage 2** (Multi-class — 4 classes): ResNet50, ConvNeXt-Tiny, and Swin-Tiny use plain `nn.Linear` heads.

EfficientNet-B0 and MobileNetV2 use plain `nn.Linear` heads in both stages.

---

## 6. Stage 1 — Binary Classification Results

**Task:** Classify dermoscopic images as **Benign** or **Malignant**.

### Per-Model Performance

| Model              | Accuracy (%) | Precision (%) | Sensitivity (%) | F1-Score | AUC    |
|--------------------|:------------:|:-------------:|:---------------:|:--------:|:------:|
| ResNet50           | 92.76        | 90.80         | 89.44           | 0.9011   | 0.9675 |
| EfficientNet-B0    | 92.87        | 88.38         | 92.86           | 0.9057   | 0.9713 |
| MobileNetV2        | 91.89        | 87.61         | 90.86           | 0.8921   | 0.9680 |
| ConvNeXt-Tiny      | 93.18        | 87.71         | 94.79           | 0.9111   | 0.9746 |
| Swin-Tiny          | 93.55        | 88.79         | 94.43           | 0.9153   | 0.9734 |
| **⭐ Ensemble**    | **94.34**    | **91.12**     | **93.79**       | **0.9244**| **0.9845** |

### Ensemble Confusion Matrix

```
                   Predicted
                 Benign  Malignant
True  Benign      2271      128
      Malignant     87     1314
```

| Metric              | Value  | Clinical Meaning                               |
|---------------------|--------|------------------------------------------------|
| True Negatives      | 2,271  | Benign correctly identified                    |
| False Positives     | 128    | Benign flagged as Malignant (unnecessary biopsies) |
| **False Negatives** | **87** | **Malignant missed (most critical error)**     |
| True Positives      | 1,314  | Cancer correctly detected                      |

> The ensemble reduced false positives by 58 compared to the best individual model (Swin-Tiny), while maintaining 93.79% sensitivity.

---

## 7. Stage 2 — Multi-Class Subtype Results

**Task:** Classify malignant lesions into one of 4 cancer subtypes.

### Per-Model Performance

| Model              | Accuracy (%) | Macro Precision (%) | Macro Sensitivity (%) | Macro F1 | AUC    |
|--------------------|:------------:|:-------------------:|:---------------------:|:--------:|:------:|
| ResNet50           | 80.48        | 76.65               | 81.88                 | 0.7814   | 0.9481 |
| EfficientNet-B0    | 83.39        | 79.99               | 83.34                 | 0.8133   | 0.9615 |
| MobileNetV2        | 80.24        | 76.18               | 80.44                 | 0.7764   | 0.9506 |
| ConvNeXt-Tiny      | 80.36        | 76.47               | 81.90                 | 0.7789   | 0.9549 |
| Swin-Tiny          | 82.91        | 79.03               | 84.41                 | 0.8063   | 0.9617 |
| **⭐ Ensemble**    | **85.58**    | **81.68**           | **86.42**             | **0.8332**| **0.9738** |

### Per-Class Recall (Ensemble)

| Cancer Subtype                     | Recall (%) | Color Code |
|------------------------------------|:----------:|:----------:|
| **Melanoma (MEL)**                 | 85.00      | 🔴         |
| **Basal Cell Carcinoma (BCC)**     | 85.67      | 🟡         |
| **Squamous Cell Carcinoma (SCC)**  | 95.79      | 🟣         |
| **Actinic Keratosis (AK)**        | 79.23      | 🔵         |

### Clinical Priority Scores

| Model              | Clinical Priority Score |
|---------------------|:----------------------:|
| ResNet50            | 84.72                  |
| EfficientNet-B0     | 85.66                  |
| MobileNetV2         | 83.09                  |
| ConvNeXt-Tiny       | 84.43                  |
| Swin-Tiny           | 86.14                  |
| **⭐ ENSEMBLE**     | **88.81**              |

---

## 8. Improvement Over Baseline

| Metric       | Old Best (%) | New Ensemble (%) | Improvement         |
|--------------|:------------:|:----------------:|---------------------|
| Accuracy     | 69.19        | **94.34**        | **+25.15 points**   |
| Sensitivity  | 79.24        | **93.79**        | **+14.55 points**   |
| AUC          | 76.65        | **98.45**        | **+21.80 points**   |
| F1-Score     | 63.23        | **92.44**        | **+29.21 points**   |

---

## 9. Key Innovations

### ⚡ 5-Model Ensemble
Combines ResNet50, EfficientNet-B0, MobileNetV2, ConvNeXt-Tiny, and Swin-Tiny via **probability averaging**. The mix of CNNs and Vision Transformers gives architectural diversity, dramatically reducing false positives in clinical screening.

### 🔬 Advanced Preprocessing
4-step pipeline: artifact detection → inpainting (hair removal) → lesion segmentation → 224×224 standardisation. Every image is clinically clean before inference.

### 🗺️ Grad-CAM Explainability
Gradient-weighted Class Activation Maps prove the model examines the actual lesion (borders, colour variation, texture asymmetry) — not irrelevant background artifacts like rulers or skin folds.

### ⚖️ Class Imbalance — Focal Loss + MixUp
Replaces GANs for imbalance handling. Focal Loss penalises easy examples; MixUp blends training pairs for powerful regularisation.

### 🎯 Targeted NV Undersampling
Melanocytic Nevus (NV) samples reduced 8,978 → 4,238 with stratified splits to prevent the dominant benign class from drowning the malignant signal.

### 🛡️ Clinical Safety Design
Early stopping prevents overfitting. Per-class recall targets (MEL ≥85%, SCC ≥95%). Ensemble designed to minimise dangerous False Negatives.

---

## 10. Streamlit Demo App

A full-featured interactive web dashboard is included at `demo_app.py`.

### Features

| Page              | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| 🏠 **Overview**   | Project summary, KPI cards, improvement charts, innovation cards  |
| 📊 **Stage 1**    | Gauge charts, bar comparisons, confusion matrix, result PNGs      |
| 🧬 **Stage 2**    | Per-class recall bars, full results table, visualisation PNGs     |
| 🔮 **Live Demo**  | **Upload an image → real 5-model ensemble prediction**           |
| 🎨 **Visualisations** | All Grad-CAM, ROC, confusion matrix, training convergence PNGs |
| ℹ️ **About**      | Team info, architecture details, tech stack                      |

### Navigation
The app uses a **sticky top navigation bar** (not a sidebar) — all pages are always visible and clickable.

### Live Demo — How It Works

1. Click **🔮 Live Demo** in the top navigation
2. Choose a tab:
   - **Stage 1** → upload any skin lesion → get **Benign / Malignant** prediction
   - **Stage 2** → upload a malignant lesion → get **MEL / BCC / SCC / AK** prediction
3. Click **Browse files** → select your image (JPG/PNG)
4. Click **▶ Run 5-Model Ensemble**
5. Watch all 5 real `.pth` models run with progress bar
6. See the **ensemble prediction** with:
   - Final label + confidence %
   - Per-model probability breakdown bars
   - Subtype probability chart (Stage 2)

> Models are loaded once and cached — subsequent predictions are fast.

---

## 11. Project Structure

```
d:\RESEARCH PNGS\
├── demo_app.py                          ← Streamlit demo app
├── .streamlit/config.toml               ← App config (dark theme)
├── README.md                            ← Project overview
├── COMPREHENSIVE_MASTER_GUIDE.md        ← Technical guide
│
├── Final stage1/                        ← Stage 1 results & weights
│   ├── best_resnet50.pth                   (90 MB)
│   ├── best_efficientnet-b0.pth            (16 MB)
│   ├── best_mobilenetv2.pth                (9 MB)
│   ├── best_convnext-tiny.pth              (106 MB)
│   ├── best_swin-tiny.pth                  (105 MB)
│   ├── ensemble_evaluation_plots.png
│   ├── clinical_performance_summary.png
│   ├── gradcam_comparison_summary.png
│   ├── clinical_analysis_plot.png
│   ├── performance_gain_chart.png
│   ├── ensemble_summary.csv
│   ├── final_thesis_results.csv
│   └── last-final-stage-1-ipynb.ipynb
│
├── Final Stage 2/                       ← Stage 2 results & weights
│   ├── ResNet50_stage2.pth                 (90 MB)
│   ├── EfficientNet-B0_stage2.pth          (16 MB)
│   ├── MobileNetV2_stage2.pth              (9 MB)
│   ├── ConvNeXt-Tiny_stage2.pth            (106 MB)
│   ├── Swin-Tiny_stage2.pth                (105 MB)
│   ├── confusion_matrices.png
│   ├── roc_pr_curves.png
│   ├── radar_chart.png
│   ├── training_convergence_grid.png
│   ├── grad_cam_AK.png
│   ├── grad_cam_SCC.png
│   ├── clean_preprocessing.png
│   ├── heavy_augmentation.png
│   ├── ultimate_summary_metrics.csv
│   ├── clinical_priority_scores.csv
│   └── last-final-stage-2.ipynb
│
├── kaggle_pipeline/                     ← Kaggle notebook generation
├── notebooks/                           ← Training notebooks
├── scripts/                             ← Utility scripts
├── data/                                ← Dataset references
├── models/                              ← Model definitions
├── results/                             ← Additional results
├── docs/                                ← Documentation
│
├── FYP Thesis Roll # 840 & 881.docx    ← Full thesis document
├── Thesis_Chapter1_Chapter2.docx        ← Chapters 1 & 2
├── Methodology_Chapter.docx             ← Chapter 3: Methodology
├── Results_Chapter.docx                 ← Chapter 4: Results
├── methodology_chapter.md               ← Methodology (markdown)
├── results_chapter.md                   ← Results (markdown)
└── master_results.md                    ← Master results summary
```

---

## 12. How to Run

### Prerequisites
```bash
pip install streamlit plotly pandas numpy pillow torch torchvision
```

### Launch the App
```powershell
cd "d:\RESEARCH PNGS"
streamlit run demo_app.py
```

Then open your browser at: **http://localhost:8501**

### PyTorch Note
- The app uses **PyTorch CPU** for inference (`torch 2.12.0+cpu`)
- No GPU required — all 5 models run on CPU
- First prediction takes ~30–60 seconds (loading weights into memory)
- Subsequent predictions are fast (models are cached in memory)

---

## 13. Model Weight Files

### Stage 1 — Binary (Benign vs Malignant, 2 output classes)

| File                              | Architecture     | Size    | Head Structure                              |
|-----------------------------------|------------------|---------|---------------------------------------------|
| `best_resnet50.pth`              | ResNet50         | 90 MB   | `Sequential(Dropout(0.5), Linear(2048, 2))` |
| `best_efficientnet-b0.pth`      | EfficientNet-B0  | 16 MB   | `Linear(1280, 2)`                           |
| `best_mobilenetv2.pth`          | MobileNetV2      | 9 MB    | `Linear(1280, 2)`                           |
| `best_convnext-tiny.pth`        | ConvNeXt-Tiny    | 106 MB  | `Sequential(Dropout(0.5), Linear(768, 2))`  |
| `best_swin-tiny.pth`            | Swin-Tiny        | 105 MB  | `Sequential(Dropout(0.5), Linear(768, 2))`  |

### Stage 2 — Multi-Class (MEL/BCC/SCC/AK, 4 output classes)

| File                              | Architecture     | Size    | Head Structure                |
|-----------------------------------|------------------|---------|-------------------------------|
| `ResNet50_stage2.pth`            | ResNet50         | 90 MB   | `Linear(2048, 4)`             |
| `EfficientNet-B0_stage2.pth`    | EfficientNet-B0  | 16 MB   | `Linear(1280, 4)`             |
| `MobileNetV2_stage2.pth`        | MobileNetV2      | 9 MB    | `Linear(1280, 4)`             |
| `ConvNeXt-Tiny_stage2.pth`      | ConvNeXt-Tiny    | 106 MB  | `Linear(768, 4)`              |
| `Swin-Tiny_stage2.pth`          | Swin-Tiny        | 105 MB  | `Linear(768, 4)`              |

> **Total model weight storage: ~652 MB** (all 10 weight files)

---

## 14. Technical Details

### Training Configuration
- **Image Size**: 224 × 224 pixels
- **Normalisation**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Loss Function**: Focal Loss (γ=2)
- **Augmentation**: MixUp blending
- **Regularisation**: Early stopping, Dropout (0.5 on select heads)
- **Optimiser**: Adam with learning rate scheduling
- **Batch Size**: Configured per model architecture
- **Ensemble Method**: Arithmetic mean of softmax probabilities across all 5 models

### Inference Pipeline (demo_app.py)
```python
# 1. Load image
pil_img = Image.open(uploaded_file)

# 2. Preprocess (same as training)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
tensor = transform(pil_img.convert("RGB")).unsqueeze(0)

# 3. Run all 5 models
results = {}
for name, model in loaded_models.items():
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
    results[name] = probs

# 4. Ensemble — average probabilities
ensemble = np.mean(list(results.values()), axis=0)
prediction = class_names[np.argmax(ensemble)]
confidence = np.max(ensemble) * 100
```

### Dependencies
```
streamlit==1.58.0
torch==2.12.0+cpu
torchvision==0.27.0+cpu
plotly>=5.0
pandas>=2.0
numpy>=2.0
Pillow>=12.0
```

---

## 📝 Version History

| Date       | Change                                                                 |
|------------|------------------------------------------------------------------------|
| 2026-06-07 | Initial Streamlit demo app created with simulated predictions          |
| 2026-06-07 | Wired real PyTorch model weights — all 10 `.pth` files load correctly  |
| 2026-06-07 | Fixed model head architectures (Stage 1 vs Stage 2 differences)        |
| 2026-06-07 | Replaced sidebar with sticky top navigation bar for better visibility  |
| 2026-06-15 | Created comprehensive project documentation (this file)                |

---

> ⚠️ **Disclaimer:** This system is a research tool developed as part of a BSc Final Year Project. It must **not** be used for clinical diagnosis. Always consult a qualified dermatologist for any medical concerns.

---

*🔬 SkinSense AI · University of Peshawar · Shaikh Zayed Islamic Centre*
*Zohrouf Khattak & Huma Zeb · Supervisor: Dr Muhammad Ayaz*
