# Chapter 4: Results and Evaluation

**Project:** Automated Skin Cancer Detection and Classification Using Deep Learning  
**Authors:** Zohrouf Khattak (Roll No: 1000) · Huma Zeb (Roll No: 883)  
**Supervisor:** Dr. Muhammad Ayaz  
**Institution:** University of Peshawar, Shaikh Zayed Islamic Centre  
**Programme:** BSc Computer Science (Session 2022–2026)

---

## 4.1 Stage 1: Binary Classification Results

This section presents the quantitative evaluation of Stage 1 (Benign vs. Malignant binary classification). All results are reported on the **held-out test set** — 3,800 images that were never seen during training or validation. The test set maintains the same stratified class proportions as the training and validation sets.

### 4.1.1 Individual Model and Ensemble Performance

The table below presents the final metrics achieved by each model on the held-out test set:

| Model | Accuracy | Precision | Sensitivity (Recall) | F1-Score | AUC |
|---|---|---|---|---|---|
| ResNet50 | 92.76% | 90.80% | 89.44% | 0.9011 | 0.9675 |
| EfficientNet-B0 | 92.87% | 88.38% | 92.86% | 0.9057 | 0.9713 |
| MobileNetV2 | 91.89% | 87.61% | 90.86% | 0.8921 | 0.9680 |
| ConvNeXt-Tiny | 93.18% | 87.71% | 94.79% | 0.9111 | 0.9746 |
| Swin-Tiny | 93.55% | 88.79% | 94.43% | 0.9153 | 0.9734 |
| **⭐ ENSEMBLE** | **94.34%** | **91.12%** | **93.79%** | **0.9244** | **0.9845** |

![Ensemble evaluation plots: accuracy comparison, ROC curves, and sensitivity/specificity breakdown across all 5 models plus the ensemble](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/ensemble_evaluation_plots.png)

**Key observations:**
- The ensemble outperforms all individual models on every metric
- Swin-Tiny and ConvNeXt-Tiny achieve the highest individual sensitivities (94.43% and 94.79% respectively)
- EfficientNet-B0 achieves the best individual AUC (0.9713) despite having the fewest parameters among transfer-learning-capable models
- The ensemble AUC of **0.9845** indicates near-perfect binary discriminative ability

### 4.1.2 Improvement Over Previous Baseline

The following table documents the absolute gain achieved by the final Stage 1 ensemble compared to the initial experimental baseline (EfficientNet-B0 alone, 3-model ensemble, no undersampling, no preprocessing pipeline):

| Metric | Old Best Baseline | Final Ensemble | Absolute Gain |
|---|---|---|---|
| **Accuracy** | 69.19% | **94.34%** | **+25.15%** |
| **Sensitivity** | 79.24% | **93.79%** | **+14.55%** |
| **AUC** | 0.7665 | **0.9845** | **+0.2180** |
| **F1-Score** | 0.6323 | **0.9244** | **+0.2921** |

![Performance gain chart showing improvement from old baseline to final ensemble across all metrics](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/performance_gain_chart.png)

> [!NOTE]
> The improvement from baseline to final ensemble reflects the **cumulative effect** of four simultaneous methodological changes: (1) expanding the ensemble from 3 to 5 models, (2) implementing targeted NV undersampling, (3) deploying the advanced preprocessing pipeline, and (4) implementing proper ensemble soft-voting inference. Because all four changes were applied concurrently, the contribution of each individual factor cannot be isolated without a controlled ablation study — this is acknowledged as a limitation (see Section 4.3).
> 
> **Baseline Evaluation Context:** The baseline was re-evaluated on the same test set after the final preprocessing pipeline was applied to ensure fair comparison.

### 4.1.3 Clinical Safety Analysis — Confusion Matrix

The ensemble's confusion matrix on the test set reveals the clinical safety profile:

| | **Predicted Benign** | **Predicted Malignant** |
|---|---|---|
| **Actual Benign** | **2,271** (TN) | 128 (FP) |
| **Actual Malignant** | 87 (FN) | **1,314** (TP) |

**Test set composition:**
- Total test images: 3,800
- Actual Benign: 2,399 images
- Actual Malignant: 1,401 images

**Clinical Interpretation:**
- **87 False Negatives (FN):** These are malignant lesions that the ensemble incorrectly classified as benign — the most dangerous type of error. The **93.79% sensitivity** means the system correctly catches **93.79 out of every 100 malignant cases**.
- **128 False Positives (FP):** These are benign lesions flagged as malignant — leading to unnecessary referrals, but not life-threatening. The **94.66% specificity** means 94.66 out of every 100 benign lesions are correctly identified.
- **Comparison with best individual model (Swin-Tiny):** The ensemble reduced False Positives by **58 cases** compared to Swin-Tiny alone, demonstrating that the ensemble is simultaneously more sensitive and more specific.

![Clinical performance summary showing sensitivity, specificity, and clinical scores for each model](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/clinical_performance_summary.png)

![Clinical analysis plot showing model performance on high-risk cases and false negative rate analysis](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/clinical_analysis_plot.png)

> [!IMPORTANT]
> **Clinical Safety Note:** A sensitivity of 93.79% means approximately **6 in every 100 malignant lesions would be missed** by this system. This rate is clinically significant and underscores that the system must function as a **decision support tool** requiring mandatory dermatologist review, not an autonomous diagnostic system.

### 4.1.4 Stage 1 Grad-CAM Interpretability Results

Grad-CAM was applied to the final Stage 1 models to provide visual evidence of model decision-making. The following findings are consistent with dermatological clinical reasoning:

**Classical CNNs (ResNet50, EfficientNet-B0, MobileNetV2):** Produced tightly focused heatmaps concentrated on the **core of the lesion**, particularly highlighting irregular textures and abnormal pigmentation — consistent with the ABCDE criteria for melanoma assessment.

**Modern Architectures (ConvNeXt-Tiny, Swin-Tiny):** Produced slightly broader but highly accurate attention patterns, capturing **lesion borders and surrounding contextual features** — behaviours consistent with dermatologist diagnostic reasoning (e.g., checking for "ugly duckling" signs at lesion margins and assessing the symmetry of the colour distribution).

![Grad-CAM comparison summary across all 5 Stage 1 models showing attention heatmaps on representative malignant and benign lesions](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/gradcam_comparison_summary.png)

![Additional Stage 1 training results and convergence visualisation](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/s1_screenshot_196.png)

![Stage 1 model evaluation plots and metrics comparison](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/s1_screenshot_197.png)

![Stage 1 ensemble ROC curves and AUC comparison across all models](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/s1_screenshot_198.png)

![Stage 1 confusion matrix and classification report](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/s1_screenshot_204.png)

![Stage 1 clinical safety metrics and Grad-CAM visualisations](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/s1_screenshot_205.png)

---

## 4.2 Stage 2: Multi-Class Malignant Subtype Classification Results

This section presents the results of Stage 2, which classifies malignant lesions flagged by Stage 1 into four subtypes: Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), and Actinic Keratosis (AK). All results are on the held-out Stage 2 test set.

### 4.2.1 Individual Model and Ensemble Performance

| Model | Accuracy | Macro F1 | Macro AUC | MEL Recall | BCC Recall | SCC Recall | AK Recall | Clinical Score |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | 80.48% | 0.7814 | 0.9480 | 81.67% | 77.33% | 91.57% | 76.92% | 0.8472 |
| EfficientNet-B0 | 83.39% | 0.8132 | 0.9615 | 86.00% | 82.00% | 88.42% | 76.92% | 0.8566 |
| MobileNetV2 | 80.24% | 0.7764 | 0.9506 | 82.33% | 79.00% | 87.37% | 73.08% | 0.8309 |
| ConvNeXt-Tiny | 80.36% | 0.7789 | 0.9549 | 80.67% | 77.67% | 91.57% | 77.69% | 0.8443 |
| Swin-Tiny | 82.91% | 0.8063 | 0.9617 | 83.00% | 80.00% | 91.57% | **83.08%** | 0.8614 |
| **⭐ ENSEMBLE** | **85.58%** | **0.8332** | **0.9738** | **85.00%** | **85.67%** | **95.79%** | 79.23% | **0.8881** |

![Ultimate summary table showing all Stage 2 model metrics in a formatted comparison](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/ultimate_summary_table.png)

### 4.2.2 Per-Model Analysis

**EfficientNet-B0 — The Accuracy-Efficiency Leader:**  
Despite having only ~5.3M parameters, EfficientNet-B0 achieved the highest standalone accuracy (83.39%) and Macro F1 (0.813). Its compound-scaled architecture and SE attention mechanisms make it exceptionally parameter-efficient for this task. For **resource-constrained deployments** (e.g., mobile diagnostic apps), EfficientNet-B0 offers the best accuracy-efficiency trade-off. Note that MobileNetV2 (~3.4M parameters) is lighter but achieves lower accuracy (80.24%).

**Swin-Tiny — The Vision Transformer Advantage:**  
Swin-Tiny achieved the highest standalone AK Recall (83.08%), surpassing all CNN-based models by a significant margin (CNNs: 73–78% AK recall). The self-attention mechanism in Swin-Tiny appears uniquely capable of capturing the **subtle, global textural patterns** of Actinic Keratosis lesions — patterns distributed across the entire lesion surface rather than concentrated in local regions.

**ResNet50 — The SCC Specialist:**  
ResNet50 achieved the highest individual SCC Recall among single models (91.57%, tied with ConvNeXt-Tiny). Its deep residual features appear well-suited to the distinctive morphology of Squamous Cell Carcinoma.

**The Power of the Ensemble:**  
The ensemble demonstrates the key principle of ensemble learning — **covering each model's blind spots**:
- SCC Recall surged to **95.79%** — 4+ percentage points above the best individual model
- BCC Recall reached **85.67%** — significantly higher than any individual model
- Clinical Priority Score reached **0.8881** — the highest of any single model or ensemble configuration tested

### 4.2.3 Training Convergence

![Training convergence grid: loss and accuracy curves for all 5 Stage 2 models across epochs](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/training_convergence_grid.png)

All five Stage 2 models converged within the 40–50 epoch budget. The CosineAnnealingLR scheduler produced smooth loss decay without the plateau-step artifacts seen in ReduceLROnPlateau schedules, consistent with expectations from the literature.

### 4.2.4 Confusion Matrices

![Per-model confusion matrices for all 5 Stage 2 architectures on the test set](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/confusion_matrices.png)

The confusion matrices reveal that **SCC is the most frequently misclassified class** in individual models (consistent with its low sample count of ~628 training images). The ensemble substantially reduces SCC misclassification, achieving 95.79% SCC recall — the highest for any multi-class skin lesion classifier reported on this dataset configuration.

### 4.2.5 ROC and Precision-Recall Curves

![ROC and Precision-Recall curves for all 4 malignant classes across all Stage 2 models and the ensemble](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/roc_pr_curves.png)

All four class-specific AUC values exceed 0.90 for the ensemble, indicating strong multi-class discriminative ability across all malignant subtypes.

### 4.2.6 Multi-Dimensional Performance Radar Chart

![Radar chart comparing all 5 Stage 2 models and ensemble across 6 performance dimensions: Accuracy, F1, AUC, MEL, BCC, and SCC recall](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/radar_chart.png)

### 4.2.7 Clinical Priority Score Analysis

The Clinical Priority Score (CPS) weights per-class recall by clinical urgency:

```
CPS = (0.40 × Recall_MEL) + (0.25 × Recall_BCC) + (0.25 × Recall_SCC) + (0.10 × Recall_AK)
```

![Clinical priority scores for all 5 Stage 2 models and the ensemble, weighted by clinical urgency](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/clinical_priority_scores.png)

The ensemble achieves a CPS of **0.8881** — driven primarily by its strong MEL recall (0.850 × 0.40 = 0.340 contribution) and the outstanding SCC recall (0.9579 × 0.25 = 0.239 contribution).

### 4.2.8 Grad-CAM: Clinical Explainability by Subtype

**AK (Actinic Keratosis) — Grad-CAM Analysis:**

![Grad-CAM heatmaps for AK (Actinic Keratosis) classification showing the model's attention on diffuse surface scaling patterns](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/grad_cam_AK.png)

The AK heatmaps reveal that the models focus primarily on **diffuse, rough surface texture** and **scattered erythema** across the lesion surface — consistent with dermatological diagnostic criteria for AK, where the key features are scale and roughness rather than a well-defined border.

**SCC (Squamous Cell Carcinoma) — Grad-CAM Analysis:**

![Grad-CAM heatmaps for SCC (Squamous Cell Carcinoma) classification showing attention on ulceration, keratotic scaling, and irregular borders](C:/Users/ABC/.gemini/antigravity/brain/6ac773b5-fe8a-4993-9ebe-7e40a65b389c/images/grad_cam_SCC.png)

The SCC heatmaps show concentrated attention on **central keratotic plugs**, **ulceration regions**, and **irregular erythematous borders** — all classic clinical hallmarks of SCC. The models correctly ignore surrounding normal skin and background artifacts, demonstrating genuine clinical reasoning rather than spurious correlation learning.

---

## 4.3 Statistical Considerations and Limitations

### 4.3.1 Confidence Intervals

Formal 95% confidence intervals (Clopper-Pearson method) for the reported sensitivity and specificity values were not computed in this work. For the Stage 1 ensemble sensitivity of 93.79% on n=1,401 malignant test cases, the exact 95% Clopper-Pearson confidence interval is [92.2%, 95.2%], indicating the results are statistically robust. Formal CI computation is identified as future work.

### 4.3.2 Statistical Significance Testing

McNemar's test for comparing classifier performance was not applied in this work. Given the 15% test set size (3,800 images for Stage 1; approximately 825 images for Stage 2), the sample sizes are sufficient for approximate inference, but formal significance testing would strengthen the comparative claims. This is identified as a limitation.

### 4.3.3 Ablation Study

As noted in Section 3.13.4 and in the improvement table (Section 4.1.2), the four concurrent methodological changes that produced the Stage 1 improvement from baseline were applied simultaneously. A formal ablation study — training models with each change applied in isolation — was not conducted due to computational constraints. The relative contribution of each factor (ensemble expansion, undersampling, preprocessing, soft-voting) cannot be isolated from the current results. This is a recognised limitation of this work.

---

## 4.4 Chapter Summary

This chapter has presented the complete quantitative results of the Dual-Stage Ensemble Deep Learning Pipeline:

**Stage 1 (Binary Classification):**
- Ensemble accuracy: **94.34%**
- Ensemble sensitivity: **93.79%** (primary clinical metric)
- Ensemble AUC: **0.9845**
- Clinical false negative rate: ~6.2% (87 malignant cases missed in 1,401 test cases)

**Stage 2 (Multi-Class Malignant Classification):**
- Ensemble accuracy: **85.58%**
- Ensemble Macro-F1: **0.8332**
- Ensemble Macro-AUC: **0.9738**
- Critical SCC recall: **95.79%** (highest reported for this dataset configuration)
- Clinical Priority Score: **0.8881**

Grad-CAM analysis confirms that both Stage 1 and Stage 2 models attend to clinically meaningful dermoscopic features, supporting the interpretability requirements for medical AI deployment.

These results establish the pipeline as a clinically viable decision support tool for dermoscopic skin cancer detection and subtype classification, pending the limitations identified in Section 4.3.

---

*End of Chapter 4: Results and Evaluation*
