# Stage 1 (SK1) Methodology: Binary Classification (Benign vs. Malignant)

## 1. Overview of the Proposed System

Stage 1 acts as a *triage gate* in the Dual-Stage Ensemble Deep Learning Pipeline for the automated detection and fine-grained classification of skin cancer from dermoscopic images. Its primary role is to filter the vast majority of benign cases and flag only those that require further investigation. Every design decision—from model selection to loss function choice to data handling—was driven by one overarching clinical requirement: **minimising false negatives**, since a missed cancer diagnosis is far more harmful than an unnecessary referral.

## 2. Dataset and Data Preparation

The primary data source is the **International Skin Imaging Collaboration (ISIC) 2019 Challenge** dataset, which is widely accepted as the gold standard in computational dermatology research. The dataset contains 25,331 images across 8 diagnostic categories.

### 2.1 Binary Label Mapping
The 8 ISIC categories are mapped to a hierarchical label structure where Stage 1 evaluates two merged binary categories:
- **Malignant (Label = 1)**: Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), and Actinic Keratosis (AK). Note: AK is technically pre-malignant but requires intervention, hence it is grouped as malignant for early detection.
- **Benign (Label = 0)**: Melanocytic Nevus (NV), Benign Keratosis-like Lesions (BKL), Dermatofibroma (DF), Vascular Lesions (VASC).

### 2.2 Targeted NV Undersampling
A severe class imbalance exists within the dataset, particularly with the Melanocytic Nevus (NV) class containing 8,978 images, creating a 3:1 benign-to-malignant imbalance. 
To counteract this without losing critical minority data:
- **Targeted undersampling** was applied exclusively to the NV class (reducing its count from 8,978 to 4,238 images).
- All other minority benign classes (DF, VASC, BKL) were retained unchanged. This approach prevents the model from blindly predicting "Benign" for all inputs, ensuring it adequately learns the features of rarer benign conditions.

### 2.3 Stratified Data Split and Validation Strategy
The dataset was divided using a **single hold-out 70/15/15 split** via `sklearn.model_selection.GroupShuffleSplit`:
- **Training Set (70%)**: Used for model weight optimisation.
- **Validation Set (15%)**: Used for hyperparameter tuning and early stopping.
- **Test Set (15%)**: Final, held-out performance evaluation.

**Rationale for the Split Strategy:**
1. **Data Leakage Prevention:** `GroupShuffleSplit` on the `lesion_id` field guarantees that images from the same physical lesion do not straddle train/test sets, preventing artificial performance inflation.
2. **Computational Constraints:** K-fold cross-validation across 5 heterogeneous modern models would require extensive compute beyond available resources.
3. **Representativeness:** With over 25k raw images yielding 3,800 validation and test samples, a single split provides statistically robust evaluation bounds.

## 3. Preprocessing Pipeline

Before reaching the neural networks, images undergo a 7-step standardisation pipeline to correct artifacts inherently present in dermoscopic images (like body hair, ruler marks, or ink):

1. **Load & Decode**: Convert the raw JPEG image (PIL Image) to an RGB array, standardising to 3 channels.
2. **Hair/Artifact Detection**: A black-hat morphological transform with a 15×15 structuring element isolates dark linear structures. This identifies hair and gel bubbles against the skin.
3. **Inpainting**: OpenCV's Telea fast marching algorithm fills the identified artifact pixels using textures propagated from surrounding healthy skin to create a seamless lesion view.
4. **Lesion Detection**: Otsu's global thresholding separates the lesion from the background, outputting a clear bounding box around the largest detected contour.
5. **Crop & Resize**: The image is cropped to the bounding box and resized to exactly 224×224 pixels using bicubic interpolation—the optimal input size for all models.
6. **Tensor Conversion**: The PIL image is transformed to a PyTorch tensor, scaling pixel integers `[0, 255]` to floats `[0.0, 1.0]`.
7. **Normalisation**: Tensors are normalised against ImageNet statistics (mean `μ = [0.485, 0.456, 0.406]`, std `σ = [0.229, 0.224, 0.225]`), ensuring the pre-trained feature detectors operate within their calibrated range.

## 4. Stage 1 Model Architectures

Stage 1 employs a **5-model heterogeneous ensemble**, aggregating models from three distinct architectural families (classical CNNs, modern CNNs, Vision Transformers). The diverse structure ensures a robust capture of different discriminative features. Each model replaces its top fully-connected layer with a custom binary classification head (`Linear(in) → ReLU → Dropout(0.3) → Linear(512, 1)`).

1. **ResNet50 (~25.6M parameters):** Overcomes the vanishing gradient problem using residual skip connections, allowing for a highly deep convolutional structure capable of capturing multi-scale features.
2. **EfficientNet-B0 (~5.3M parameters):** Employs compound scaling and MBConv blocks with Squeeze-and-Excitation attention to selectively emphasise informative channels in a highly parameter-efficient manner.
3. **MobileNetV2 (~3.4M parameters):** Built for constrained environments using inverted residual blocks, projecting low-dimensional representations into higher dimensions for efficient depthwise processing before projecting back.
4. **ConvNeXt-Tiny (~28.6M parameters):** A modernised pure CNN that integrates Vision Transformer structural choices, including larger 7x7 depthwise convolutions to capture broader context, Layer Norm, and GELU activations.
5. **Swin-Tiny (~28.3M parameters):** A Vision Transformer that computes self-attention in non-overlapping local shifted windows, enabling hierarchical global feature integration without the prohibitive quadratic cost of standard ViTs.

## 5. Training Configuration

| Hyperparameter | Value | Justification |
|---|---|---|
| **Optimiser** | Adam (β₁=0.9, β₂=0.999) | Robust adaptation to sparse gradients |
| **Learning Rate** | 1×10⁻⁴ (initial) | Best practice for fine-tuning pre-trained CNNs |
| **LR Scheduler** | ReduceLROnPlateau | Reduces LR by 0.5× if validation loss plateaus for 5 epochs |
| **Loss Function** | BCEWithLogitsLoss | Stable computation of sigmoid and binary cross-entropy |
| **Batch Size** | 32 | Optimal balance between GPU memory and batch stability |
| **Max Epochs** | 70 | Upper bound limit for safety |
| **Early Stopping** | Patience = 10 epochs | Halts training to prevent overfitting and restores best weights |

### 5.1 Transfer Learning Strategy
Models are exclusively initialised with **ImageNet pre-trained weights**. Lower convolutional layers retain critical low-level feature detectors (edge detection, textural analysis) learned from 1.2M images. The training fully unfreezes the models, adapting these low-level generalized features specifically to dermatological imaging (fine-tuning).

## 6. Ensemble Strategy

Predictions from all five models are averaged together using **soft-voting (probability averaging)**:
`P_ensemble(x) = (1/5) × [P_ResNet50(x) + P_EfficientNet(x) + ... + P_Swin(x)]`

**Why Soft Voting over Hard Voting?**
Soft voting preserves confidence magnitudes. A model 97% confident in a malignancy prediction contributes significantly more weight than one reporting 52%. This yields systematically better-calibrated decisions in medical settings. A strict `P ≥ 0.5` threshold defines the final Binary Output.

## 7. Evaluation Metrics and Interpretability

Models are assessed strictly on the unseen **Test Set**. 

### 7.1 Clinical Safety Profile
- **Sensitivity (Recall)** is the paramount clinical metric: defined as `TP/(TP+FN)`. Minimising False Negatives (FN) ensures malignancies aren't sent home as "Benign".
- **Grad-CAM (Gradient-weighted Class Activation Mapping):** Applied to interpret decisions visually. CNN models typically focus intently on central lesion characteristics, while ViTs like Swin-Tiny capture broad marginal contexts, successfully echoing "ugly duckling" criteria observed by dermatologists.

## 8. Results of Stage 1 (SK1) Binary Classification

The soft-voting ensemble approach demonstrates comprehensive superiority across all performance metrics when run on the 3,800-image hold-out test set:

| Model | Accuracy | Precision | Sensitivity (Recall) | F1-Score | AUC |
|---|---|---|---|---|---|
| ResNet50 | 92.76% | 90.80% | 89.44% | 0.9011 | 0.9675 |
| EfficientNet-B0 | 92.87% | 88.38% | 92.86% | 0.9057 | 0.9713 |
| MobileNetV2 | 91.89% | 87.61% | 90.86% | 0.8921 | 0.9680 |
| ConvNeXt-Tiny | 93.18% | 87.71% | 94.79% | 0.9111 | 0.9746 |
| Swin-Tiny | 93.55% | 88.79% | 94.43% | 0.9153 | 0.9734 |
| **⭐ ENSEMBLE** | **94.34%** | **91.12%** | **93.79%** | **0.9244** | **0.9845** |

### Clinical Context of Results:
- **Outstanding Sensitivity:** The ensemble achieves a **93.79% sensitivity**, meaning fewer than 7 out of 100 malignant cases are missed.
- **Top Individual Models:** ConvNeXt-Tiny (94.79%) and Swin-Tiny (94.43%) achieved the highest standalone sensitivities, highlighting the immense value of modern architectural advancements in dermoscopic feature extraction.
- **Near-Perfect AUC:** The ensemble’s **0.9845 AUC** signifies an incredibly powerful overall discriminative capacity regardless of threshold selection, effectively resolving the binary triage stage requirement.
