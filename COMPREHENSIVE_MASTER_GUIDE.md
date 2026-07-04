# Comprehensive Master Guide: Automated Skin Cancer Classification
**Dual Ensemble Deep Learning Pipeline using ISIC 2019**

This document contains the entire scope of the project from scratch. It is designed to act as the single source of truth when setting up the project on a new machine.

---

## 1. The Dataset & Preprocessing

**Dataset Used**: ISIC 2019 Challenge (International Skin Imaging Collaboration).
- **Scale**: Contains 25,331 training images across 8 diagnostic categories.
- **Goal**: Detect melanoma and other malignant skin tumors.

**Preprocessing Steps**:
Before feeding any image into our neural networks, it undergoes the following standard medical imaging transformations:
1. **Resizing**: All images are standardized to exactly `224x224` pixels.
2. **Tensor Conversion**: The image pixels are converted into PyTorch Tensors (mathematical matrices).
3. **Normalization**: The pixel values are normalized using standard ImageNet mean and standard deviation (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).

---

## 2. The 5-Model Ensemble Architecture

Across all stages of this project, we employ an ensemble (a team) of 5 distinct state-of-the-art Deep Learning architectures. Averaging the predictions of multiple models drastically reduces false positives in medical diagnosis.

1. **ResNet50**: A deep Convolutional Neural Network (CNN) with 50 layers. It uses "skip connections" to solve the vanishing gradient problem in deep networks.
2. **EfficientNet-B0**: An architecture that mathematically balances network depth, width, and resolution to achieve high accuracy with minimal computational cost.
3. **MobileNetV2**: A highly lightweight CNN built using inverted residual blocks, ideal for fast, real-time predictions.
4. **ConvNeXt-Tiny**: A modern pure CNN that was rebuilt from the ground up to mimic the architectural strengths of Vision Transformers, providing incredible accuracy.
5. **Vision Transformer (ViT-B/16)**: A completely different paradigm from CNNs. It chops the image up into 16x16 patches and treats them like a sequence of words (using self-attention), identifying global patterns that CNNs often miss.

---

## 3. Project Pipeline Stages

Our pipeline is divided into distinct, logical stages to maximize classification accuracy for rare cancer types.

### Stage 1: Binary Classification (Benign vs. Malignant)
* **Objective**: Act as a highly sensitive filter. Is this lesion cancerous (Malignant) or harmless (Benign)?
* **Weight Initialization (Pre-trained)**: The 5 models are loaded with `pretrained=True`. This means they start with weights learned from millions of Google ImageNet photos. Because they already know how to extract features (edges, textures, colors), they learn medical features very rapidly.
* **Epochs**: Trained for **20-30 epochs**.

### Stage 2a: Multi-Class Classification & Imbalance Handling
* **Objective**: For images flagged as "Malignant", classify exactly what subtype of cancer it is (e.g., Melanoma, BCC, SCC).
* **The Imbalance Problem**: Some cancers are extremely rare in the dataset compared to Melanoma. If ignored, the model simply guesses "Melanoma" every time and ignores the rare cancers.
* **The Solution (No GANs)**: Instead of generating fake, blurry images using a GAN, we use robust mathematical techniques:
  1. **Focal Loss**: A loss function that applies a massive mathematical penalty if the model misclassifies a rare class, forcing it to pay attention.
  2. **MixUp Augmentation**: A technique that dynamically blends two images and their labels together during training. It acts as a powerful regularizer preventing the model from memorizing the few examples of rare cancers.
* **Weight Initialization (From Scratch)**: The 5 models are loaded with `weights=None`. They start completely blind. We force them to learn *only* the specific multi-class cancer features without any bias from ImageNet.
* **Epochs**: Trained for **40-50 epochs** (because they are learning from scratch).

### Stage 2b: Interpretability via Grad-CAM
* **Objective**: "Black Box" medical AI is dangerous. We must prove *why* the model made its decision.
* **Execution**: We apply Gradient-weighted Class Activation Mapping (Grad-CAM). 
* **How it works**: By tracking the gradients flowing back into the final convolutional layer, we generate a visual heatmap on top of the original skin image. Red/hot areas show exactly which pixels the model looked at (e.g., irregular borders, abnormal colors) to make its diagnosis. This proves the model is genuinely diagnosing cancer and not just looking at a ruler or background skin.

---

## 4. End-to-End Workflow summary

If you need to re-run the entire project on a new laptop:
1. Run `kaggle_pipeline/01_Stage1_Binary_Classification_Ensemble.ipynb` to train the Stage 1 binary filter.
2. Run `kaggle_pipeline/02_Stage2_MultiClass_Imbalance_Handled.ipynb` to train the Stage 2 multi-class models (which automatically handles the imbalance internally via Focal Loss).
3. Use the trained `.pth` weights to generate your Stage 2b Grad-CAM heatmaps for the final report.
