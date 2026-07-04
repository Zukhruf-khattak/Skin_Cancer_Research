# Automated Skin Cancer Detection: Dual Ensemble Pipeline Guide

This document serves as the comprehensive guide for your final year project's machine learning pipeline. It explains the architectures used, the advanced techniques implemented to handle data imbalance, deep learning concepts relevant to your defense, and the exact steps to run the code on Kaggle.

---

## 1. Pipeline Architecture Overview

Based on the ISIC 2019 dataset, the project utilizes a **Dual Ensemble Pipeline**. 

### Stage 1: Binary Classification (Benign vs. Malignant)
The goal of Stage 1 is to filter out Benign images so that Stage 2 only has to worry about identifying the specific type of cancer. 
* **The 5-Model Ensemble**: ResNet50, EfficientNet-B0, MobileNetV2, ConvNeXt, and Vision Transformer (ViT).
* **Epochs**: 20 to 30 epochs.

### Stage 2: Multi-Class Classification (Malignant Subtypes)
The goal of Stage 2 is to classify the malignant tumors into specific subtypes (e.g., Melanoma, BCC, SCC).
* **The 5-Model Ensemble**: The exact same 5 architectures are used, but they are trained for multi-class classification.
* **Epochs**: 40 to 50 epochs.

---

## 2. Key Deep Learning Concepts Explained

### Transfer Learning vs. Training From Scratch
In deep learning, initializing a model's weights is crucial. 
* **Pre-trained (Transfer Learning)**: In Stage 1, we set `pretrained=True`. This means the models download weights from being trained on millions of everyday images (ImageNet). Even though ConvNeXt and ViT have never seen your skin cancer dataset before, they already know how to "see" edges, shapes, and textures. We then train them on the ISIC dataset. This gives them a massive head-start, allowing them to master binary classification in just 20 epochs.
* **Training from Scratch**: In Stage 2, we set `weights=None` (or `pretrained=False`). The models start completely "blind" with random mathematical weights. Your project requires Stage 2 to learn strictly from the multi-class cancer features without prior bias, which is why they train from scratch and require more epochs (50 epochs).

### Handling Data Imbalance (Replacing the GAN)
Originally, a Conditional GAN was considered to generate fake images for rare cancers (like SCC or BCC). However, GANs often generate artifacts or blurry textures that harm medical diagnosis models. Instead, we use mathematically robust state-of-the-art techniques:
1. **Focal Loss**: A specialized loss function that mathematically forces the neural network to pay more attention to the minority classes. If the model gets a common class wrong, the penalty is small. If it gets a rare class wrong, the penalty is massive.
2. **MixUp Augmentation**: During training, the code takes two random images and mathematically blends their pixels and labels together. This prevents the model from just memorizing the few examples of the rare classes (overfitting) and forces it to learn underlying patterns.

---

## 3. Step-by-Step Kaggle Execution Guide

Below are the instructions to run the two generated notebooks on Kaggle.

### Step 3.1: Setup Your Kaggle Environment
1. Go to kaggle.com and click **Create -> New Notebook**.
2. Turn on the GPU: Go to the right-side panel -> **Session Options** -> **Accelerator** -> Select **GPU T4 x2** (or P100/T4).
3. Add the Dataset: Click **Add Data** on the right panel, search for "ISIC 2019", and add the dataset. This will make it available at `../input/isic-2019/`.

### Step 3.2: Run Stage 1 (Binary Classification)
1. Upload `01_Stage1_Binary_Classification_Ensemble.ipynb` to your Kaggle Notebook via **File -> Import Notebook**.
2. Update the data paths in the notebook to point to your ISIC 2019 dataset directory.
3. Click **Run All** (or **Run -> Run All**).
4. The notebook will load the 5 models with pretrained weights, adapt them for binary classification, and execute the training loop for 20 epochs.

### Step 3.3: Run Stage 2 (Multi-Class + Imbalance Handling)
1. Create a **NEW** Kaggle Notebook. Turn on the GPU and add the ISIC 2019 dataset just like in Step 3.1.
2. Upload `02_Stage2_MultiClass_Imbalance_Handled.ipynb` via **File -> Import Notebook**.
3. Update the data paths to point to the ISIC 2019 dataset.
4. Click **Run All**.
5. The notebook will initialize the models from scratch and automatically apply **Focal Loss** and **MixUp** while training for 50 epochs.

### Step 3.4: Saving Your Results
When training is finished, click the **Save Version** button in the top right corner. Select **Save & Run All (Commit)**. This runs the notebook in the background. Once it finishes, go to the notebook's viewer page and download the trained `.pth` weights from the "Output" section for your final project submission.
