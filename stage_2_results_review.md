# Stage 2 Results Review: Multi-Class Classification

## 1. Methodology & Pipeline Changes (Stage 1 vs. Stage 2)
To achieve the performance leap seen in Stage 2, several critical architectural and methodology changes were introduced. Even the slightest changes are documented below:

*   **Task Scope Expanded**: Transitioned from a simple Binary Classification (Benign vs. Malignant) to a complex Multi-Class Classification (specifically focusing on MEL, BCC, SCC, and AK).
*   **Model Roster Expanded**: Stage 1 relied on three CNNs. Stage 2 introduced **ConvNeXt-Tiny** (a modern CNN architecture) and **Swin-Tiny** (a Vision Transformer) to the ensemble, bringing the total to 5 distinct architectures.
*   **Weight Initialization (The Blank Slate)**: Stage 1 used ImageNet pre-trained weights (`pretrained=True`). In Stage 2, models were loaded with **no pre-trained weights (`weights=None`)**. They were trained entirely from scratch to ensure they learned highly specific dermatological features without any residual bias from standard ImageNet photos.
*   **Training Duration Increased**: Because the models were trained from scratch, the training cycle was extended from 20-30 epochs (in Stage 1) to **40-50 epochs**.
*   **Loss Function Overhaul**: Transitioned from standard Binary Cross-Entropy (`BCEWithLogitsLoss`) to **Focal Loss**. This change mathematically forces the network to pay massive attention to rare, hard-to-classify cancers rather than just guessing the majority class.
*   **Advanced Augmentation (MixUp)**: Introduced **MixUp Augmentation**, which dynamically blends two different skin lesion images and their labels together during training. This acts as a severe regularizer to prevent the models from memorizing the limited rare cancer samples.
*   **Interpretability Added**: Introduced a **Grad-CAM** pipeline at the end of Stage 2 to generate heatmaps, proving the models were making clinical decisions based on actual lesion morphology rather than background artifacts.

---

## 2. Overall Performance Leap
The transition from Stage 1 to Stage 2 marks a massive improvement in the pipeline's performance. 
In Stage 1, the best model (EfficientNet-B0) peaked at **69.19% accuracy** and struggled to hit sensitivity targets. In Stage 2, the models achieved up to **85.57% accuracy** (Ensemble), showcasing a highly robust multi-class capability despite the increased difficulty of the task.

## 3. Head-to-Head Architectural Comparison

| Model | Accuracy | Macro F1 | Macro AUC | MEL Recall | BCC Recall | SCC Recall | AK Recall | Clinical Score |
|-------|----------|----------|-----------|------------|------------|------------|-----------|----------------|
| ResNet50 | 80.48% | 0.7814 | 0.9480 | 81.66% | 77.33% | 91.57% | 76.92% | 0.8472 |
| EfficientNet-B0 | 83.39% | 0.8132 | 0.9615 | 86.00% | 82.00% | 88.42% | 76.92% | 0.8566 |
| MobileNetV2 | 80.24% | 0.7764 | 0.9506 | 82.33% | 79.00% | 87.36% | 73.07% | 0.8308 |
| ConvNeXt-Tiny | 80.36% | 0.7788 | 0.9548 | 80.66% | 77.66% | 91.57% | 77.69% | 0.8443 |
| Swin-Tiny | 82.90% | 0.8063 | 0.9617 | 83.00% | 80.00% | 91.57% | **83.07%**| 0.8613 |
| ⭐ **ENSEMBLE** | **85.57%** | **0.8331** | **0.9737** | **85.00%** | **85.66%** | **95.78%** | 79.23% | **0.8880** |

### 3.1 The Lightweight Champion: EfficientNet-B0
*   **Strengths:** Achieved the highest standalone Accuracy (83.39%) and Macro F1 (0.813). It also had the best standalone Melanoma (MEL) Recall at 86.00%.
*   **Analysis:** Despite its low parameter count, EfficientNet-B0 is highly parameter-efficient and serves as the best standalone CNN for this task.

### 3.2 The Vision Transformer Advantage: Swin-Tiny
*   **Strengths:** Completely dominated in Actinic Keratosis (AK) Recall (83.07%), whereas all CNNs struggled heavily with this class (73-77%).
*   **Analysis:** The self-attention mechanisms in Swin-Tiny likely allowed it to capture subtle global textural patterns of AK lesions that convolutional models missed.

### 3.3 The Classic CNNs: ResNet50 vs. MobileNetV2
*   **Strengths:** ResNet50 maintained strong SCC Recall (91.57%). MobileNetV2 was the weakest overall (80.24% accuracy).
*   **Analysis:** While ResNet50 provides a solid baseline, modern architectures like EfficientNet and Swin significantly outclass it.

### 3.4 The Modern CNN: ConvNeXt-Tiny
*   **Strengths:** Solid SCC recall, but overall accuracy (80.36%) was unexpectedly similar to ResNet50.
*   **Analysis:** ConvNeXt did not demonstrate a clear edge over EfficientNet or Swin-Tiny on this specific dataset.

## 4. The Power of the Ensemble
The Ensemble model is the definitive winner of Stage 2. It effectively covers the blind spots of the individual models:
*   **SCC Detection:** Pushed Squamous Cell Carcinoma recall to an incredible **95.78%**.
*   **BCC Detection:** Pushed Basal Cell Carcinoma recall to **85.66%**.
*   **Clinical Priority Score:** Achieved the highest clinical safety score of **0.888**.

## 5. Conclusion
Stage 2 successfully transformed a struggling binary classifier into a clinically viable multi-class ensemble. The generated visualizations (Grad-CAM, ROC curves, confusion matrices) provide robust empirical evidence. The Ensemble is recommended for deployment where compute permits, while EfficientNet-B0 is the optimal choice for resource-constrained environments.
