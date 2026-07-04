# Final Stage 2 Methodology: Multi-Class Malignant Subtype Classification

## 1. Overview of the Proposed System

Stage 2 represents the **fine-grained diagnostic layer** of the pipeline. It only processes images that Stage 1 has flagged as malignant. Its objective is to determine the specific type of cancer from four critical subtypes: Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), and Actinic Keratosis (AK). This is clinically vital since treatment protocols drastically differ between these subtypes (e.g., immediate surgical excision for Melanoma versus topical treatments for Actinic Keratosis). 

Stage 2 faces a fundamentally different and more complex architectural challenge than Stage 1:
- The dataset is drastically reduced to the malignant-only subset.
- Imbalance is highly skewed (e.g., MEL is dominant, SCC is very rare).
- Visual differences between subtypes are extremely subtle.

## 2. Dataset Filtering and Preparation

The Stage 2 training dataset is derived by strictly isolating the four malignant classes from the ISIC 2019 dataset:
- **MEL (Melanoma)**: ~4,522 images
- **BCC (Basal Cell Carcinoma)**: ~3,323 images
- **AK (Actinic Keratosis)**: ~867 images
- **SCC (Squamous Cell Carcinoma)**: ~628 images

## 3. Class Imbalance and Data Augmentation Strategies

Unlike Stage 1, which used undersampling, Stage 2 employs two advanced in-training techniques to handle severe imbalance without discarding any precious malignant data.

### 3.1 Focal Loss
Standard Cross-Entropy loss would allow the dominant MEL and BCC classes to easily overwhelm the rare SCC and AK classes. **Focal Loss** mitigates this by applying a modulating factor that dynamically down-weights the loss contribution of easy, well-classified examples, aggressively shifting the model's focus (learning budget) towards hard, misclassified examples (typically the rare classes). 
- **Parameters:** `γ=2` (focusing parameter), `α` (class frequency weights).

### 3.2 MixUp Data Augmentation
To combat the small sample sizes of SCC and AK, **MixUp** creates virtual synthetic training samples by linearly interpolating between random pairs of images and their corresponding labels (`λ ~ Beta(0.2, 0.2)`). This forces the model to behave linearly between data points and creates infinite variations, heavily reducing the risk of overfitting.

### 3.3 Heavy Online Augmentation
A comprehensive pipeline of transformations is applied dynamically during training to maximize generalization:
- Random Horizontal/Vertical Flips (Orientation invariance)
- Random Rotation ±30° 
- Random Resized Crops (Scale invariance)
- Colour Jitter (Brightness/Contrast/Saturation variation)
- Random Grayscale (Forces reliance on textural morphology rather than colour alone)
- Random Erasing (Occlusion robustness)

## 4. Model Architectures and "From Scratch" Training Philosophy

Stage 2 utilizes the identical 5-model heterogeneous ensemble structural framework as Stage 1 (ResNet50, EfficientNet-B0, MobileNetV2, ConvNeXt-Tiny, Swin-Tiny). However, the training paradigm is completely altered:

### 4.1 "Blank Slate" Initialisation
**All five models are trained with randomly initialised weights (`weights=None`), completely rejecting ImageNet pre-training.** 
- **Rationale:** The dermatological features distinguishing malignant subtypes (e.g., asymmetric pigmentation networks vs. arborising telangiectasia) have zero correlation with natural images from ImageNet (dogs, cars). Using ImageNet weights would introduce harmful inductive biases. Starting from scratch forces the networks to learn features entirely grounded in dermatological morphology.

## 5. Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Weight Init** | Random (From Scratch) | Avoid ImageNet domain bias |
| **Optimiser** | Adam | Unchanged from Stage 1 |
| **Learning Rate** | 1×10⁻³ | Increased (10x higher than Stage 1) to enable learning from a blank slate |
| **LR Scheduler** | CosineAnnealingLR (T_max=50) | Smoother decay path suitable for from-scratch training |
| **Loss Function** | Focal Loss (γ=2) | Addresses extreme rare-class imbalance dynamically |
| **Batch Size** | 32 | Memory stability |
| **Max Epochs** | 40–50 | Faster convergence under Focal Loss |

## 6. Ensemble Strategy and Clinical Priority Score (CPS)

Stage 2 outputs are combined via **soft-voting (probability averaging)** across the 5 models to determine the final 4-class prediction.

### 6.1 Clinical Priority Score
A composite metric, the Clinical Priority Score (CPS), was developed to assign heavier weights to the recall rates of the most dangerous and rapidly progressing malignancies:
`CPS = (0.40 × Recall_MEL) + (0.25 × Recall_BCC) + (0.25 × Recall_SCC) + (0.10 × Recall_AK)`
Melanoma receives the highest weight due to its severe mortality rate if misdiagnosed.

## 7. Results of Final Stage 2 Multi-Class Classification

The performance of individual models and the soft-voting ensemble is evaluated on the held-out Stage 2 test set. **Macro-F1** is the primary standalone metric as it treats all four classes equally, regardless of dataset size.

| Model | Accuracy | Macro F1 | Macro AUC | MEL Recall | BCC Recall | SCC Recall | AK Recall | Clinical Score |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | 80.48% | 0.7814 | 0.9480 | 81.67% | 77.33% | 91.57% | 76.92% | 0.8472 |
| EfficientNet-B0 | 83.39% | 0.8132 | 0.9615 | 86.00% | 82.00% | 88.42% | 76.92% | 0.8566 |
| MobileNetV2 | 80.24% | 0.7764 | 0.9506 | 82.33% | 79.00% | 87.37% | 73.08% | 0.8309 |
| ConvNeXt-Tiny | 80.36% | 0.7789 | 0.9549 | 80.67% | 77.67% | 91.57% | 77.69% | 0.8443 |
| Swin-Tiny | 82.91% | 0.8063 | 0.9617 | 83.00% | 80.00% | 91.57% | **83.08%** | 0.8614 |
| **⭐ ENSEMBLE** | **85.58%** | **0.8332** | **0.9738** | **85.00%** | **85.67%** | **95.79%** | 79.23% | **0.8881** |

### Key Observations from Results:
- **The Ensemble Advantage:** The ensemble leverages the architectural blind spots of each network. While Swin-Tiny uniquely excels at Actinic Keratosis (83.08%) through its broad textural attention mechanism, ResNet50 excels at Squamous Cell Carcinomas. By fusing these models, the Ensemble peaks at **95.79% SCC Recall** and **85.67% BCC Recall**, far exceeding the capabilities of any single architecture.
- **Top Efficiency Trade-off:** Despite having only ~5.3M parameters, EfficientNet-B0 achieved an impressive standalone Macro F1 (0.8132), making it the prime candidate for localized deployment strategies on mobile.
- **High Clinical Safety:** A peak Clinical Priority Score of **0.8881** means the system is highly protective regarding the most severe disease subtypes (MEL and SCC).
