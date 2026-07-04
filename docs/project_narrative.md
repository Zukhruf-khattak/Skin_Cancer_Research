# Project Narrative & Key Findings

> **Note for Thesis Writing:** Use this document to guide the narrative of your thesis. It explains the "story" of the research, why certain decisions were made, and how the results compare to the initial proposal.

---

## 1. Proposal vs. Reality

Your final project went above and beyond the initial proposal, resulting in a much more comprehensive study.

| Stage | What the Proposal Said | What Was Actually Done | Status |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Binary)** | ResNet50, EfficientNet-B0, MobileNetV2 | ResNet50, EfficientNet-B0, MobileNetV2 | ✅ Exactly matched proposal. |
| **Stage 2 (Multi-class)**| DenseNet121, Inception-v3, VGG16 | **10 modern models tested** (ResNet18/34/50, EfficientNet-B0/B1/B2, MobileNetV2/V3, DenseNet121, ShuffleNet) | ✅ Exceeded proposal. Better architecture selection. |

*Note for Thesis: You will need to add a brief sentence in the Methodology chapter justifying the shift from VGG/Inception to testing a wider array of modern, lightweight architectures.*

---

## 2. The Research Story (Step-by-Step)

Your project follows a clear scientific narrative of identifying a problem and systematically solving it.

### Step 1: Stage 1 (Binary Classification)
*   **Action:** You built a robust preprocessing pipeline (hair removal, cropping) and trained 3 models to classify "Benign vs Malignant" on the full dataset.
*   **Result:** EfficientNet-B0 achieved the highest overall accuracy (~69%). MobileNetV2 achieved the highest sensitivity (79.24%).
*   **The Problem:** The models struggled to hit the 90% sensitivity target mentioned in the proposal, likely due to the inherent difficulty of distinguishing early-stage malignancies from benign nevi without clinical context.

### Step 2: Stage 2a (The "Baseline" Attempt)
*   **Action:** You attempted to train 10 models to classify the exact malignant subtypes (Melanoma vs. BCC vs. SCC vs. Mel. Metastasis) using the raw, imbalanced data.
*   **The Problem:** The dataset was heavily skewed (Melanoma was 43% of the data, while SCC was only 5%). 
*   **Result:** The models suffered from majority-class collapse. Lightweight models like MobileNet-V3 effectively learned to just guess "Melanoma" for everything, resulting in a poor 43% accuracy.

### Step 3: Stage 2b (The "Solution")
*   **Action:** Recognizing the failure in Stage 2a, you hypothesized that data imbalance was the root cause. You applied **Upsampling** to balance the training data, giving the models equal exposure to rare cancers like SCC.
*   **Result:** A massive success. Accuracy across the board skyrocketed. MobileNet-V3 went from being the worst model to the best, achieving over 90% accuracy.

### Step 4: Advanced Experiments (GANs and SMOTE)
*   **Action:** You explored state-of-the-art techniques for synthetic data generation to see if they could perform better than simple upsampling. You trained Conditional GANs and applied SMOTE.
*   **Result:** While SMOTE did not outperform simple upsampling, the GAN experiment successfully generated realistic synthetic data (validated via t-SNE plots). This demonstrates advanced technical capability and thorough experimentation.

---

## 3. The Critical Proof: Why Stage 2b Was Necessary

This table is the core of your thesis argument. It proves that the data imbalance was the primary bottleneck in your system.

| Model | Raw Trained Accuracy (Stage 2a) | After Upsampling Accuracy (Stage 2b) | Massive Improvement |
| :--- | :---: | :---: | :--- |
| **MobileNet-V3** | 43.13% | **90.34%** | 📈 **+47.21%** |
| **EfficientNet-B1** | 58.58% | **89.91%** | 📈 **+31.33%** |
| **EfficientNet-B0** | 58.58% | **88.63%** | 📈 **+30.05%** |
| **ResNet18** | 61.80% | **87.77%** | 📈 **+25.97%** |
| **EfficientNet-B2** | 55.15% | **87.55%** | 📈 **+32.40%** |
| **ShuffleNet** | 63.73% | **87.55%** | 📈 **+23.82%** |
| **DenseNet121** | 70.17% | **86.91%** | 📈 **+16.74%** |
| **MobileNet-V2** | 50.43% | **85.84%** | 📈 **+35.41%** |
| **ResNet50** | 58.15% | **84.33%** | 📈 **+26.18%** |
| **ResNet34** | **70.39%** | **83.91%** | 📈 **+13.52%** |

---

## 4. Why Include Stage 2a in the Thesis?

You must include Stage 2a in your thesis as your **Baseline Experiment**. 

If you only presented Stage 2b, the examiners would ask: *"How do you know the upsampling actually helped? Maybe the models were just good anyway."*

By presenting Stage 2a first, you build a powerful scientific argument:
1.  **Baseline:** "We tested the raw data and proved it failed due to imbalance (Stage 2a)."
2.  **Intervention:** "We applied a specific technique to fix the specific problem (Upsampling)."
3.  **Validation:** "We re-tested and proved our technique worked, increasing accuracy by up to 47% (Stage 2b)."

This proves you didn't just run code blindly; you acted like scientists diagnosing and solving a data problem.
