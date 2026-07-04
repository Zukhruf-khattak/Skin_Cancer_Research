# Automated Skin Cancer Detection and Classification Using Deep Learning

**Thesis Outline**

## Chapter 1: Introduction
- 1.1 Background and Motivation
- 1.2 Problem Statement
- 1.3 Aims and Objectives
- 1.4 Scope of the Study
- 1.5 Thesis Organization

## Chapter 2: Literature Review
- 2.1 Traditional Diagnostic Methods
- 2.2 Machine Learning in Dermatology
- 2.3 Deep Learning Approaches (CNNs, Transfer Learning)
- 2.4 Handling Class Imbalance
- 2.5 Interpretability in Medical AI (Grad-CAM)
- 2.6 Research Gaps

## Chapter 3: Methodology
- 3.1 Dataset Description (ISIC 2019)
- 3.2 Proposed Two-Stage Framework
- 3.3 Preprocessing Pipeline
  - 3.3.1 Hair Removal
  - 3.3.2 Lesion Detection and Cropping
  - 3.3.3 Data Augmentation
- 3.4 Stage 1: Binary Classification (Benign vs. Malignant)
  - 3.4.1 Architectures (ResNet50, EfficientNet-B0, MobileNetV2)
- 3.5 Stage 2: Malignant Subtype Classification
  - 3.5.1 Addressing Class Imbalance via Upsampling
- 3.6 Generative Adversarial Networks (GANs) for Data Augmentation
- 3.7 Evaluation Metrics

## Chapter 4: Results and Discussion
- 4.1 Stage 1: Binary Classification Results
- 4.2 Stage 2a: Multi-class Classification (Baseline)
- 4.3 Stage 2b: Multi-class Classification (With Upsampling)
  - 4.3.1 Impact of Upsampling on Minority Classes (SCC)
- 4.4 GAN Augmentation Analysis
- 4.5 Model Interpretability (Grad-CAM Analysis)
- 4.6 Discussion of Findings

## Chapter 5: Conclusion and Future Work
- 5.1 Summary of Contributions
- 5.2 Limitations
- 5.3 Future Directions

## References
