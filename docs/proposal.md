Automated Skin Cancer Detection and Classification Using Deep Learning


Submitted By:

Zohrouf Khattak             Roll.No: 1000

Huma Zeb                       Roll.No: 883


Supervisor: Dr Muhammad Ayaz


Shaikh Zayed Islamic Centre, University Of PeshawarDepartment of Computer ScienceSession 2022-2026


Abstract

The study is based on a two-step deep learning model of automated skin cancer detection and classification with the use of the ISIC 2019 Dataset. The binary classification approach is applied in the first stage to determine benign and malignant skin lesions. This stage compares three convolutional neural network models. The second step involves through classification of malignant images into particular cancer subtypes such as melanoma, basal cell carcinoma, squamous cell carcinoma and actinic keratosis. In this multi-classification task, some other deep learning models are used. A preprocessing step including resizing, normalization, and data augmentation of the dermoscopic images is done prior to training to increase the performance and generalization of the model. The models are measured by various performance measures such as accuracy, precision, recall (sensitivity), F1-score and area under the receiver operating characteristic curve (AUC). Grad-CAM visualization is also applied to indicate the significant regions in the images that affect the model predictions to enhance model interpretability. The developed framework should be able to offer a stable and computationally inexpensive computer-assisted endoscopic system of identifying skin cancer. Through a comparative analysis of six deep learning models at two classification phases, the study will aim to establish which models are most effective in classifying skin lesions with great accuracy and at the same time have a practical application in real-life clinical settings.

1.  Introduction

Skin cancer is one of the most rapidly growing oncological entities in the world and the sixth most common type of cancer [1]. Its pathogenesis can be explained by the uncontrolled and abnormal growth of the cutaneous cellular elements, which is often triggered by the long-term exposure to ultraviolet (UV) radiation, genetic determinants, or immune system deficiencies. The resulting neoplasms of the skin are traditionally classified as benign and malignant; the form of benign growths, including nevi, are not life-threatening but malignant tumors have a life-threatening infectious ability due to their local invasion and distant metastasis. The main cell lineages that are prone to malignant transformation are basal cells, keratinocytes and melanocytes that give rise to basal cell carcinoma (BCC), squamous cell carcinoma (SCC), and melanoma respectively. The most lethal of them is melanoma due to its tendency to recur, despite therapeutic treatment [1].

Globally, it is estimated that 50,000 people die annually of skin cancer, which is about 0.7 percent of all cancer-related deaths, and costs associated with treatment are estimated to amount to approximately USD 30 million annually[2]. Globally, malignant melanoma is attributed to over 5,000 deaths annually and it is quite challenging to treat in its progressive stages[3]. Therefore, early diagnosis plays a critical role in improving the survival rates and reducing the general disease burden.

Traditional diagnostic methods, including dermatological visual diagnosis, can often be subjective, time-consuming, and prone to misclassify because phenotypic overlap between the various types of lesions can be great. To overcome these drawbacks, computer-aided diagnostic (CAD) systems based on imaging are being more widely used to aid dermatologists. The deep learning (DL) architecture, specifically the convolutional neural networks (CNNs) has demonstrated the ability to detect and classify a cutaneous lesion at the level of a dermatologist[4]. However, non-invasive image technologies like dermoscopy are still restricted by such limitations as noise, variability of illumination, artifacts, and inadequate spatial resolution which altogether reduce the quality of the examination[5].

This research aims to create an automated deep learning-based mechanism of detecting and classifying skin cancer using dermoscopic image. Furthermore, it will decrease the cases of misdiagnosis, aid dermatologists in making clinical judgement and eventually decrease the rates of death by increasing accuracy of diagnoses and efficiency of operations.


### 2. Literature Review


### 2.1 Machine Learning Approaches

Initial research in the field of skin cancer detection used classical machine-learning algorithms mostly. For example, applied Support Vector Machines (SVMs), k-nearest neighbors (k-NN), and Decision Trees to carry out classification operations [1]. The above studies showed that automated diagnosis was possible and provided the foundation of computer-aided diagnostic systems. However, the effectiveness of these was hampered by two major problems. To begin with, they required manual feature extraction, where researchers designed descriptors either based on color, texture, or shape, which makes the process extremely dependent on human knowledge. Second, the models were trained on relatively small datasets thus limiting their ability to generalize to different populations. It is these limitations that led to the creation of more complex ways that could be used to automatically learn more complex hierarchical representations of the data.


### 2.2 Deep Learning Approaches

The events of the recent past concerning artificial intelligence (AI) and deep learning have transformed the sphere of automated medical diagnosis, especially of skin cancer. Another 2025 review [6] that was published in the International Conference on Sentiment Analysis and Deep Learning (ICSADL) indicated the usefulness of various ML and DL models, such as Random Forests, Support Vector Machines (SVMs), Convolutional Neural Networks (CNNs), ResNet, and Inception models in melanoma detection. Nevertheless, continuous problems, including data imbalance, interpretability, computational needs, and model generalization in different populations were also noted in the study. These results indicate that more research should be conducted on the optimization of CNN models including EfficientNet, DenseNet, and ResNet50 with a focus on precise, compact, and generalizable skin lesion detection models.

The development of deep learning particularly the convolutional neural networks (CNNs) proved to be a milestone in studying medical pictures. The 7-trained on more than 129,000 clinical images, [7] was able to reach the dermatologist level of the task to differentiate between melanoma and keratinocyte carcinomas. The subsequent investigation by [8] was a comparison between CNNs and 58 dermatologists in terms of melanoma detection and determined that the models were quite accurate at the task as compared to the majority of humans. Likewise, [9] also compared CNN performance to 157 dermatologists and found out that AI systems tended to be the same or sometimes even more successful than a professional expert. These findings affirmed the validity of CNNs as trustworthy diagnostic patients in the dermatology sciences.

Research was also increased with benchmarking programs. [10]. Offered the ISIC 2018 challenge featuring big and standardized datasets and metrics of evaluation, which promoted comparisons taking place across models and innovation. A study conducted in [11] compared AI systems with those of the dermatologists in an international study with the findings saying that AI was just as accurate as human experts. Such success or not, the CNN-based systems had often been discovered to be computationally expensive, and required large datasets of annotations to perform optimally, which casts doubt on the scalability and accessibility issue.

The latest developments in 2025 have shown astounding advancement of CNN-based melanoma detection using transfer learning and attention. A two-stage system [12] using YOLOv8 to localize lesions and an improved version of F-SegNet to perform the segmentation process reached an accuracy of 98.50 on ISIC 2019 and PH2 datasets. Similarly, [13] proposed a hybrid network combining channel and spatial-attention (CA-SLNet) that uses ResNet50 and VGG16 and achieves a higher accuracy of more than 98 percent on various ISIC datasets. Such studies point out the ongoing transition towards attention-based, transfer-learning CNN models, which are more accurate in diagnosis and less costly to compute.

In a related study [14] built a deep learning classifier that used convolutional neural networks (CNNs) along with an artificial neural network (ANN) with patient metadata about several variables, including age, gender, and lesion location to detect malignant melanoma. The hybrid CNN+ANN model has struck the balance accuracy of 92.34 that was better than the CNN-only model with a balance accuracy of 73.69. This demonstrates that the incorporation of metadata is able to support better diagnostic performance and reduce overfitting. The current study is however restricted to image-only CNN architectures to guarantee simplicity, computationally efficient and reproducibility in low-resource environments.

The review of over 100 studies on deep learning-based segmentation and classification techniques to analyze skin lesions was conducted by [15]. The authors pointed at preprocessing, segmentation, and classification phases of dermoscopic image analysis and revealed low contrast, blurred lesion edges, and the lack of sufficient training data as the key complications. Such difficulties point to the fact that more effective and precise classification systems, such as the one suggested in the present research that is based on binary categorization of benign and malign lesions with the help of deep convolutional neural networks, are needed. Furthermore, [16] developed a computer-aided diagnosis system that identifies dermoscopic images in the ISIC 2019 data set in eight diagnostic categories in a deep convolutional neural network with transfer learning. They work on a pre-trained CNN backbone that is modified to a problem with eight classes (melanoma, melanocytic nevus, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, vascular lesion, and squamous cell carcinoma) and assess performance in terms of accuracy, precision, sensitivity, and specificity. The authors use data augmentation and class-imbalance management methods and indicate high overall accuracy with respect to the difficult ISIC 2019 benchmark. This study is commonly quoted as an informative foundation of full 8-class ISIC 2019 classification, but the high-class imbalance of the data and lack of external validation imply that there is still a question mark regarding the extrapolation to heterogeneous and real-world clinical environments.

2.3 Transfer Learning and Hybrid Models

Large data demands and high computing costs led to the increasing use of transfer learning by researchers to reduce these constraints. This plan takes the advantage of the pre-trained architectures like ResNet, DenseNet, and Inception, which are then fine-tuned on the classification of skin cancer. [17] Proposed a multi-scale multi-network ensemble model that is based on transfer learning and is more robust and superior to single CNNs. In a like manner, [4]  came up with a system that characterizes various pigmented skin lesions, therefore, going beyond binary categorization of benign versus malignant.

Subsequent improvement has shown that the transfer learning can reduce the reliance on large annotated datasets [2]. Used pre-trained CNNs and fine-tuned them to a high accuracy using relatively small corpora. Hybrid solutions also appear to be a promising direction [3] combined CNNs with ensemble algorithms to enhance generalization, and [18] have suggested a scalable computer-aided diagnosis (CAD) system that can be used in clinical practice. More recently, [5] trained deep learning models on dermoscopic images and achieved increased rates of melanoma detection and decreased rates of misdiagnosis. Together, these studies highlight the possibility of transfer learning and hybrid solutions to achieve efficiency and accuracy, and thus makes them more viable to be practically implemented.


### 2.4 Epidemiological Insights

In addition to the methodological improvements, epidemiology studies show how early and precise detection is crucial [19]  Have found that skin cancer is among the most dominant and fastest-growing types of cancers throughout the entire world, which reinforces the need to have dependable, convenient, and automated diagnostics. These results support the argument in favor of the implementation of AI-based solutions into the clinical practice especially in the area where the number of dermatologists is limited.

Previous research on skin cancer detection has made good progress, but many studies still have problems. Earlier machine learning techniques primarily relied on manually engineered features and small datasets, which limited their reliability and generalization [1]. However, with the advent of deep learning, especially convolutional neural networks (CNNs), the accuracy of skin lesion classification has improved significantly. CNNs can automatically learn and extract meaningful patterns from dermoscopic images. Several studies have reported that their performance is comparable to, or even exceeds, that of experienced dermatologists [7]. However, many challenges still exit, for example a major issue in skin cancer is class imbalance of dataset, where malignant cases are far fewer than benign ones. This imbalance causes models to perform poorly on the minority (malignant) class. Another challenge is model interpretability as most deep learning models act as “black boxes,” making it difficult for clinicians to understand or trust their decisions. Additionally, many high-performing models require large computational resources and are difficult to establish in low resource or clinical environments [6] However Some recent approaches also depend on additional patient information such as age, gender, or lesion location which may not always be available in real-world settings [14].

Nevertheless, the article [16] also contains certain significant limitations that are the reason to conduct a new research. First, it employs an eight-class, single-stage classifier, therefore, not explicitly trained to achieve maximum malignant-versus-benign sensitivity, an essential factor in clinical practice. Second, although the basic imbalance handling is utilized, the high level of class imbalance in ISIC 2019 cannot be eliminated, and the results on rare malignant classes might still not be the best. Third, the accuracy is in the center stage of the work, and model interpretability (e.g., Grad-CAM) or running on a low-resource device are not highlighted, and there is an unsolved question regarding clinical trust, transparency, and applicability in the real world.

To address these gaps, in this research, we will solve the mentioned issues by creating a simple and easy-to-run deep-learning system that only needs dermoscopic images. We use three lightweight transfer-learning models (ResNet50, DenseNet121, and EfficientNetB0) that work well even on normal computers. We also Grad-CAM to show which areas of the image the model pays attention to while making a prediction, so the results are easier to trust and interpret. Because the overall method is lightweight, simple to follow, and visually clear, it becomes a practical option even for settings where resources or advanced hardware are limited.


Research Gaps

- Most of the existing models fail to directly maximize the malignant-versus-benign sensitivity, which puts the practice of identifying dangerous cancers at higher risk.​
- Severe class imbalance in datasets like ISIC 2019 causes weak performance on rare but clinically important malignant classes.​
- The majority of deep learning models are black boxes with few visual explanations, which decrease the trust of clinicians and application in practice.
3. Problem Statement

One of the fastest spreading and most deadly cancers in the world is skin cancer, especially melanoma. Early diagnosis is very successful in enhancing survival rates; however, the current method of diagnosis mostly depends on visual examination by dermatologists, which is usually time-consuming, subjective, and expensive. In developing countries, where there is limited access to dermatologists, proper and timely diagnosis is even harder, resulting in late diagnosis and higher mortality. Although deep learning has greatly enhanced the precision of automated skin cancer detection, a number of challenges still exist. The earlier machine learning methods relied on manually engineered features and small datasets, which constrained their robustness and generalization ability. Even with deep learning models, the most serious problem is the dramatic disparity in class distributions, where benign samples greatly outnumber malignant ones, resulting in models having low performance on the minority (malignant) classes that are highly important in clinical scenarios. Another major problem is the lack of interpretability of the models. Most deep networks act as black boxes, providing minimal information on how they arrive at their decisions, which discourages trust among clinicians. Moreover, some of these high-performing models demand high-end computing resources and therefore cannot be used in low-resource or routine clinical settings. Certain recent research also depends on additional patient information, such as age or lesion location, which is not always available and thus makes such approaches less practical. In addition, some of the available literature focuses mainly on overall accuracy while ignoring both malignant-versus-benign sensitivity and model transparency, which are crucial for confident clinical use.

To eliminate these shortcomings, this study suggests a lightweight and interpretable deep-learning framework for skin cancer detection using solely dermoscopic images. It uses three effective transfer-learning models, namely ResNet50, DenseNet121, and EfficientNetB0, which can be applied on conventional computing devices. Visual explanations with Grad-CAM are employed to identify the locations in the image that the model pays attention to when making predictions, in order to increase interpretability and clinical trust. Due to the simplicity, resource efficiency, and visual interpretability of the system, it can be used as a practical and deployable tool for melanoma detection, particularly in low-resource medical imaging environments.

4. Aims and Objectives

Aim

To create a precise and effective deep-learning architecture of automated skin cancer detection and classification in dermoscopic images, which would be able to help clinicians in making an early diagnosis and enhance diagnostic accuracy.

Research Objectives

- Develop a machine learning model to accurately classify patients with skin lesion cancer.
- Compare different machine learning models for classifying skin lesion cancer.

5. Methodology

5.1 Dataset

The current research is based on the use of the data of the ISIC 2019, which is a universally recognized standard on the classification of lesions of the skin. In the dataset, it is possible to find dermoscopic images, divided into two major categories:

In the case of this project, we will conduct a two stage classification.

- Benign: Nevus, Dermatofibroma, Benign Keratosis, Vascular Lesions.
- Malignant: Basal Cell Carcinoma, Squamous Cell Carcinoma, Actinic Keratosis and Melanoma.
Though the ISIC 2019 data has eight diagnostic categories, the hierarchical structure proposed separates benign and malignant lesions first and only the latter is further differentiated into cancer subtypes.

5.2 Proposed Two-stage Framework.

Stage 1: Binary Classification.

Firstly, we divided the pictures into Benign vs. Malignant. We will experiment with a couple of solid deep-learning models, which are famous not only because they achieve good accuracy but also fast inference: ResNet -50, EfficientNet -B0 and MobileNet -V2. They are commonly used as medical imaging systems due to their light weight with high power.

Stage 2: Classification based on Malignancy Subtype.

When we report an image as malignant we will further explore it and in more detail classify it into one of four categories Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma and Actinic Keratosis. To do this step, we are going to compare the DenseNet 121, Inception 3, and VGG 16 models. By using the three in running we are able to make a comprehensive comparison of the manner in which each of the CNN shapes up.

5.3 Preprocessing

- The images are normalized and resized to 224x224.
- The model is robustened by data augmentation (rotation, flipping, zooming, change of brightness) to enhance the model.
5.4 Training

- Transfer learning will be applied using ImageNet weights to ensure that the networks do not need to be trained initially.
- Training settings are:
- Optimizer: Adam
- Batch size: 32
- Early stopping is used to combat overfitting.
- Loss: Binary Cross-Entropy in Stage 1 and categorical cross-entropy in Stage 2.
5.5 Evaluation Metrics

- We will measure each of the models with the usual measures: Accuracy, Precision, Recall (Sensitivity), F1 -Score, AUC and a Confusion Matrix.
- A comparison of all the six models will reveal the architecture that is superior.
- Important image regions are visualized with the help of Grad-CAM to be interpreted.
5.6 Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is the one that creates activation heatmaps, which make the model easier to understand because they show parts of the image that cause its decisions. The positive and negative classifications are both analyzed qualitatively to gain a better insight into predictive behavior.


Figure 1: Proposed Methodology for Automated Skin Cancer Detection and Classification using

Deep   Learning


6. Expected Outcomes

We are hopeful that we will develop a powerful and effective system that will be able to identify skin cancer using dermoscopic images. The community will have a good comparison of the various architectures by testing six deep-learning models on our two-stage pipeline. The goal is to  Properly identify benign and malignant lesions. Exactly categorize malignant subtypes. Provide true Grad-CAM visual interpretations. Perform at the level of dermatologists without being too heavy to be operated in low-resource configurations.


7. References

[1]M. K. Monika, N. Arun Vignesh, Ch. Usha Kumari, M. N. V. S. S. Kumar, and E. L. Lydia, “Skin cancer detection and classification using machine learning,” Mater. Today Proc., vol. 33, pp. 4266–4270, 2020, doi: 10.1016/j.matpr.2020.07.366.

[2]M. Fraiwan and E. Faouri, “On the Automatic Detection and Classification of Skin Cancer Using Deep Transfer Learning,” Sensors, vol. 22, no. 13, p. 4963, Jun. 2022, doi: 10.3390/s22134963.

[3]A. Bassel, A. B. Abdulkareem, Z. A. A. Alyasseri, N. S. Sani, and H. J. Mohammed, “Automatic Malignant and Benign Skin Cancer Classification Using a Hybrid Deep Learning Approach,” Diagnostics, vol. 12, no. 10, p. 2472, Oct. 2022, doi: 10.3390/diagnostics12102472.

[4]S. Jinnai, N. Yamazaki, Y. Hirano, Y. Sugawara, Y. Ohe, and R. Hamamoto, “The Development of a Skin Cancer Classification System for Pigmented Skin Lesions Using Deep Learning,” Biomolecules, vol. 10, no. 8, p. 1123, Jul. 2020, doi: 10.3390/biom10081123.

[5]A. Abdulrahman Albraikan, N. Nemri, M. Abdullah Alkhonaini, A. Mustafa Hilal, I. Yaseen, and A. Motwakel, “Automated Deep Learning Based Melanoma Detection and Classification Using Biomedical Dermoscopic Images,” Comput. Mater. Contin., vol. 74, no. 2, pp. 2443–2459, 2023, doi: 10.32604/cmc.2023.026379.

[6]K. Wanjari and P. Verma, “A Review on the Applications of Machine Learning and Deep Learning Techniques for Skin Cancer Detection,” in 2025 4th International Conference on Sentiment Analysis and Deep Learning (ICSADL), Bhimdatta, Nepal: IEEE, Feb. 2025, pp. 1718–1723. doi: 10.1109/ICSADL65848.2025.10933289.

[7]A. Esteva et al., “Dermatologist-level classification of skin cancer with deep neural networks,” Nature, vol. 542, no. 7639, pp. 115–118, Feb. 2017, doi: 10.1038/nature21056.

[8]H. A. Haenssle et al., “Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists,” Ann. Oncol., vol. 29, no. 8, pp. 1836–1842, Aug. 2018, doi: 10.1093/annonc/mdy166.

[9]T. J. Brinker et al., “Deep learning outperformed 136 of 157 dermatologists in a head-to-head dermoscopic melanoma image classification task,” Eur. J. Cancer, vol. 113, pp. 47–54, May 2019, doi: 10.1016/j.ejca.2019.04.001.

[10]N. Codella et al., “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC),” 2019, arXiv. doi: 10.48550/ARXIV.1902.03368.

[11]P. Tschandl et al., “Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification: an open, web-based, international, diagnostic study,” Lancet Oncol., vol. 20, no. 7, pp. 938–947, Jul. 2019, doi: 10.1016/S1470-2045(19)30333-X.

[12]S. Banerjee, S. Bhattacharyya, and S. N. Saha, “Advanced Melanoma Detection Using Deep Transfer Learning,” Int. Res. J. Multidiscip. Scope, vol. 06, no. 01, pp. 1239–1252, 2025, doi: 10.47857/irjms.2025.v06i01.02938.

[13]M. A. Naeem, S. Yang, M. Mittal, J. Narayan, M. A. Saleem, and M. Shabaz, “Deep feature fusion for melanoma detection using ResNet-VGG networks with attention mechanism,” PeerJ Comput. Sci., vol. 11, p. e3248, Oct. 2025, doi: 10.7717/peerj-cs.3248.

[14]D. N. A. Ningrum et al., “Deep Learning Classifier with Patient’s Metadata of Dermoscopic Images in Malignant Melanoma Detection,” J. Multidiscip. Healthc., vol. Volume 14, pp. 877–885, Apr. 2021, doi: 10.2147/JMDH.S306284.

[15]R. Baig, M. Bibi, A. Hamid, S. Kausar, and S. Khalid, “Deep Learning Approaches Towards Skin Lesion Segmentation and Classification from Dermoscopic Images - A Review,” Curr. Med. Imaging Former. Curr. Med. Imaging Rev., vol. 16, no. 5, pp. 513–533, May 2020, doi: 10.2174/1573405615666190129120449.

[16]M. A. Kassem, K. M. Hosny, and M. M. Fouad, “Skin Lesions Classification Into Eight Classes for ISIC 2019 Using Deep Convolutional Neural Network and Transfer Learning,” IEEE Access, vol. 8, pp. 114822–114832, 2020, doi: 10.1109/ACCESS.2020.3003890.

[17]A. Mahbod, G. Schaefer, C. Wang, G. Dorffner, R. Ecker, and I. Ellinger, “Transfer learning using a multi-scale and multi-network ensemble for skin lesion classification,” Comput. Methods Programs Biomed., vol. 193, p. 105475, Sep. 2020, doi: 10.1016/j.cmpb.2020.105475.

[18]D. Adla, G. V. R. Reddy, P. Nayak, and G. Karuna, “Deep learning-based computer aided diagnosis model for skin cancer detection and classification,” Distrib. Parallel Databases, vol. 40, no. 4, pp. 717–736, Dec. 2022, doi: 10.1007/s10619-021-07360-z.

[19]R. L. Siegel, K. D. Miller, H. E. Fuchs, and A. Jemal, “Cancer statistics, 2022,” CA. Cancer J. Clin., vol. 72, no. 1, pp. 7–33, Jan. 2022, doi: 10.3322/caac.21708.

