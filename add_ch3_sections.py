"""
add_ch3_sections.py
Inserts two new sections into Huma Thesis_Updated.docx:
  A. Section 3.2.5 – ISIC 2019 Dataset Characteristics
     (inserted after 3.2.4 Stratified Data Partitioning)
  B. Section 3.4.1 – Model Selection Justification
     (inserted after the 3.4 Model Architectures intro bullets, before 3.5)
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt

INPUT  = r"d:/RESEARCH PNGS/Huma Thesis_Updated.docx"
OUTPUT = r"d:/RESEARCH PNGS/Huma Thesis_Final.docx"

doc = Document(INPUT)

# ─────────────────────────────────────────────────────────────────────────────
# XML helpers
# ─────────────────────────────────────────────────────────────────────────────

def body():
    return doc.element.body

def find_para_by_text(search_text):
    for p in body().iter(qn('w:p')):
        full = ''.join(n.text or '' for n in p.iter(qn('w:t')))
        if search_text in full:
            return p
    return None

def insert_elements_after(ref_elem, elements):
    parent = ref_elem.getparent()
    idx = list(parent).index(ref_elem)
    for offset, elem in enumerate(elements):
        parent.insert(idx + 1 + offset, elem)

def make_para(text, bold=False, italic=False, style_id=None, font_size=None):
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    if style_id:
        ps = OxmlElement('w:pStyle')
        ps.set(qn('w:val'), style_id)
        pPr.append(ps)
    p.append(pPr)

    if not text:
        return p

    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    if bold:
        rPr.append(OxmlElement('w:b'))
    if italic:
        rPr.append(OxmlElement('w:i'))
    if font_size:
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), str(int(font_size * 2)))
        rPr.append(sz)
        szCs = OxmlElement('w:szCs')
        szCs.set(qn('w:val'), str(int(font_size * 2)))
        rPr.append(szCs)
    r.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    p.append(r)
    return p

def make_bold_label_para(label, rest, style_id=None):
    """Paragraph with a bold label followed by normal text."""
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    if style_id:
        ps = OxmlElement('w:pStyle')
        ps.set(qn('w:val'), style_id)
        pPr.append(ps)
    p.append(pPr)
    # Bold run
    r1 = OxmlElement('w:r')
    rPr1 = OxmlElement('w:rPr')
    rPr1.append(OxmlElement('w:b'))
    r1.append(rPr1)
    t1 = OxmlElement('w:t')
    t1.text = label
    t1.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r1.append(t1)
    p.append(r1)
    # Normal run
    r2 = OxmlElement('w:r')
    rPr2 = OxmlElement('w:rPr')
    r2.append(rPr2)
    t2 = OxmlElement('w:t')
    t2.text = rest
    t2.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r2.append(t2)
    p.append(r2)
    return p

def make_table_xml(headers, rows):
    tbl = OxmlElement('w:tbl')
    tblPr = OxmlElement('w:tblPr')
    tblStyle = OxmlElement('w:tblStyle')
    tblStyle.set(qn('w:val'), 'TableGrid')
    tblPr.append(tblStyle)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), '9360')
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)
    tbl.append(tblPr)
    # tblGrid
    tblGrid = OxmlElement('w:tblGrid')
    cw = str(9360 // len(headers))
    for _ in headers:
        gc = OxmlElement('w:gridCol')
        gc.set(qn('w:w'), cw)
        tblGrid.append(gc)
    tbl.append(tblGrid)

    def make_cell(text, bold=False):
        tc = OxmlElement('w:tc')
        p = OxmlElement('w:p')
        r = OxmlElement('w:r')
        rPr = OxmlElement('w:rPr')
        if bold:
            rPr.append(OxmlElement('w:b'))
        r.append(rPr)
        t = OxmlElement('w:t')
        t.text = text
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        r.append(t)
        p.append(r)
        tc.append(p)
        return tc

    def make_row(cells_data, bold=False):
        tr = OxmlElement('w:tr')
        for ct in cells_data:
            tr.append(make_cell(ct, bold=bold))
        return tr

    tbl.append(make_row(headers, bold=True))
    for row in rows:
        tbl.append(make_row(row))
    return tbl

def get_list_style_id():
    for s in doc.styles:
        if s.name == 'List Paragraph':
            return s.style_id
    return 'ListParagraph'

# ═════════════════════════════════════════════════════════════════════════════
# A. SECTION 3.2.5 – ISIC 2019 Dataset Characteristics
#    Insert after "3.2.4  Stratified Data Partitioning" body paragraph
# ═════════════════════════════════════════════════════════════════════════════
print(">>> Inserting Section 3.2.5 – ISIC 2019 Dataset Characteristics ...")

anchor_324 = find_para_by_text('Both stages share a common data partitioning strategy')
if anchor_324 is None:
    print("ERROR: Cannot find 3.2.4 body paragraph.")
    import sys; sys.exit(1)

lst = get_list_style_id()

elems_325 = []
elems_325.append(make_para(''))

# Sub-heading
elems_325.append(make_para('3.2.5  ISIC 2019 Dataset Characteristics', bold=True))

# Overview paragraph
elems_325.append(make_para(
    "The ISIC 2019 Challenge dataset constitutes the foundation of both pipeline stages. "
    "It comprises 25,331 dermoscopic images spanning eight distinct skin lesion categories, "
    "curated from multiple international dermatology centres and made publicly available through "
    "the International Skin Imaging Collaboration (ISIC) Archive. Table 3.0 summarises the "
    "class-level distribution across the complete dataset."
))
elems_325.append(make_para(''))

# Table caption
elems_325.append(make_para('Table 3.0 — ISIC 2019 Class Distribution', bold=True))

# Class distribution table
tbl_headers = ['Class Label', 'Abbreviation', 'Type', 'Images (n)', 'Proportion (%)']
tbl_rows = [
    ['Melanoma',                     'MEL',  'Malignant', '4,522',  '17.85'],
    ['Basal Cell Carcinoma',         'BCC',  'Malignant', '3,323',  '13.12'],
    ['Squamous Cell Carcinoma',      'SCC',  'Malignant',   '628',   '2.48'],
    ['Actinic Keratosis',            'AK',   'Malignant',   '867',   '3.42'],
    ['Melanocytic Nevus',            'NV',   'Benign',    '12,875',  '50.83'],
    ['Benign Keratosis',             'BKL',  'Benign',    '2,624',  '10.36'],
    ['Dermatofibroma',               'DF',   'Benign',      '239',   '0.94'],
    ['Vascular Lesion',              'VASC', 'Benign',      '253',   '1.00'],
    ['Total',                        '—',    '—',         '25,331', '100.00'],
]
elems_325.append(make_table_xml(tbl_headers, tbl_rows))
elems_325.append(make_para(''))

# Key characteristics as bold-label bullets
characteristics = [
    ("Image Resolution: ",
     "Images are provided in varying resolutions (typically 600×450 to 1024×768 pixels) and are "
     "standardised to 224×224 pixels during preprocessing to match the input requirements of all "
     "five model architectures."),
    ("Imaging Modality: ",
     "All images are dermoscopic (epiluminescence microscopy), captured using standardised "
     "dermoscopes under controlled clinical conditions at participating institutions. No "
     "clinical or macroscopic photographs are included."),
    ("Class Imbalance: ",
     "The dataset exhibits severe class imbalance. The Melanocytic Nevus (NV) class alone "
     "constitutes 50.83% of all images (12,875 samples), while the minority classes — "
     "Dermatofibroma (239 images, 0.94%) and Vascular Lesion (253 images, 1.00%) — are "
     "underrepresented by a factor of approximately 54:1 relative to NV. This necessitates "
     "dedicated class imbalance mitigation strategies (Section 3.2.2 and 3.6.2)."),
    ("Annotation Quality: ",
     "All diagnostic labels are derived from histopathological confirmation or expert "
     "dermatologist consensus, ensuring gold-standard ground truth for supervised learning. "
     "No crowd-sourced or weakly labelled samples are included."),
    ("Data Format: ",
     "Images are stored as JPEG files with an accompanying CSV metadata file providing "
     "the diagnostic label for each image identifier. No additional patient metadata "
     "(age, sex, anatomical site) was utilised in this study, though the metadata file "
     "contains such information."),
    ("Pre-existing Partitioning: ",
     "The ISIC 2019 challenge provided a separate unlabelled test set. For this study, "
     "a single stratified hold-out split (70% train / 15% validation / 15% test) was "
     "applied to the labelled portion, with patient-level stratification maintained via "
     "scikit-learn's StratifiedShuffleSplit to prevent data leakage across partitions."),
]

for label, rest in characteristics:
    elems_325.append(make_bold_label_para(label, rest, style_id=lst))

elems_325.append(make_para(''))

insert_elements_after(anchor_324, elems_325)
print("    [OK] Section 3.2.5 – ISIC 2019 Dataset Characteristics inserted.")


# ═════════════════════════════════════════════════════════════════════════════
# B. SECTION 3.4.1 – Model Selection Justification
#    Insert AFTER the 3.4 architecture bullet list (after Swin-Tiny bullet)
#    and BEFORE Section 3.5 Stage 1 Training
# ═════════════════════════════════════════════════════════════════════════════
print(">>> Inserting Section 3.4.1 – Model Selection Justification ...")

# Anchor: the blank paragraph between the bullets and Section 3.5 heading
# Find the Swin-Tiny bullet paragraph
anchor_swin = find_para_by_text(
    'Swin-Tiny (~28.3M parameters):  A Vision Transformer that computes self-attention'
)
if anchor_swin is None:
    print("ERROR: Cannot find Swin-Tiny bullet paragraph.")
    import sys; sys.exit(1)

elems_341 = []
elems_341.append(make_para(''))

# Section heading
elems_341.append(make_para('3.4.1  Model Selection Justification', bold=True))

# Intro
elems_341.append(make_para(
    "The selection of the five constituent architectures is not arbitrary but is grounded in "
    "the specific technical and clinical requirements of skin lesion classification. Each model "
    "was chosen to contribute a distinct inductive bias and representational strength to the "
    "ensemble, maximising the diversity of learned feature spaces and thus the ensemble's "
    "error-correction capability. The following sub-sections provide detailed justification "
    "for each architectural choice."
))
elems_341.append(make_para(''))

# 3.4.1.1 ResNet50
elems_341.append(make_para('3.4.1.1  ResNet50 — Deep Residual Feature Extraction', bold=True))
elems_341.append(make_para(
    "ResNet50 was selected as the foundational convolutional backbone due to its proven efficacy "
    "in medical imaging classification tasks (He et al., 2016). Its core innovation — the residual "
    "skip connection — resolves the vanishing gradient problem that afflicts very deep networks, "
    "enabling the extraction of hierarchical dermoscopic features across 50 layers without "
    "performance degradation. With approximately 25.6 million parameters, ResNet50 provides "
    "sufficient capacity to capture complex lesion morphology while remaining computationally "
    "tractable on available hardware. Its extensive ImageNet pre-training provides a rich "
    "initialisation of low-level texture and edge features directly applicable to dermoscopy, "
    "where texture and border irregularity are primary diagnostic cues. ResNet50 has been widely "
    "adopted as a baseline in dermoscopy classification literature, making it an appropriate "
    "anchor for performance comparison."
))
elems_341.append(make_para(''))

# 3.4.1.2 EfficientNet-B0
elems_341.append(make_para('3.4.1.2  EfficientNet-B0 — Parameter-Efficient Compound Scaling', bold=True))
elems_341.append(make_para(
    "EfficientNet-B0 was selected for its exceptional accuracy-to-parameter efficiency ratio "
    "(Tan and Le, 2019). Its compound scaling methodology — simultaneously scaling network depth, "
    "width, and resolution according to a fixed coefficient — enables EfficientNet-B0 to achieve "
    "accuracy competitive with architectures three to four times larger, while requiring only "
    "5.3 million parameters. This efficiency is clinically significant: it implies the model "
    "can be deployed on resource-constrained hardware without sacrificing diagnostic accuracy. "
    "The MBConv (Mobile Inverted Bottleneck) blocks employed by EfficientNet-B0 are well-suited "
    "to capturing the fine-grained colour gradients and subtle texture variations that "
    "differentiate malignant subtypes such as early-stage melanoma from benign nevi. "
    "EfficientNet architectures have consistently demonstrated top-tier performance on the "
    "ISIC benchmark, as evidenced by Gessert et al. (2020) achieving AUC 0.94 on ISIC 2019."
))
elems_341.append(make_para(''))

# 3.4.1.3 MobileNetV2
elems_341.append(make_para('3.4.1.3  MobileNetV2 — Lightweight Deployment-Oriented Architecture', bold=True))
elems_341.append(make_para(
    "MobileNetV2 was incorporated to represent the lightweight end of the architecture "
    "spectrum (Sandler et al., 2018). Its inverted residual blocks with linear bottlenecks "
    "reduce computational cost dramatically (3.4 million parameters) while preserving "
    "representational capacity for dermatological features. Including MobileNetV2 serves "
    "two purposes: it contributes predictions derived from a qualitatively different "
    "feature extraction pathway, enhancing ensemble diversity; and it demonstrates the "
    "pipeline's viability in mobile and edge deployment contexts — a practical consideration "
    "for low-resource healthcare settings. Studies such as Sarker et al. (2023) confirm "
    "that lightweight CNNs produce compact, clinically interpretable Grad-CAM heatmaps, "
    "supporting the explainability goals of this framework."
))
elems_341.append(make_para(''))

# 3.4.1.4 ConvNeXt-Tiny
elems_341.append(make_para('3.4.1.4  ConvNeXt-Tiny — Modernised CNN with Transformer Design Principles', bold=True))
elems_341.append(make_para(
    "ConvNeXt-Tiny bridges the architectural gap between classical CNNs and Vision Transformers "
    "(Liu et al., 2022). It incorporates design choices borrowed from ViT — larger kernel sizes "
    "(7×7), layer normalisation, GELU activations, and inverted bottleneck blocks — within a "
    "pure convolutional framework, capturing both local texture detail and broader contextual "
    "patterns without the quadratic attention cost of full self-attention. With approximately "
    "28.6 million parameters, it provides a complementary representational profile to both "
    "the traditional ResNet and the attention-based Swin-Tiny, making it a strategically "
    "important contributor to ensemble diversity. Its strong ImageNet-22K pre-training "
    "further enhances transfer learning effectiveness for dermoscopic classification."
))
elems_341.append(make_para(''))

# 3.4.1.5 Swin-Tiny
elems_341.append(make_para('3.4.1.5  Swin-Tiny — Hierarchical Vision Transformer for Global Context', bold=True))
elems_341.append(make_para(
    "Swin-Tiny was selected to introduce global spatial reasoning into the ensemble — a "
    "capability absent from purely convolutional architectures (Liu et al., 2021). Its "
    "shifted-window self-attention mechanism captures long-range spatial relationships "
    "between distant lesion regions, which is clinically relevant: features such as "
    "asymmetric pigment networks, irregular vascular patterns, and diffuse border "
    "irregularity span large portions of the lesion and benefit from global context. "
    "The shifted-window partitioning provides the model with hierarchical multi-scale "
    "representations while controlling the quadratic complexity of standard self-attention. "
    "Nersisson et al. (2024) demonstrated that Swin Transformer achieves 91.2% accuracy "
    "on ISIC 2019, confirming its suitability for dermoscopy classification. "
    "Including Swin-Tiny ensures the ensemble captures both fine-grained local texture "
    "(from CNNs) and global structural context (from the transformer)."
))
elems_341.append(make_para(''))

# 3.4.1.6 Ensemble Rationale
elems_341.append(make_para('3.4.1.6  Ensemble Rationale — Diversity as a Diagnostic Strength', bold=True))
elems_341.append(make_para(
    "The five architectures span three distinct computational families: classical residual CNNs "
    "(ResNet50), efficiency-optimised CNNs (EfficientNet-B0, MobileNetV2), a modernised CNN "
    "with transformer principles (ConvNeXt-Tiny), and a pure Vision Transformer (Swin-Tiny). "
    "This heterogeneity is intentional. Ensemble learning theory demonstrates that combining "
    "diverse, decorrelated base learners reduces both bias and variance in the aggregate "
    "prediction, improving generalisation beyond what any single model achieves "
    "(Dietterich, 2000). In the clinical context of skin lesion classification, where "
    "inter-class boundaries are ambiguous and misclassification of malignant lesions carries "
    "severe clinical consequences, this error-reduction property is of paramount importance. "
    "Soft-voting probability averaging was adopted as the aggregation strategy (Section 3.7) "
    "because it preserves confidence magnitudes and allows the ensemble to override an "
    "individual model's overconfident misclassification through the collective evidence "
    "of the remaining models."
))
elems_341.append(make_para(''))

# Summary table
elems_341.append(make_para('Table 3.0b — Model Selection Summary', bold=True))
sel_headers = ['Model', 'Parameters', 'Family', 'Key Strength', 'Selection Rationale']
sel_rows = [
    ['ResNet50',      '25.6M', 'Classical CNN',          'Deep hierarchical features',         'Baseline anchor; proven dermoscopy efficacy'],
    ['EfficientNet-B0','5.3M', 'Efficient CNN',          'High accuracy / low parameter count', 'Colour gradient & texture discrimination'],
    ['MobileNetV2',   '3.4M',  'Lightweight CNN',        'Minimal compute cost',               'Ensemble diversity; edge deployment viability'],
    ['ConvNeXt-Tiny', '28.6M', 'Modernised CNN',         'Transformer design principles',       'Bridges CNN and transformer feature spaces'],
    ['Swin-Tiny',     '28.3M', 'Vision Transformer',    'Global spatial context via attention', 'Long-range lesion structure capture'],
]
elems_341.append(make_table_xml(sel_headers, sel_rows))
elems_341.append(make_para(''))

insert_elements_after(anchor_swin, elems_341)
print("    [OK] Section 3.4.1 – Model Selection Justification inserted.")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
doc.save(OUTPUT)
print(f"\n✅ Saved → {OUTPUT}")
