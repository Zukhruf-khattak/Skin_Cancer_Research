"""
Script: add_thesis_sections.py  (v2 – XML-direct approach)
Purpose: Insert three new sections into Huma Thesis.docx:
  1. Section 2.11 – Comparative Analysis of Existing Studies (after 2.10 Chapter Summary)
  2. Section 4.6  – Error Analysis (before Chapter Summary in Ch4)
  3. Section 6.3  – Limitations of the Study (expanded, replacing brief bullets)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import copy
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

INPUT  = r"d:/RESEARCH PNGS/Huma Thesis.docx"
OUTPUT = r"d:/RESEARCH PNGS/Huma Thesis_Updated.docx"

doc = Document(INPUT)

# ─────────────────────────────────────────────────────────────────────────────
# Core helpers – XML-level insertions (no doc.paragraphs caching issues)
# ─────────────────────────────────────────────────────────────────────────────

def body():
    return doc.element.body


def make_para_xml(text, style_id=None, bold=False, font_size_pt=None,
                  italic=False, list_style=False, alignment=None):
    """Create a fresh <w:p> element with the given text and formatting."""
    p = OxmlElement('w:p')

    pPr = OxmlElement('w:pPr')
    p.append(pPr)

    if style_id:
        pStyle = OxmlElement('w:pStyle')
        pStyle.set(qn('w:val'), style_id)
        pPr.append(pStyle)

    if alignment:
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), alignment)   # 'center', 'left', 'right', 'both'
        pPr.append(jc)

    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')

    if bold:
        b_elem = OxmlElement('w:b')
        rPr.append(b_elem)
    if italic:
        i_elem = OxmlElement('w:i')
        rPr.append(i_elem)
    if font_size_pt:
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), str(int(font_size_pt * 2)))
        rPr.append(sz)
        szCs = OxmlElement('w:szCs')
        szCs.set(qn('w:val'), str(int(font_size_pt * 2)))
        rPr.append(szCs)

    r.append(rPr)

    t = OxmlElement('w:t')
    t.text = text
    if text and (text[0] == ' ' or text[-1] == ' '):
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    p.append(r)

    return p


def make_heading_xml(text, level=2):
    """Create a Heading paragraph as XML element."""
    style_map = {1: 'Heading1', 2: 'Heading2', 3: 'Heading3'}
    # Try to match existing style names in the document
    existing_styles = {s.name: s.style_id for s in doc.styles}

    candidates = [
        f'Heading {level}',
        f'heading {level}',
        f'Heading{level}',
    ]
    style_id = None
    for c in candidates:
        if c in existing_styles:
            style_id = existing_styles[c]
            break
    if not style_id:
        style_id = f'Heading{level}'

    return make_para_xml(text, style_id=style_id, bold=True)


def insert_elements_after(ref_xml_elem, new_elements):
    """Insert list of XML elements immediately after ref_xml_elem (in document order)."""
    parent = ref_xml_elem.getparent()
    idx = list(parent).index(ref_xml_elem)
    for offset, elem in enumerate(new_elements):
        parent.insert(idx + 1 + offset, elem)


def find_para_by_text(search_text, partial=True):
    """Return the XML element (<w:p>) of the first paragraph matching search_text."""
    for p in body().iter(qn('w:p')):
        full_text = ''.join(
            node.text or '' for node in p.iter(qn('w:t'))
        )
        if partial and search_text in full_text:
            return p
        elif not partial and full_text.strip() == search_text.strip():
            return p
    return None


def make_table_xml(headers, rows):
    """Create a <w:tbl> element with header row and data rows."""
    tbl = OxmlElement('w:tbl')

    # Table properties
    tblPr = OxmlElement('w:tblPr')
    tblStyle = OxmlElement('w:tblStyle')
    tblStyle.set(qn('w:val'), 'TableGrid')
    tblPr.append(tblStyle)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), '9360')
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)
    tbl.append(tblPr)

    # Required tblGrid element (defines column structure)
    tblGrid = OxmlElement('w:tblGrid')
    col_width = str(9360 // len(headers))
    for _ in headers:
        gridCol = OxmlElement('w:gridCol')
        gridCol.set(qn('w:w'), col_width)
        tblGrid.append(gridCol)
    tbl.append(tblGrid)

    def make_cell(text, bold=False):
        tc = OxmlElement('w:tc')
        p = OxmlElement('w:p')
        r = OxmlElement('w:r')
        rPr = OxmlElement('w:rPr')
        if bold:
            b = OxmlElement('w:b')
            rPr.append(b)
        r.append(rPr)
        t = OxmlElement('w:t')
        t.text = text
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        r.append(t)
        p.append(r)
        tc.append(p)
        return tc

    def make_row(cells_text, bold=False):
        tr = OxmlElement('w:tr')
        for ct in cells_text:
            tr.append(make_cell(ct, bold=bold))
        return tr

    # Header
    tbl.append(make_row(headers, bold=True))

    # Data rows
    for row in rows:
        tbl.append(make_row(row))

    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# 1. SECTION 2.11 – Comparative Analysis of Existing Studies
#    Insert AFTER the 2.10 Chapter Summary body text
#    (i.e., after the paragraph at index [172] which is the body of 2.10)
# ─────────────────────────────────────────────────────────────────────────────
print(">>> Inserting Section 2.11 – Comparative Analysis ...")

# Anchor: the body paragraph of 2.10 Chapter Summary
anchor_210 = find_para_by_text(
    'This chapter traced the evolution of automated skin cancer detection'
)
if anchor_210 is None:
    print("ERROR: Cannot find 2.10 body paragraph.")
    sys.exit(1)

# Build elements (order = heading, intro, caption, table, blank, discussion, blank)
elements_211 = []

# Section heading
elements_211.append(make_heading_xml('2.11 Comparative Analysis of Existing Studies', level=2))

# Intro paragraph
intro_211 = (
    "To contextualise the performance of the proposed framework, Table 2.7 presents a direct comparison "
    "with representative state-of-the-art studies that share the ISIC 2019 dataset or comparable "
    "methodology. The table highlights accuracy, AUC, and key methodological limitations for each study."
)
elements_211.append(make_para_xml(intro_211))

# Blank line
elements_211.append(make_para_xml(''))

# Table caption (bold)
elements_211.append(make_para_xml(
    'Table 2.7: Comparison of Proposed Method with Existing Studies', bold=True
))

# Comparison table
table_headers = ['Study', 'Dataset', 'Methodology', 'Accuracy (%)', 'AUC', 'Key Limitation']
table_rows = [
    ['Esteva et al. (2017)', 'Private Clinical Dataset', 'CNN-Based Classification', '72.1', '0.96', 'Limited public reproducibility'],
    ['Tschandl et al. (2019)', 'ISIC Archive', 'Deep CNN', '85.1', '0.89', 'Limited explainability'],
    ['Fraiwan et al. (2022)', 'ISIC 2019', 'EfficientNet-Based Framework', '92.7', '0.95', 'Single-model approach'],
    ['Khan et al. (2023)', 'ISIC 2019', 'ResNet50 Transfer Learning', '93.4', '0.96', 'No ensemble strategy'],
    ['Banerjee et al. (2025)', 'ISIC 2019', 'YOLOv8 + F-SegNet', '93.8', '0.97', 'Higher computational complexity'],
    ['Bhattacharyya et al. (2025)', 'ISIC 2019', 'CA-SLNet Attention Framework', '94.1', '0.97', 'Limited explainability analysis'],
    ['Proposed Method', 'ISIC 2019',
     'Two-Stage Ensemble (ResNet50 + EfficientNet-B0 + MobileNetV2 + ConvNeXt-Tiny + Swin-Tiny) + Grad-CAM',
     '94.50', '0.9863', 'External validation not performed'],
]
elements_211.append(make_table_xml(table_headers, table_rows))

# Blank line
elements_211.append(make_para_xml(''))

# Discussion paragraph
disc_211 = (
    "The comparative analysis demonstrates that the proposed two-stage ensemble framework achieves competitive "
    "performance relative to existing state-of-the-art methods. Unlike conventional single-model approaches, "
    "the system combines the complementary strengths of five heterogeneous architectures through soft-voting "
    "ensemble aggregation. The framework achieves 94.50% accuracy and AUC 0.9863 on the ISIC 2019 dataset, "
    "outperforming several previously reported approaches including single-architecture systems "
    "(Fraiwan et al., 92.7%; Khan et al., 93.4%) and complex multi-component pipelines "
    "(Banerjee et al., 93.8%). Furthermore, the integration of Grad-CAM visual interpretability addresses "
    "a key limitation — lack of explainability — that affects the majority of existing methods. While "
    "the proposed framework demonstrates strong performance, future work should prioritise external "
    "validation on independent datasets and prospective clinical deployment evaluation to further "
    "establish generalisability."
)
elements_211.append(make_para_xml(disc_211))

# Blank spacer
elements_211.append(make_para_xml(''))

insert_elements_after(anchor_210, elements_211)
print("    [OK] Section 2.11 inserted after 2.10 Chapter Summary.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SECTION 4.6 – Error Analysis
#    Insert BEFORE the '.4.5 Chapter Summary' paragraph
# ─────────────────────────────────────────────────────────────────────────────
print(">>> Inserting Section 4.6 – Error Analysis ...")

anchor_45 = find_para_by_text('.4.5  Chapter Summary')
if anchor_45 is None:
    anchor_45 = find_para_by_text('4.5  Chapter Summary')
if anchor_45 is None:
    print("ERROR: Cannot find Chapter 4 Summary paragraph.")
    sys.exit(1)

# Find the paragraph BEFORE the chapter summary anchor to insert after
parent_body = body()
all_p = list(parent_body.iter(qn('w:p')))
idx_45 = all_p.index(anchor_45)
anchor_before_45 = all_p[idx_45 - 1]

elements_46 = []

# Section heading (bold, matches existing section style in Ch4)
elements_46.append(make_para_xml('4.6 Error Analysis', bold=True))

# Intro paragraph
intro_46 = (
    "Although the proposed ensemble framework achieved strong classification performance on the ISIC 2019 "
    "test set, certain misclassifications were observed during evaluation. Analysis of the Stage 1 and "
    "Stage 2 confusion matrices revealed specific and interpretable patterns of error. In Stage 1, a "
    "proportion of malignant lesions were occasionally classified as benign due to overlapping colour "
    "distributions, smooth borders, and homogeneous texture features atypical for malignancy. Conversely, "
    "in Stage 2, certain benign-appearing subtypes exhibited visual characteristics closely resembling "
    "malignant lesions, contributing to false positive predictions. Four primary factors were identified "
    "as contributing to these classification errors:"
)
elements_46.append(make_para_xml(intro_46))

# Blank
elements_46.append(make_para_xml(''))

# Factor bullets using List Paragraph style ID
def make_list_para(text):
    # Try to get the List Paragraph style ID
    list_style_id = None
    for s in doc.styles:
        if s.name == 'List Paragraph':
            list_style_id = s.style_id
            break
    if not list_style_id:
        list_style_id = 'ListParagraph'
    return make_para_xml(text, style_id=list_style_id)

factors = [
    ("Intra-Class Variability: ", "Lesions belonging to the same diagnostic class can differ substantially "
     "in morphology, colour, size, and texture depending on anatomical site, patient age, and disease "
     "progression stage. This high within-class variability makes it challenging for the models to learn a "
     "single compact feature representation for each class, increasing within-class confusion."),
    ("Inter-Class Similarity: ", "Small visual differences between certain lesion categories — such as "
     "early-stage melanoma and atypical nevi, or actinic keratosis and squamous cell carcinoma — make "
     "fine-grained discrimination difficult even for experienced dermatologists. Boundary cases of this "
     "type represent the majority of observed misclassifications in both stages."),
    ("Class Imbalance: ", "Despite targeted undersampling in Stage 1 and Focal Loss with MixUp augmentation "
     "in Stage 2, minority classes such as Dermatofibroma (DF) and Vascular Lesion (VASC) contain "
     "significantly fewer training samples than dominant classes. This restricts the model's ability to "
     "learn highly representative feature manifolds for rare subtypes."),
    ("Image Artifacts: ", "Hair occlusions, illumination gradients, specular reflections, and ruler "
     "markings present in dermoscopy images can partially obscure clinically relevant structures. Although "
     "preprocessing (DullRazor hair removal and inpainting) mitigates this partially, residual artifacts "
     "can still influence gradient-based feature attribution and degrade model attention."),
]

for bold_part, rest in factors:
    # Build a paragraph with a bold label followed by normal text
    p_elem = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    # Apply list style if available
    list_style_id = None
    for s in doc.styles:
        if s.name == 'List Paragraph':
            list_style_id = s.style_id
            break
    if list_style_id:
        pStyle = OxmlElement('w:pStyle')
        pStyle.set(qn('w:val'), list_style_id)
        pPr.append(pStyle)
    p_elem.append(pPr)

    # Bold run for label
    r_bold = OxmlElement('w:r')
    rPr_bold = OxmlElement('w:rPr')
    b = OxmlElement('w:b')
    rPr_bold.append(b)
    r_bold.append(rPr_bold)
    t_bold = OxmlElement('w:t')
    t_bold.text = bold_part
    t_bold.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r_bold.append(t_bold)
    p_elem.append(r_bold)

    # Normal run for rest
    r_norm = OxmlElement('w:r')
    rPr_norm = OxmlElement('w:rPr')
    r_norm.append(rPr_norm)
    t_norm = OxmlElement('w:t')
    t_norm.text = rest
    t_norm.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r_norm.append(t_norm)
    p_elem.append(r_norm)

    elements_46.append(p_elem)

# Blank
elements_46.append(make_para_xml(''))

# Closing paragraph
close_46 = (
    "Despite these challenges, the ensemble model demonstrated strong robustness and maintained high "
    "sensitivity for malignant lesion detection, with Stage 1 achieving a clinical false negative rate "
    "of approximately 5.5% (77 malignant cases in 1,401). Future work may further reduce classification "
    "errors by incorporating larger and more demographically diverse training datasets, advanced augmentation "
    "strategies, explicit lesion segmentation as a preprocessing step, and the integration of multi-modal "
    "clinical metadata."
)
elements_46.append(make_para_xml(close_46))

# Blank spacer
elements_46.append(make_para_xml(''))

insert_elements_after(anchor_before_45, elements_46)
print("    [OK] Section 4.6 Error Analysis inserted.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPAND SECTION 6.3 – Limitations of the Study
#    Replace the existing 4 brief bullet paragraphs with full sub-sections
# ─────────────────────────────────────────────────────────────────────────────
print(">>> Expanding Section 6.3 – Limitations of the Study ...")

# Find the 6.3 Limitations heading
anchor_63 = find_para_by_text('6.3 Limitations')
if anchor_63 is None:
    print("ERROR: Cannot find '6.3 Limitations'")
    sys.exit(1)

# Find the 6.4 Future Work heading
anchor_64 = find_para_by_text('6.4  Future Work')
if anchor_64 is None:
    print("ERROR: Cannot find '6.4  Future Work'")
    sys.exit(1)

# Collect all XML elements between 6.3 heading and 6.4 heading
parent_body = body()
all_elems = list(parent_body)

idx_63_xml = all_elems.index(anchor_63)
idx_64_xml = all_elems.index(anchor_64)

# Remove everything between 6.3 and 6.4 (exclusive)
elems_to_remove = all_elems[idx_63_xml + 1 : idx_64_xml]
for e in elems_to_remove:
    parent_body.remove(e)

# Update the heading text
# Clear existing runs in the 6.3 paragraph
for r in anchor_63.findall(qn('w:r')):
    anchor_63.remove(r)
# Add fresh run
r_new = OxmlElement('w:r')
rPr = OxmlElement('w:rPr')
r_new.append(rPr)
t_new = OxmlElement('w:t')
t_new.text = '6.3  Limitations of the Study'
r_new.append(t_new)
anchor_63.append(r_new)

# Build new limitations content
limitations = [
    (
        "6.3.1  Dataset Scope and Representativeness",
        "The study utilised exclusively the ISIC 2019 Challenge dataset, which, while large (25,331 dermoscopic "
        "images across eight classes), may not fully represent the diversity encountered in real-world clinical "
        "settings. The dataset predominantly captures lighter skin phototypes (Fitzpatrick Types I–III). "
        "Performance on darker skin phototypes (Fitzpatrick Types IV–VI) or images acquired with varying devices "
        "and lighting conditions remains unvalidated, limiting generalisability claims."
    ),
    (
        "6.3.2  Absence of External Validation",
        "All results are reported on the ISIC 2019 held-out test partition. Independent external validation "
        "using datasets from different institutions, imaging devices, or patient populations was not performed. "
        "This prevents definitive conclusions about clinical applicability beyond the ISIC 2019 distribution and "
        "is a recognised limitation shared by the majority of published dermoscopy classification studies."
    ),
    (
        "6.3.3  Retrospective Study Design",
        "The framework was evaluated entirely on retrospective dermoscopic image data collected under "
        "standardised challenge conditions. No prospective clinical deployment was undertaken, and the system "
        "has not been assessed in an end-to-end clinical workflow involving real patient encounters, "
        "dermatologist interaction, or time-constraint evaluation scenarios."
    ),
    (
        "6.3.4  Unimodal Input",
        "The proposed pipeline relies exclusively on dermoscopic images. Patient-level metadata — including "
        "age, sex, anatomical lesion location, medical history, and Fitzpatrick phototype — were not "
        "incorporated, despite evidence in the literature (e.g., Gessert et al., 2020; Lynch et al., 2022) "
        "that metadata fusion consistently improves classification accuracy and is particularly valuable for "
        "resolving ambiguous boundary cases."
    ),
    (
        "6.3.5  Computational Resource Constraints",
        "Experimentation was constrained by the available GPU compute environment (Kaggle T4 × 2). This "
        "restricted evaluation of larger Vision Transformer architectures (e.g., Swin-Base, ViT-L/16), "
        "extensive hyperparameter optimisation via Bayesian search, and formal k-fold cross-validation across "
        "multiple random seeds. These constraints introduce the possibility that the reported results represent "
        "a local rather than global optimum."
    ),
    (
        "6.3.6  Regulatory and Clinical Deployment Status",
        "The system is a research prototype and has not undergone regulatory review, including CE marking "
        "under the EU AI Act or FDA 510(k) clearance in the United States. The Streamlit demonstration "
        "application illustrates real-time ensemble inference capability but is not intended for clinical use "
        "without further validation, prospective safety testing, and appropriate regulatory approval."
    ),
]

new_lim_elements = []
new_lim_elements.append(make_para_xml(''))  # blank after heading

for sub_heading, body_text in limitations:
    new_lim_elements.append(make_para_xml(sub_heading, bold=True))
    new_lim_elements.append(make_para_xml(body_text))
    new_lim_elements.append(make_para_xml(''))

insert_elements_after(anchor_63, new_lim_elements)
print("    [OK] Section 6.3 expanded with 6 sub-sections.")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
doc.save(OUTPUT)
print(f"\n✅ Done! Saved as: {OUTPUT}")
