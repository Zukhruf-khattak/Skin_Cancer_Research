import re
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_LINE_SPACING

INPUT_FILE = "Huma Thesis_Updated.docx"
OUTPUT_FILE = "Huma_Thesis_Formatted.docx"

print(f"Loading {INPUT_FILE}...")
doc = Document(INPUT_FILE)

def set_run_font(run, size_pt=12, bold=None):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    if bold is not None:
        run.bold = bold

def format_paragraph(para, size=12, bold=None, spacing=WD_LINE_SPACING.ONE_POINT_FIVE):
    para.paragraph_format.line_spacing_rule = spacing
    for run in para.runs:
        set_run_font(run, size, bold if bold is not None else run.bold)

def format_heading(para, style_name, size):
    para.style = style_name
    format_paragraph(para, size=size, bold=True)

print("Applying formatting...")
for para in doc.paragraphs:
    text = para.text.strip()
    if not text:
        continue
        
    # Format Chapter Headings (Heading 1)
    if re.match(r'^CHAPTER\s+[IVX]+', text, re.IGNORECASE) or re.match(r'^Chapter\s+\d+', text, re.IGNORECASE):
        format_heading(para, 'Heading 1', 18)
        
    # Heading 2 (e.g., 1.1, 2.10, 4.2)
    elif re.match(r'^[1-9]\.\d+\s+[A-Za-z]', text):
        format_heading(para, 'Heading 2', 16)
        
    # Heading 3 (e.g., 1.1.1)
    elif re.match(r'^[1-9]\.\d+\.\d+\s+[A-Za-z]', text) or re.match(r'^[1-9]\.\d+\.\d+\s+—', text):
        format_heading(para, 'Heading 3', 14)
        
    # Heading 4 or irregular like 4.2.1b
    elif re.match(r'^[1-9]\.\d+\.\d+\.\d+\s+[A-Za-z]', text) or re.match(r'^[1-9]\.\d+\.\d+[a-z]\s+', text):
        format_heading(para, 'Heading 4', 12)
        
    # Unnumbered known headings (Focal Loss, MixUp)
    elif text in ["Focal Loss", "MixUp Data Augmentation"]:
        # Style as bold normal
        para.style = 'Normal'
        format_paragraph(para, size=12, bold=True)
        
    # Regular text
    else:
        # Keep existing style if it's already a valid non-heading style, but enforce font
        if para.style.name.startswith('Heading'):
            # If it's a heading style but doesn't match our regex, it might be an unnumbered section like 'Results'
            # Let's keep it but enforce font.
            if para.style.name == 'Heading 1':
                format_paragraph(para, size=18, bold=True)
            elif para.style.name == 'Heading 2':
                format_paragraph(para, size=16, bold=True)
            elif para.style.name == 'Heading 3':
                format_paragraph(para, size=14, bold=True)
            else:
                format_paragraph(para, size=12)
        else:
            format_paragraph(para, size=12)

# Update document default styles just in case
styles = doc.styles
for style_name, size in [('Normal', 12), ('Heading 1', 18), ('Heading 2', 16), ('Heading 3', 14), ('Heading 4', 12)]:
    try:
        style = styles[style_name]
        style.font.name = 'Times New Roman'
        style.font.size = Pt(size)
        style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    except KeyError:
        pass

print(f"Saving to {OUTPUT_FILE}...")
doc.save(OUTPUT_FILE)
print("Done!")
