import re
from docx import Document

INPUT_FILE = "Huma Thesis_Updated.docx"

try:
    doc = Document(INPUT_FILE)
except Exception as e:
    print(f"Error loading document: {e}")
    exit(1)

print("Checking for implicit headings...")
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    style = para.style.name
    
    # Check if text looks like a heading (e.g. "1.1 Introduction", "Chapter 1", etc.)
    # but might not be marked as a heading.
    if text and len(text) < 150:
        match = re.match(r'^(Chapter\s+\d+|[1-9]\.\d+(\.\d+)?(\.\d+)?)\s+', text, re.IGNORECASE)
        if match:
            # Check if it's already a heading style
            if not style.startswith("Heading"):
                print(f"Para {i} | Style: {style} | Looks like heading: {text}")
        elif re.match(r'^[1-9]\s+[A-Z]', text) and len(text) < 60:
            if not style.startswith("Heading"):
                print(f"Para {i} | Style: {style} | Looks like heading: {text}")

