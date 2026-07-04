import docx
from docx.oxml.ns import qn
import re

def verify(filename):
    print(f"=== Verifying {filename} ===")
    try:
        doc = docx.Document(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return
        
    toc_found = False
    for i, p in enumerate(doc.paragraphs[:60]):
        text = p.text.strip().upper()
        if 'TABLE OF CONTENTS' in text or 'CONTENTS' in text:
            print(f"Found TOC heading at para {i}: {p.text.strip()}")
            
        for instr in p._p.findall('.//' + qn('w:instrText')):
            if instr.text and 'TOC' in instr.text:
                toc_found = True
                print(f"Found TOC field code at para {i}: {instr.text.strip()}")
                
    if not toc_found:
        print("WARNING: No TOC field code found!")
        
    print("\nSections:")
    for si, section in enumerate(doc.sections):
        sectPr = section._sectPr
        pnt = sectPr.find(qn('w:pgNumType'))
        fmt = pnt.get(qn('w:fmt')) if pnt is not None else 'none'
        start = pnt.get(qn('w:start')) if pnt is not None else 'none'
        print(f"  Section {si}: fmt={fmt}, start={start}")

verify(r'd:\RESEARCH PNGS\zukhruftheisis_final_version.docx')
