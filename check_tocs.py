import docx
from docx.oxml.ns import qn
import re

def check_toc(filename):
    print(f"=== Checking {filename} ===")
    try:
        doc = docx.Document(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return
        
    toc_paras = []
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip().upper()
        if 'TABLE OF CONTENTS' in text or 'CONTENTS' in text:
            toc_paras.append(i)
            print(f"Found TOC heading at para {i}: {p.text.strip()}")
            
        # Check for TOC field codes
        for instr in p._p.findall('.//' + qn('w:instrText')):
            if instr.text and 'TOC' in instr.text:
                print(f"Found TOC field code at para {i}: {instr.text}")
                
    print(f"Total TOC headings found: {len(toc_paras)}\n")

check_toc(r'd:\RESEARCH PNGS\zukhruftheisis_final.docx')
check_toc(r'd:\RESEARCH PNGS\zukhruftheisis_v3.docx')
