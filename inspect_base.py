import docx
from docx.oxml.ns import qn

def inspect_base_toc():
    doc = docx.Document(r'd:\RESEARCH PNGS\zukhruftheisis_final.docx')
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip().upper()
        if 'CONTENTS' in text or 'TABLE OF' in text:
            print(f"[{i}] {p.text.strip()}")
            
inspect_base_toc()
