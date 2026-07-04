import docx
from docx.oxml.ns import qn

def detail_headings(filename):
    doc = docx.Document(filename)
    
    for i, p in enumerate(doc.paragraphs):
        for bk in p._p.findall('.//' + qn('w:bookmarkStart')):
            name = bk.get(qn('w:name'), '')
            if name.startswith('TOC_BM_'):
                print(f"Para {i}: Heading '{p.text.strip()[:30]}' - {name}")
                break

detail_headings(r'd:\RESEARCH PNGS\zukhruftheisis_FINAL_TOC.docx')
