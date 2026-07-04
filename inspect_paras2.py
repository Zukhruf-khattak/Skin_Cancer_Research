import docx
from docx.oxml.ns import qn

def inspect_chapter1_paras(filename):
    doc = docx.Document(filename)
    
    print("Paras 149 to 153:")
    for i in range(149, 154):
        p = doc.paragraphs[i]
        text = p.text.strip()
        breaks = p._p.findall('.//' + qn('w:br'))
        rendered = p._p.findall('.//' + qn('w:lastRenderedPageBreak'))
        
        b_types = [br.get(qn('w:type'), 'textWrapping') for br in breaks]
        
        sect = False
        pPr = p._p.find(qn('w:pPr'))
        if pPr is not None and pPr.find(qn('w:sectPr')) is not None:
            sect = True
            
        print(f"Para {i}: txt='{text[:30]}' | br={b_types} | rendered={len(rendered)} | sect={sect}")

inspect_chapter1_paras(r'd:\RESEARCH PNGS\zukhruftheisis_FINAL_TOC.docx')
