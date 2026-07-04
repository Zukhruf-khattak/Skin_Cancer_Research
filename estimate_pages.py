import docx
from docx.oxml.ns import qn

def estimate_page_numbers(filename):
    doc = docx.Document(filename)
    
    current_page = 1
    page_numbers = {}
    
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        
        # Check for explicit page breaks
        breaks = p._p.findall('.//' + qn('w:br'))
        for br in breaks:
            if br.get(qn('w:type')) == 'page':
                current_page += 1
                
        # Check for rendered page breaks
        rendered_breaks = p._p.findall('.//' + qn('w:lastRenderedPageBreak'))
        if rendered_breaks:
            current_page += len(rendered_breaks)
            
        # Check section breaks (often mean new page)
        sect_break = False
        pPr = p._p.find(qn('w:pPr'))
        if pPr is not None and pPr.find(qn('w:sectPr')) is not None:
            current_page += 1
            
        if text and (text.upper().startswith('CHAPTER') or '1.1 Background' in text or '2.1' in text):
            page_numbers[text[:30]] = current_page
            print(f"Para [{i}] Page ~{current_page}: {text[:40]}")
            
    print(f"Total pages estimated: {current_page}")

estimate_page_numbers(r'd:\RESEARCH PNGS\zukhruftheisis_cleaned.docx')
