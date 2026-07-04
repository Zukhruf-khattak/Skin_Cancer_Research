import docx
from docx.oxml.ns import qn

def verify_pages(filename):
    doc = docx.Document(filename)
    
    count = 0
    for p in doc.paragraphs:
        hls = p._p.findall('.//' + qn('w:hyperlink'))
        for hl in hls:
            anchor = hl.get(qn('w:anchor'), '')
            if anchor.startswith('TOC_BM_'):
                text = hl.find('.//' + qn('w:t')).text
                # Find the page number which is the last w:t in the paragraph
                runs = p._p.findall('.//' + qn('w:r'))
                page = runs[-1].findall('.//' + qn('w:t'))[-1].text
                print(f"{text[:30]} -> Page {page}")
                count += 1
                if count > 5 and count < 95:
                    continue
                    
verify_pages(r'd:\RESEARCH PNGS\zukhruftheisis_FINAL_TOC_Numbered.docx')
