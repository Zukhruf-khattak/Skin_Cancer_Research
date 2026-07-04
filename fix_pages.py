import docx
from docx.oxml.ns import qn

def fix_unwanted_pages(in_file, out_file):
    doc = docx.Document(in_file)
    
    paras_to_remove = []
    
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        breaks = p._p.findall('.//' + qn('w:br'))
        has_pb = any(br.get(qn('w:type')) == 'page' for br in breaks)
        
        # Remove paras 46-49 (the empty page breaks before TOC)
        if 46 <= i <= 49:
            paras_to_remove.append((i, p, "Mess before TOC"))
            continue
            
        # Check consecutive empty lines
        if i >= 160 and i <= 163 and not text:
            paras_to_remove.append((i, p, "Empty lines before Chapter III"))
            continue
            
    print(f"Removing {len(paras_to_remove)} unwanted paragraphs")
    for i, p, reason in paras_to_remove:
        print(f"  Removing para {i}: {reason}")
        p._element.getparent().remove(p._element)
        
    doc.save(out_file)
    print(f"Saved to {out_file}")

fix_unwanted_pages(r'd:\RESEARCH PNGS\zukhruftheisis_final_version.docx', r'd:\RESEARCH PNGS\zukhruftheisis_final_version.docx')
