import docx
from docx.oxml.ns import qn

def inspect_cleaned():
    doc = docx.Document(r'd:\RESEARCH PNGS\zukhruftheisis_cleaned.docx')
    print("=== First 70 paras ===")
    for i, p in enumerate(doc.paragraphs[:70]):
        text = p.text.strip()
        has_fldChar = False
        fld_instr = ""
        for instr in p._p.findall('.//' + qn('w:instrText')):
            has_fldChar = True
            fld_instr += instr.text
            
        hls = p._p.findall('.//' + qn('w:hyperlink'))
        
        if text or has_fldChar or hls:
            print(f"[{i}] text='{text[:50]}' | fld='{fld_instr}' | hls={len(hls)}")
            
inspect_cleaned()
