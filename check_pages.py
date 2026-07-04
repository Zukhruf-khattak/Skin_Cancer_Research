import docx
from docx.oxml.ns import qn

def inspect_unwanted_pages(filename):
    print(f"=== Inspecting {filename} for unwanted pages/breaks ===")
    try:
        doc = docx.Document(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return
        
    empty_count = 0
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        
        # Check for page breaks
        breaks = p._p.findall('.//' + qn('w:br'))
        has_page_break = any(br.get(qn('w:type')) == 'page' for br in breaks)
        
        if not text and not has_page_break:
            empty_count += 1
            if empty_count >= 3:
                print(f"Para [{i}]: {empty_count} consecutive empty paragraphs")
        else:
            empty_count = 0
            
        if has_page_break:
            print(f"Para [{i}]: contains PAGE BREAK. text='{text}'")

        if 'TABLE OF CONTENTS' in text.upper():
            print(f"Para [{i}]: {text}")
        if 'CHAPTER I' in text.upper():
            print(f"Para [{i}]: {text}")
            
    print("\nSections (which imply page breaks usually):")
    for si, section in enumerate(doc.sections):
        print(f"Section {si}: starts with type {section.start_type}")

inspect_unwanted_pages(r'd:\RESEARCH PNGS\zukhruftheisis_final_version.docx')
