import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_update_fields(doc_path, output_path):
    doc = docx.Document(doc_path)
    
    # 3. Modify settings.xml to update fields on open
    try:
        settings = doc.settings.element
        updateFields = settings.find(qn('w:updateFields'))
        if updateFields is None:
            updateFields = OxmlElement('w:updateFields')
            updateFields.set(qn('w:val'), 'true')
            settings.append(updateFields)
            print("Set updateFields to true in settings.")
    except Exception as e:
        print(f"Could not update settings: {e}")

    doc.save(output_path)
    print(f"Saved to {output_path}")

add_update_fields(r'd:\RESEARCH PNGS\zukhruftheisis_v4.docx', r'd:\RESEARCH PNGS\zukhruftheisis_final_version.docx')
