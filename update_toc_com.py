import os
import sys

def update_toc_with_word(doc_path):
    import win32com.client
    print(f"Opening Word to update TOC for: {doc_path}")
    
    # Try to start Word
    try:
        word = win32com.client.Dispatch("Word.Application")
    except Exception as e:
        print(f"Failed to start Word Application: {e}")
        return False
        
    word.Visible = False
    
    try:
        abs_path = os.path.abspath(doc_path)
        doc = word.Documents.Open(abs_path)
        print("Document opened successfully.")
        
        # Update fields (including TOC)
        for field in doc.Fields:
            field.Update()
            
        # specifically update TOCs just in case
        for toc in doc.TablesOfContents:
            toc.Update()
            
        print("TOC and fields updated.")
        
        doc.Save()
        doc.Close()
        print("Document saved and closed.")
        
    except Exception as e:
        print(f"Error while updating document: {e}")
    finally:
        word.Quit()
        
    return True

if __name__ == '__main__':
    # First, let's make a copy of cleaned that has the native TOC
    import shutil
    shutil.copy('zukhruftheisis_final_version.docx', 'zukhruftheisis_Final_Native_TOC.docx')
    update_toc_with_word('zukhruftheisis_Final_Native_TOC.docx')
