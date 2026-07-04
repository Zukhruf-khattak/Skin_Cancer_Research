import os

def update_toc_with_word(doc_path):
    import win32com.client
    print(f"Opening Word to update TOC for: {doc_path}")
    
    # Try to start Word
    try:
        word = win32com.client.DispatchEx("Word.Application")
    except Exception as e:
        print(f"Failed to start Word Application: {e}")
        return False
        
    word.Visible = False
    word.DisplayAlerts = False  # Suppress popups
    
    try:
        abs_path = os.path.abspath(doc_path)
        doc = word.Documents.Open(abs_path)
        print("Document opened successfully.")
        
        # Update specifically TOCs
        for toc in doc.TablesOfContents:
            toc.Update()
            
        print("TOC updated.")
        
        doc.Save()
        doc.Close()
        print("Document saved and closed.")
        
    except Exception as e:
        print(f"Error while updating document: {e}")
    finally:
        word.Quit()
        
    return True

if __name__ == '__main__':
    update_toc_with_word('zukhruftheisis_Final_Native_TOC.docx')
