import zipfile
import xml.etree.ElementTree as ET
import os

def parse_docx(docx_path, output_md_path):
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    }
    
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} not found.")
        return
        
    print(f"Parsing {docx_path}...")
    
    with zipfile.ZipFile(docx_path) as docx:
        # Load main document XML
        try:
            doc_xml = docx.read('word/document.xml')
        except KeyError:
            print("Error: word/document.xml not found in zip. Is this a valid docx?")
            return
            
        root = ET.fromstring(doc_xml)
        body = root.find('w:body', namespaces)
        if body is None:
            print("Error: w:body not found in document XML.")
            return
            
        md_lines = []
        
        # Traverse child elements of body in order
        for child in body:
            tag = child.tag.split('}')[-1]
            
            if tag == 'p':
                # Parse paragraph
                p_text = []
                pPr = child.find('w:pPr', namespaces)
                
                # Check for heading style
                heading_level = 0
                if pPr is not None:
                    pStyle = pPr.find('w:pStyle', namespaces)
                    if pStyle is not None:
                        val = pStyle.get(f'{{{namespaces["w"]}}}val')
                        if val and val.startswith('Heading'):
                            try:
                                heading_level = int(val.replace('Heading', ''))
                            except ValueError:
                                heading_level = 1
                
                # Check if it is a list item
                is_list = False
                if pPr is not None:
                    numPr = pPr.find('w:numPr', namespaces)
                    if numPr is not None:
                        is_list = True
                
                # Extract text runs
                for r in child.findall('w:r', namespaces):
                    t = r.find('w:t', namespaces)
                    if t is not None and t.text:
                        p_text.append(t.text)
                
                text = "".join(p_text).strip()
                if text:
                    if heading_level > 0:
                        md_lines.append("\n" + "#" * heading_level + " " + text + "\n")
                    elif is_list:
                        md_lines.append(f"- {text}")
                    else:
                        md_lines.append(text + "\n")
                elif not is_list:
                    # Keep empty lines for spacing
                    md_lines.append("")
                    
            elif tag == 'tbl':
                # Parse table
                md_lines.append("")
                rows = []
                for row_elem in child.findall('w:tr', namespaces):
                    row_data = []
                    for cell_elem in row_elem.findall('w:tc', namespaces):
                        # Extract paragraph texts from cell
                        cell_text = []
                        for cell_p in cell_elem.findall('w:p', namespaces):
                            p_text = []
                            for r in cell_p.findall('w:r', namespaces):
                                t = r.find('w:t', namespaces)
                                if t is not None and t.text:
                                    p_text.append(t.text)
                            p_str = "".join(p_text).strip()
                            if p_str:
                                cell_text.append(p_str)
                        row_data.append(" ".join(cell_text).replace('|', '\\|'))
                    rows.append(row_data)
                
                if rows:
                    col_count = max(len(r) for r in rows)
                    # Format as markdown table
                    for i, r in enumerate(rows):
                        # Pad row with empty cells if it's shorter
                        while len(r) < col_count:
                            r.append("")
                        md_lines.append("| " + " | ".join(r) + " |")
                        if i == 0:
                            # Header separator
                            md_lines.append("| " + " | ".join(["---"] * col_count) + " |")
                md_lines.append("")
                
        # Clean up consecutive empty lines
        cleaned_lines = []
        prev_empty = False
        for line in md_lines:
            if line.strip() == "":
                if not prev_empty:
                    cleaned_lines.append("")
                    prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
                
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(cleaned_lines))
            
    print(f"Successfully converted to {output_md_path}")

if __name__ == "__main__":
    parse_docx("zukhruf proposal.docx", "zukhruf_proposal.md")
