import json
import os

def extract_notebook_code(notebook_path, output_py_path):
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return
        
    print(f"Loading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    cell_count = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell_count += 1
            cell_source = cell.get('source', [])
            if cell_source:
                code_cells.append(f"# === CODE CELL {cell_count} ===")
                code_cells.append("".join(cell_source))
                code_cells.append("\n\n")
                
    with open(output_py_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(code_cells))
        
    print(f"Extracted {cell_count} code cells to {output_py_path}")

if __name__ == "__main__":
    extract_notebook_code("1-stage/stage-1-ipynb.ipynb", "1-stage/extracted_stage1.py")
