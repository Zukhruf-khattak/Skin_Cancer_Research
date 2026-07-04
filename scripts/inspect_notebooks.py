import json
import os
import re

def analyze_notebook(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    print(f"\nAnalyzing {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    text_content = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            code_cells.append("".join(cell.get('source', [])))
        elif cell.get('cell_type') == 'markdown':
            text_content.append("".join(cell.get('source', [])))
            
    # Look for model classes or instantiations
    print("--- Model References ---")
    model_patterns = [
        r'models\.resnet\d+', r'models\.mobilenet_v\d+', r'models\.densenet\d+', 
        r'timm\.create_model\([\'"][a-zA-Z0-9_-]+[\'"]', r'models\.vgg\d+', r'models\.inception\d+',
        r'class \w+Net', r'class \w+Model'
    ]
    
    found_models = set()
    for code in code_cells:
        for pattern in model_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                found_models.add(match)
                
    if found_models:
        print("Found model references:")
        for m in sorted(found_models):
            print(f"  - {m}")
    else:
        print("No model references found.")
        
    # Search for training results or summaries
    print("\n--- Training / Validation / Test Keywords ---")
    keywords = ['Accuracy:', 'Recall:', 'F1-Score:', 'Precision:', 'classification_report', 'Confusion Matrix']
    results_found = []
    for code in code_cells:
        for kw in keywords:
            if kw in code:
                # Look for prints or comments
                lines = code.split('\n')
                for line in lines:
                    if kw in line and (line.strip().startswith('#') or 'print(' in line):
                        results_found.append(line.strip())
                        
    for line in results_found[:15]:
        print(f"  {line}")
        
    if len(results_found) > 15:
        print(f"  ... and {len(results_found) - 15} more matches.")

if __name__ == "__main__":
    analyze_notebook("stage-2-skin-cancer (1).ipynb")
    analyze_notebook("stage-2b-skin-cancer.ipynb")
    analyze_notebook("skin-cancer-stage-1.ipynb")
