import docx
from docx.shared import Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE

def update_thesis_formatting(filepath, output_filepath):
    doc = docx.Document(filepath)
    
    black = RGBColor(0, 0, 0)
    
    # Update Styles
    for style in doc.styles:
        if style.type == WD_STYLE_TYPE.PARAGRAPH and style.name.startswith('Heading'):
            if style.name == 'Heading 1':
                if style.font:
                    style.font.size = Pt(14)
                    style.font.color.rgb = black
            else:
                if style.font:
                    style.font.size = Pt(12)
                    style.font.color.rgb = black
                    
    # Update direct formatting in paragraphs
    for para in doc.paragraphs:
        if para.style and para.style.name.startswith('Heading'):
            is_heading_1 = (para.style.name == 'Heading 1')
            target_size = Pt(14) if is_heading_1 else Pt(12)
            
            for run in para.runs:
                # Apply size and color directly to override any inline formatting
                run.font.size = target_size
                run.font.color.rgb = black
                
    doc.save(output_filepath)
    print(f"Successfully saved updated document to {output_filepath}")

if __name__ == '__main__':
    input_file = 'Thesis_Chapters_1_to_6_Corrected_Final.docx'
    output_file = 'Thesis_Chapters_1_to_6_Corrected_Final.docx'
    update_thesis_formatting(input_file, output_file)
