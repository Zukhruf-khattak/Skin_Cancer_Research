import docx
from docx.shared import Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE

def inspect_styles(filepath):
    doc = docx.Document(filepath)
    print("--- Heading Styles ---")
    for style in doc.styles:
        if style.type == WD_STYLE_TYPE.PARAGRAPH and 'Heading' in style.name:
            font = style.font
            color = font.color.rgb if font.color else None
            size = font.size.pt if font.size else None
            print(f"Style: {style.name}, Size: {size}, Color: {color}")

if __name__ == '__main__':
    inspect_styles('Thesis_Chapters_1_to_6_Corrected_Final.docx')
