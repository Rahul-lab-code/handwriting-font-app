# filepath: handwriting-font-app/handwriting-font-app/src/fontforge_script.py

import fontforge
import os

def create_font_from_svg(svg_directory, output_font_path):
    # Create a new font
    font = fontforge.font()
    font.fontname = "CustomHandwriting"
    font.familyname = "Custom Handwriting"
    font.fullname = "Custom Handwriting Font"
    
    # Iterate through SVG files in the specified directory
    for svg_file in os.listdir(svg_directory):
        if svg_file.endswith('.svg'):
            # Get the character from the filename (assuming the format is char_x.svg)
            char_name = svg_file.split('.')[0]  # e.g., char_a
            unicode_value = ord(char_name[-1])  # Get the unicode value of the character
            
            # Create a new glyph in the font
            glyph = font.createChar(unicode_value, char_name)
            glyph.importOutlines(os.path.join(svg_directory, svg_file))
            glyph.correctDirection()  # Ensure the glyph direction is correct

    # Generate the font file
    font.generate(output_font_path)
    print(f"[INFO] Font generated and saved to {output_font_path}")

if __name__ == "__main__":
    svg_directory = "data/generated_characters"  # Path to the directory containing SVG files
    output_font_path = "data/fonts/CustomHandwriting.ttf"  # Path to save the generated font file
    create_font_from_svg(svg_directory, output_font_path)