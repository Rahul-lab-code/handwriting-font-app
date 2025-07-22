import os
import subprocess
from pathlib import Path
from PIL import Image

def convert_images_to_svg(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_file in input_dir.glob('*.png'):
        # Convert PNG to PBM (temporary file)
        pbm_file = image_file.with_suffix('.pbm')
        with Image.open(image_file) as img:
            img = img.convert('1')  # Convert to black and white
            img.save(pbm_file)

        svg_file = output_dir / f"{image_file.stem}.svg"
        command = f"potrace {pbm_file} -s -o {svg_file}"
        subprocess.run(command, shell=True)
        print(f"[INFO] Converted {image_file.name} to {svg_file.name}")

        # Remove the temporary PBM file
        pbm_file.unlink()

if __name__ == "__main__":
    input_directory = 'data/generated_characters'
    output_directory = 'data/fonts/svg'
    convert_images_to_svg(input_directory, output_directory)