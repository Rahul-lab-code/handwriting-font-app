import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract

def ocr_label_image(img_path):
    img = Image.open(img_path)
    text = pytesseract.image_to_string(
        img,
        config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    )
    text = text.strip()
    return text[0] if text else ''

def create_label_file(segmented_dir, label_file):
    segmented_dir = Path(segmented_dir)
    label_file = Path(label_file)

    if not segmented_dir.is_dir():
        raise ValueError(f"The directory {segmented_dir} does not exist.")

    labels = []
    for img_file in sorted(segmented_dir.glob('*.png')):
        ocr_label = ocr_label_image(img_file)
        if ocr_label:  # Only add if OCR found a character
            print(f"OCR label for {img_file.name}: '{ocr_label}'")
            labels.append((img_file.name, ocr_label))
        else:
            print(f"[SKIP] No character detected for {img_file.name}")

    df = pd.DataFrame(labels, columns=['filename', 'label'])
    df.to_csv(label_file, index=False)
    print(f"Labels saved to {label_file}")

def main():
    segmented_dir = 'data/segmented_characters'
    label_file = 'data/labels.csv'
    create_label_file(segmented_dir, label_file)
    print("Labeling complete.")

if __name__ == "__main__":
    main()