# filepath: handwriting-font-app/handwriting-font-app/src/char_segmentation.py

import cv2
from pathlib import Path

DATA_DIR = Path('data/raw_pages')
OUTPUT_DIR = Path('data/segmented_characters')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_characters_from_image(image_path, output_dir):
    print(f"[INFO] Processing {image_path.name}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Found {len(contours)} contours")
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours = [cnt for _, cnt in sorted(zip(bounding_boxes, contours), key=lambda b: (b[0][1], b[0][0]))]
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100:
            continue
        char_img = img[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (64, 64))
        cv2.imwrite(str(output_dir / f'char_{count}.png'), char_img)
        count += 1
    print(f"[INFO] Extracted {count} characters from {image_path.name}")

def process_all_images():
    files = list(DATA_DIR.glob('*.jpg'))
    print(f"[DEBUG] Looking for .jpg files in: {DATA_DIR.resolve()}")
    print(f"[DEBUG] Found {len(files)} .jpg files: {[f.name for f in files]}")
    for img_file in files:
        extract_characters_from_image(img_file, OUTPUT_DIR)

if __name__ == "__main__":
    process_all_images()