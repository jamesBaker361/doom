import cv2
import numpy as np
from PIL import Image
import os

def extract_sprites(sprite_sheet_path, output_dir, min_area=100):
    os.makedirs(output_dir, exist_ok=True)

    # Load image with alpha channel if present
    sheet = cv2.imread(sprite_sheet_path, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale (ignore alpha)
    if sheet.shape[2] == 4:
        gray = cv2.cvtColor(sheet, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate non-background pixels
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find contours (sprite regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue  # skip small dots

        sprite = sheet[y:y+h, x:x+w]

        # Save each sprite
        sprite_img = Image.fromarray(cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA))
        sprite_img.save(os.path.join(output_dir, f"sprite_{count:03d}.png"))
        count += 1

    print(f"✅ Extracted {count} sprites to '{output_dir}'")

extract_sprites("eric_lecards.png","eric")