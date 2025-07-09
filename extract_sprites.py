import cv2
import numpy as np
from PIL import Image
import os

limit=10


import cv2
import numpy as np
from PIL import Image
import os

def extract_sprites_from_colored_sheet(image_path, output_dir, color_tol=30, min_area=100):
    os.makedirs(output_dir, exist_ok=True)

    # Load image in RGB
    sheet = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Step 1: Estimate background color (top-left pixel is usually safe)
    bg_color = sheet[0, 0]
    print(f"Detected background color: {bg_color}")

    # Step 2: Build a mask: keep pixels that are *not close* to background color
    diff = np.abs(sheet.astype(int) - bg_color.astype(int))
    mask = np.any(diff > color_tol, axis=2).astype(np.uint8) * 255

    # Optional: close gaps and remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Step 3: Find sprite contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Extract and save each sprite
    count = 0
    for cnt in contours:
        if count==limit:
            break
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue

        sprite_rgb = sheet[y:y+h, x:x+w]
        sprite_mask = mask[y:y+h, x:x+w]

        # Add alpha channel using mask
        alpha = sprite_mask
        sprite_rgba = np.dstack([sprite_rgb, alpha])

        out_img = Image.fromarray(sprite_rgba)
        out_img.save(os.path.join(output_dir, f"sprite_{count:03d}.png"))
        count += 1

    print(f"✅ Extracted {count} sprites to '{output_dir}'")

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
        if i==limit:
            break
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue  # skip small dots

        sprite = sheet[y:y+h, x:x+w]

        # Save each sprite
        sprite_img = Image.fromarray(cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA))
        sprite_img.save(os.path.join(output_dir, f"sprite_{count:03d}.png"))
        count += 1

    print(f"✅ Extracted {count} sprites to '{output_dir}'")

for character in ["eric_lecarde","john_morris","mario","megaman","tails","sonic"]:
    output_dir=os.path.join("sprites",character)
    os.makedirs(output_dir,exist_ok=True)
    extract_sprites_from_colored_sheet(f"{character}.png",output_dir)