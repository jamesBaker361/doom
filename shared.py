from PIL import Image
import numpy as np
import cv2
PATH="videos"
def pil_grid(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    return new_im

def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(np.max(gray),np.min(gray))
    coords = np.column_stack(np.where(gray > 10))  # any nonzero pixel

    if coords.size == 0:
        raise ValueError("Image is entirely black.")

    # Get bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img[y0:y1+1, x0:x1+1]

    return cropped