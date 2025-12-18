from datasets import load_dataset,Dataset
from shared import all_tokens,game_state_dict
import os
import csv
import cv2 as cv
from PIL import Image
import random
import numpy as np
from data_loaders import NONE_STRING

def mask_black_with_neighbors_pil(img_pil, thresh=10, min_neighbors=5):
    """
    Takes a PIL RGB image, computes a mask of pixels that are NOT
    black-with-many-black-neighbors. Returns a NumPy uint8 mask (1 = keep, 0 = masked).
    """
    # Convert PIL → numpy (RGB) → BGR for OpenCV
    img = np.array(img_pil)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    black = (gray < thresh).astype(np.uint8)

    # Kernel to count 8 neighbors (exclude center)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    neighbor_count = cv.filter2D(black, -1, kernel)

    # Select pixels: black AND has enough black neighbors
    mask = (black & (neighbor_count >= min_neighbors)).astype(np.uint8)

    # Final mask: 1 - mask, scaled to [0,255]
    mask_out = (1 - mask) * 255

    # Convert back to PIL grayscale
    return Image.fromarray(mask_out, mode="L")

def paste_random_small_image( small:Image.Image,bg_size=(256, 256),mask=None):
    """
    Creates a bg_size image with a random RGB color and pastes
    the image at small_path at a random valid location.
    Returns the final PIL Image object.
    """
    # 1. Random background color
    '''bg_color = tuple(random.randint(0, 255) for _ in range(3))
    bg = Image.new("RGB", bg_size, bg_color)'''
    noise = np.random.randint(0, 256, (bg_size[1], bg_size[0], 3), dtype=np.uint8)
    bg = Image.fromarray(noise, mode="RGB")

    sw, sh = small.size
    bw, bh = bg_size

    # 3. Random valid location (fully inside)
    x = random.randint(0, bw - sw)
    y = random.randint(0, bh - sh)

    if mask is not None:
        bg.paste(small, (x, y),mask)
    else:
        # 4. Paste with no mask
        bg.paste(small, (x, y))

    return bg


'''small=Image.open(os.path.join("distorted_sprite_from_sheet","CastlevaniaBloodlines-Genesis","9.jpg"))
mask=mask_black_with_neighbors_pil(small)
paste_random_small_image(small,mask=mask).save("tiny.jpg")
exit(0)'''

output_dict={
    "image":[],
    "game":[],
    "state":[]
}

interval=15

reps=3

for game,state_list in game_state_dict.items():
    for state in state_list:
        directory=os.path.join("videos",str(interval),game,state)

        for episode in os.listdir(directory):                    
            #index=0
            subdir=os.path.join(directory,episode)
            csv_path=os.path.join(subdir,'data.csv')
            with open(csv_path,"r") as file:
                reader=csv.DictReader(file)
                
                for i,row in enumerate(reader):
                    
                    
                    cv_image=cv.cvtColor(cv.imread(row["save_path"],cv.IMREAD_COLOR),cv.COLOR_BGR2RGB)
                    output_dict["state"].append(state)
                    output_dict["game"].append(game)
                    output_dict["image"].append(Image.fromarray(cv_image))

    sprite_directory=os.path.join("distorted_sprite_from_sheet",game)
    for file in os.listdir(sprite_directory):
        if file.endswith("jpg"):
            cv_image=cv.cvtColor(cv.imread(row["save_path"],cv.IMREAD_COLOR),cv.COLOR_BGR2RGB)
            pil_image=Image.fromarray(cv_image)
            mask=mask_black_with_neighbors_pil(pil_image)
            for _ in range(reps):
                pasted=paste_random_small_image(pil_image,mask=mask)
                output_dict["image"].append(pasted)
                output_dict["game"].append(game)
                output_dict["state"].append(NONE_STRING)

                    
Dataset.from_dict(output_dict).push_to_hub(f"jlbaker361/classification-ivg-reps-{reps}")