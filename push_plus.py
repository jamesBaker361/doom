from datasets import Dataset
import csv
import os
from shared import game_state_dict,game_key_dict
import cv2 as cv
from PIL import Image
import numpy as np
from huggingface_hub import HfApi
from typing import Tuple
import torch
api=HfApi()

os.makedirs("testing_save",exist_ok=True)

def mask_black_with_neighbors(img, thresh=10, min_neighbors=5):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    black = (gray < thresh).astype(np.uint8)

    # Kernel to count 8 neighbors (center excluded)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    neighbor_count = cv.filter2D(black, -1, kernel)

    # Pixel is selected if black AND neighbor_count >= min_neighbors
    mask = (black & (neighbor_count >= min_neighbors)).astype(np.uint8)

    return 1-mask

CUTOFF=0.01

black_list_dict={
    'SonicTheHedgehog2-Genesis':[f"{s}.jpg" for s in range(61,67)]+[f"{s}.jpg" for s in range(158,163)]+
    ["133.jpg","134.jpg","176.jpg","13.jpg"]+[f"{s}.jpg" for s in range(117,129)]+[f"{s}.jpg" for s in range(100,106)]+
    [f"{s}.jpg" for s in range(45,48)], #['EmeraldHillZone.Act1','AquaticRuinZone.Act1','CasinoNightZone.Act1','HillTopZone.Act1'],
        'SuperMarioWorld-Snes':[], #["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'],
        'CastlevaniaBloodlines-Genesis': ["107.jpg"],

}

def get_sprite_match(background:Image.Image,sprite_dir:str)-> Tuple[bool, np.ndarray]:
    background_array=np.asarray(background)
    #print(background_array.shape)
    sprite_dir=os.path.join("sprite_from_sheet",game)
    ranking=[]
    for x,sprites_jpg in enumerate(os.listdir(sprite_dir)):
        if sprites_jpg.endswith("jpg"):
            sprite_path=os.path.join(sprite_dir,sprites_jpg)
            sprite=Image.open(sprite_path).convert("RGB")
            sprite_array=np.asarray(sprite)
            mask = np.all(sprite_array != 0, axis=-1).astype(np.uint8)
            max_scores=[]
            loc_list=[]
            for d in range(3):
                back=background_array[:,:,d]
                spr=sprite_array[:,:,d]
                try:
                    res=cv.matchTemplate(back,spr,cv.TM_SQDIFF_NORMED,mask=mask)
                except Exception:
                    print(sprite_path,sprite_array.shape)
                    raise Exception("sdjf")
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                max_scores.append(min_val)
                loc_list.append(min_loc)

            if loc_list[0]== loc_list[1] and loc_list[1]==loc_list[2]: # or loc_list[0]==loc_list[2]:
                ranking.append([np.mean(max_scores),
                                sprites_jpg,
                                loc_list,
                                mask,
                                sprite_array])

    if len(ranking)==0:
        return False,torch.ones_like(background_array)
    else:
        ranking.sort(key=lambda x: x[0])
        [score, path,loc_list,mask,sprite_array]=ranking[0]
        top_left = loc_list[0]
        height,width = sprite_array.shape[:2]

        # copy background
        overlay = np.zeros_like(background_array)

        # draw white where the sprite matches (respecting the mask)
        y, x = top_left[1], top_left[0]

        print(mask.shape,overlay.shape,sprite_array.shape)

        for c in range(3):
            for h in range(height):
                for w in range(width):
                    #if mask[h][w]:
                    overlay[y+h][x+w][c]=255
        return True,overlay


interval=15
limit=5
button_list=["B","LEFT","RIGHT","DOWN","UP"]
for game,states_list in game_state_dict.items():
    sprite_dir=os.path.join("sprite_from_sheet",game)

    key_list=game_key_dict[game]
    output_dict={
        key:[] for key in key_list+["image","state","game","episode","step","overlay","use_overlay"] 
    }

    #print(len(template_list))
    for state in states_list[::-1]:
        
        directory=os.path.join("new_vid",str(interval),game,state)
        repo=f"jlbaker361/{game}_{state}_{interval}_{limit}_coords"
        if os.path.exists(directory):
            if api.repo_exists(repo):
                print(repo,"exists")
                continue
            else:
                count=0
                for episode in os.listdir(directory):

                    
                    #index=0
                    subdir=os.path.join(directory,episode)
                    csv_path=os.path.join(subdir,'data.csv')
                    with open(csv_path,"r") as file:
                        reader=csv.DictReader(file)
                        
                        for row in reader:
                            if count>=limit:
                                Dataset.from_dict(output_dict).push_to_hub(repo)
                                print(f"pushed {repo}")
                                break
                            output_dict["step"].append(count)
                            count+=1
                            for key,value in row.items():
                                output_dict[key].append(value)
                            cv_image=cv.cvtColor(cv.imread(row["save_path"],cv.IMREAD_COLOR),cv.COLOR_BGR2RGB)
                            pil_image=Image.fromarray(cv_image)
                            output_dict["state"].append(state)
                            output_dict["game"].append(game)
                            output_dict["image"].append(pil_image)
                            output_dict["episode"].append(episode)
                            use_overlay,overlay=get_sprite_match(pil_image, sprite_dir)
                            output_dict["overlay"].append(overlay)
                            output_dict["use_overlay"].append(use_overlay)

                            #index+=1
                            
                Dataset.from_dict(output_dict).push_to_hub(repo)
                print(f"pushed {repo}")
        else:
            print(directory, "does not exist")                    