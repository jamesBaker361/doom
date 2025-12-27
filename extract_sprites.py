from datasets import Dataset
import csv
import os
from shared import game_state_dict,game_key_dict,SONIC_1GAME,SONIC_GAME,MARIO_GAME,CASTLE_GAME
import cv2 as cv
from PIL import Image
import numpy as np
from huggingface_hub import HfApi
from typing import Tuple
import torch
api=HfApi()


black_list_dict={
    SONIC_GAME:[],
    SONIC_1GAME:[],
        MARIO_GAME:[], #["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'],
        CASTLE_GAME :["107.jpg"],

}

def get_sprite_match(background:Image.Image,sprite_dir:str,)-> Tuple[bool, np.ndarray]:
    background_array=np.asarray(background)
    #print(background_array.shape)
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

