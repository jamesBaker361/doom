from datasets import Dataset
import csv
import os
from shared import game_state_dict,game_key_dict
import cv2 as cv
from PIL import Image
import numpy as np

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

CUTOFF=0.1

button_list=["B","LEFT","RIGHT","DOWN","UP"]
for game,states_list in game_state_dict.items():
    key_list=game_key_dict[game]
    output_dict={
        key:[] for key in key_list+["image","state","game","episode"]
    }
    '''template_list=[
        cv.cvtColor(cv.imread( os.path.join("sprite_from_sheet",f"{game}", f"{n}.jpg") ,cv.IMREAD_COLOR),cv.COLOR_BGR2RGB) for n in 
        range(1,len([f for f in os.listdir(os.path.join("sprite_from_sheet",f"{game}")) 
                     if f.endswith("jpg") ] ) )
    ]
    mask_list=[
        mask_black_with_neighbors(template) for template in template_list
    ]

    template_list=[
        cv.cvtColor(cv.imread( os.path.join("sprite_from_sheet",f"{game}", f"{button}.jpg") ,cv.IMREAD_COLOR),cv.COLOR_BGR2RGB) for button in button_list
    ]
    mask_list=[
        mask_black_with_neighbors(template) for template in template_list
    ]'''
    #print(len(template_list))
    for state in states_list[::-1]:
        count=0
        directory=os.path.join("videos",game,state)
        if os.path.exists(directory):
            for episode in os.listdir(directory)[:2]:
                #index=0
                subdir=os.path.join(directory,episode)
                csv_path=os.path.join(subdir,'data.csv')
                with open(csv_path,"r") as file:
                    reader=csv.DictReader(file)
                    for row in reader:

                        for key,value in row.items():
                            output_dict[key].append(value)
                        cv_image=cv.cvtColor(cv.imread(row["save_path"],cv.IMREAD_COLOR),cv.COLOR_BGR2RGB)
                        output_dict["state"].append(state)
                        output_dict["game"].append(game)
                        output_dict["image"].append(Image.fromarray(cv_image))
                        output_dict["episode"].append(episode)
                        #index+=1
                        count+=1
                        '''if count==10:
                            Dataset.from_dict(output_dict).push_to_hub(f"jlbaker361/{game}_{state}_10")
                            print("pushed")
                            break'''

                        if count==1000:
                            Dataset.from_dict(output_dict).push_to_hub(f"jlbaker361/{game}_{state}_100")
                            print("pushed")
                            break