from datasets import Dataset,load_dataset
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

interval=15
limit=5

for game,states_list in game_state_dict.items():

    #print(len(template_list))
    for state in states_list[::-1]:
        
        repo=f"jlbaker361/{game}_{state}_{interval}_{limit}_coords"
        if api.repo_exists(repo,repo_type="dataset"):
            print(repo,"exists")
            data=load_dataset(repo,split="train",download_mode="force_redownload",)
            for row in data:
                print(row)