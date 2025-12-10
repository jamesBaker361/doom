import retro
import pygame
import numpy as np
import os
import time
from PIL import Image
import argparse
import csv
import cv2 as cv
from shared import pil_grid,crop_black

rects=[
    [(10,10),(100,100)],
    [(10,10),(100,100)],
    [(10,10),(100,100)],
]

moving_rects=[
        [(10,10),(100,100)],
    [(10,10),(100,100)],
    [(10,10),(100,100)],
]
games=['SonicTheHedgehog2-Genesis','SuperMarioWorld-Snes','CastlevaniaBloodlines-Genesis']



for game,rect,move_rect in zip(games,rects,moving_rects):
    env = retro.make(
            game=game,
            #state=args.state,
            #record=save_dir,
            use_restricted_actions=retro.Actions.ALL
        )
    BUTTONS = env.buttons
    action = np.zeros(len(BUTTONS), dtype=np.int8)
    
    env.reset()
    still_obs, rew, terminated, truncated, info=env.step(action)
    #still_obs=cv.rectangle(still_obs,rect[0],rect[1],color=1,thickness=5)
    still_image = Image.fromarray(still_obs).resize((256,256))
    still_image.save(f"None_{game[:10]}.jpg")
    env.reset()

    KEY_TO_BUTTON = {
        pygame.K_d: "RIGHT",
        pygame.K_a: "LEFT",
        pygame.K_w: "UP",
        pygame.K_s: "DOWN",

        pygame.K_q: "B",      # Jump
        #pygame.K_x: "LEFT",
        #.K_c: "C",

        pygame.K_RETURN: "Start",
    }

    # Reverse lookup: button name → index

    button_index = {b: i for i, b in enumerate(BUTTONS)}

    image_list=[]

    for button_name in ["RIGHT","LEFT","B"]:
        action = np.zeros(len(BUTTONS), dtype=np.int8)
        for _ in range(100):
            obs, rew, terminated, truncated, info=env.step(action)
        #for key, button_name in KEY_TO_BUTTON.items():
        idx = button_index.get(button_name, None)
        if idx is not None:
            action[idx] = 1

        for _ in range(7):
            obs, rew, terminated, truncated, info=env.step(action)
        #obs==cv.rectangle(obs,rect[0],rect[1],color=1,thickness=5)

        image=Image.fromarray(obs).resize((256,256))
        image.save(f"{button_name}_{game[:10]}.jpg")
\
        env.reset()
    env.close()