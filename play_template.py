import retro
import pygame
import numpy as np
import os
import time
from PIL import Image
import argparse
import csv
from shared import game_state_dict,crop_black,crop_black_gray
import cv2 as cv
RIGHT_STEPS=10

if __name__=='__main__':
    os.makedirs("threshold",exist_ok=True)
    os.makedirs("components",exist_ok=True)

    states={
        'SonicTheHedgehog2-Genesis':['EmeraldHillZone.Act1','AquaticRuinZone.Act1','CasinoNightZone.Act1','HillTopZone.Act1'],
        'SuperMarioWorld-Snes':["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'],
        'CastlevaniaBloodlines-Genesis-v0':['Level1-1','Level2-1','Level3-1','Level4-1']
    }

    state_help=""
    for k,v in states.items():
        state_help+=f"{k} has states: "+",".join(v)+"\n"

    PATH="templates_diff"
    os.makedirs(PATH,exist_ok=True)

    # ---- SETTINGS ----


    # Create environment with recording enabled
    

    # ---- Init pygame ----
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption(" Controller - Focus this window to play!")

    clock = pygame.time.Clock()

    done = False

    # Recorded list of full button vectors
    action_log = []

    print("Controls:")
    print(" AWASD = Move")
    print(" Q = Buttons (jump/spin)")
    print(" Enter = Start")
    print(" ESC = Quit")
    print("------------------")
    
    for game ,state_list in game_state_dict.items():
        os.makedirs(os.path.join("threshold",f"{game}"),exist_ok=True)
        
        for button_name in ["B","UP","RIGHT","LEFT","DOWN"]:
            thresh_list=[]
            obs_list=[]
            component_list=[]
            
            for state in state_list:   
                

                env = retro.make(
                game=game,
                state=state,
                #record=save_dir,
                use_restricted_actions=retro.Actions.ALL
                )
                env.reset()
                
                


                # ---- BUTTON MAPPING ----
                # retro uses a fixed order of buttons depending on the game:
                # Example Genesis mapping: ['B','NULL','C','A','Start','Up','Down','Left','Right']
                BUTTONS = env.buttons
                print(BUTTONS)
                print(env.unwrapped.buttons)

                # Keyboard → Genesis button
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

                action = np.zeros(len(BUTTONS), dtype=np.int8)
                obs, rew, terminated, truncated, info=env.step(action)



                env.reset()
                save_parent=os.path.join(PATH,game,state)
                os.makedirs(save_parent, exist_ok=True)

                

                action = np.zeros(len(BUTTONS), dtype=np.int8)
                for _ in range(100):
                    env.step(action)
                    env.render()
                    #clock.tick(60)

                idx = button_index.get("RIGHT", None)
                if idx is not None:
                    action[idx] = 1
                for n in range(5):
                    before_obs, rew, terminated, truncated, info=env.step(action)
                action = np.zeros(len(BUTTONS), dtype=np.int8)
                before_obs, rew, terminated, truncated, info=env.step(action)

                before_obs=cv.resize(before_obs, (256,256))
                action = np.zeros(len(BUTTONS), dtype=np.int8)


                
                for count in range(3):
                    action = np.zeros(len(BUTTONS), dtype=np.int8)


                    pressed_list=[]

                    pressed_list.append(button_name)
                    idx = button_index.get(button_name, None)
                    if idx is not None:
                        action[idx] = 1
                    print(action,button_name,idx)



                    # Step emulator
                    
                    obs, rew, terminated, truncated, info=env.step(action)
                    obs=cv.resize(obs,(256,256))

                env.close()
                diff = cv.absdiff(before_obs.copy(),obs.copy())
                gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
                _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
                image = Image.fromarray(thresh) #.resize((256,256))
                save_path_image=os.path.join(save_parent,f"{button_name}.jpg")
                #image.save(save_path_image)
                thresh_list.append(thresh)
                obs_list.append(obs)
                n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
                component_areas = stats[1:, cv.CC_STAT_AREA]  # skip backgroun

                # Get indices of the two largest components
                largest_indices = component_areas.argsort()[-2:][::-1] + 1 
                background_mask = np.isin(labels, largest_indices)
                con_components=cv.bitwise_and(obs,obs,mask=background_mask.astype(np.uint8))
                component_list.append(con_components)
                Image.fromarray(con_components).save(os.path.join("components",f"{game}_{state}_{button_name}.jpg"))

                env.close()
            n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
            # Ignore background (label 0) and get component areas
            component_areas = stats[1:, cv.CC_STAT_AREA]  # skip background
            if len(component_areas) == 0:
                raise ValueError("No components found")

            # Get indices of the two largest components
            largest_indices = component_areas.argsort()[-2:][::-1] + 1  # +1 because we skipped background

            # Store masked images of these components
            best_label=largest_indices[0]
            best_max=0.0
            for label in largest_indices:
                component_mask = (labels == label).astype(np.uint8)
                con_components = cv.bitwise_and(obs, obs, mask=component_mask)
                #component_list.append(con_components)
                
                # Save component for inspection
                os.makedirs(os.path.join("components", game), exist_ok=True)
                Image.fromarray(con_components).save(
                    os.path.join("components", game, f"{button_name}_{label}.jpg")
                )


                component_mask = (labels == label)
                #print(np.mean(thresh))
                extracted=cv.bitwise_and(obs,obs,mask=component_mask.astype(np.uint8))
                #print(extracted.shape)
                try:
                    template=crop_black(extracted)
                    h,w=template.shape[:2]
                    max_sum=0
                    for j,comp_j in enumerate(component_list[:-1]):
                        res = cv.matchTemplate(comp_j,template,cv.TM_SQDIFF_NORMED)
                        min_val, _, min_loc, max_loc = cv.minMaxLoc(res)
                        max_val=1-min_val
                        max_sum+=max_val
                        top_left = min_loc
                        bottom_right = (top_left[0] + w, top_left[1] + h)
                        rect=cv.rectangle(comp_j.copy(),top_left, bottom_right, 255, 2)
                        Image.fromarray(rect).save(os.path.join("threshold",f"{game}",f"{button_name}_{label}_{j}.jpg"))
                        print(game,button_name,label,j, max_sum)
                    '''component_mask = (labels == best_label).resize((256,256))
                    extracted=cv.bitwise_and(obs,obs,mask=component_mask.astype(np.uint8))
                    Image.fromarray(extracted).save(os.path.join("threshold",f"{game}_{button_name}_{label}.jpg"))'''
                    if max_sum>best_max:
                        print(f"{best_max} -> {max_sum} {best_label}-> {label}")
                        best_max=max_sum
                        best_label=label
                        
                except ValueError:
                    pass
            component_mask = (labels == best_label)
            extracted=cv.bitwise_and(obs,obs,mask=component_mask.astype(np.uint8))
            Image.fromarray(extracted).save(os.path.join("threshold",f"{game}_{button_name}.jpg"))
            Image.fromarray(crop_black(extracted)).save(os.path.join("sprite",f"{game}_{button_name}.jpg"))
        



            
