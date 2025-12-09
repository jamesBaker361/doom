import retro
import pygame
import numpy as np
import os
import time
from PIL import Image
import argparse
import csv

RIGHT_STEPS=10

if __name__=='__main__':

    states={
        'SonicTheHedgehog2-Genesis':['EmeraldHillZone.Act1','AquaticRuinZone.Act1','CasinoNightZone.Act1','HillTopZone.Act1'],
        'SuperMarioWorld-Snes':["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'],
        'CastlevaniaBloodlines-Genesis-v0':['Level1-1','Level2-1','Level3-1','Level4-1']
    }

    state_help=""
    for k,v in states.items():
        state_help+=f"{k} has states: "+",".join(v)+"\n"

    PATH="videos_mini"
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
    
    for game in ['SonicTheHedgehog2-Genesis','SuperMarioWorld-Snes','CastlevaniaBloodlines-Genesis']:
        env = retro.make(
        game=game,
        #state=args.state,
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
        
        

        for button_name in ["B","UP","RIGHT","LEFT","DOWN","NONE"]:
            env.reset()
            save_parent=os.path.join(PATH,game,button_name)
            os.makedirs(save_parent, exist_ok=True)
            n_games=len(os.listdir(save_parent))
            save_dir=save_parent
            os.makedirs(save_dir,exist_ok=True)
        
            count=0
            
            output_dict={
                key:[] for key in info
            }
            output_dict["action"]=[]
            output_dict["save_path"]=[]
            action = np.zeros(len(BUTTONS), dtype=np.int8)
            for _ in range(100):
                env.step(action)
                env.render()
                #clock.tick(60)

            
            for _ in range(RIGHT_STEPS):
                
                right= np.zeros(len(BUTTONS), dtype=np.int8)
                
                idx = button_index.get("RIGHT", None)
                right[idx]=1
                env.step(right)
                #env.step(action)
                env.render()
                #clock.tick(60)
                #print(right)
            for _ in range(50):
                env.step(action)
                env.render()

            for count in range(30):
                action = np.zeros(len(BUTTONS), dtype=np.int8)


                pressed_list=[]

                pressed_list.append(button_name)
                idx = button_index.get(button_name, None)
                if idx is not None:
                    action[idx] = 1
                output_dict["action"].append(button_name)
                print(action,button_name,idx)



                # Step emulator
                
                obs, rew, terminated, truncated, info=env.step(action)

                image=obs
                image = Image.fromarray(image).resize((256,256))
                save_path_image=os.path.join(save_dir,f"{count}.jpg")
                image.save(save_path_image)
                env.render()

                # Save user action
                action_log.append(action.copy())

                #clock.tick(60)  # limit to 60 FPS for smooth control

                for key in info:
                    output_dict[key].append(info[key])
                output_dict['save_path'].append(save_path_image)


            
            #pygame.quit()

            # Save the action sequence in numpy format
            #np.save(ACTIONS_FILE, np.array(action_log, dtype=np.int8))
            #print(f"Saved {len(action_log)} actions to {ACTIONS_FILE}")

            #print("\nA .bk2 file has also been created in:", RECORD_DIR)

            with open(os.path.join(save_dir,"data.csv"), "w+") as outfile:

                # pass the csv file to csv.writer function.
                writer = csv.writer(outfile)

                # pass the dictionary keys to writerow
                # function to frame the columns of the csv file
                writer.writerow(output_dict.keys())
            
                # make use of writerows function to append
                # the remaining values to the corresponding
                # columns using zip function.
                writer.writerows(zip(*output_dict.values()))
        env.close()
