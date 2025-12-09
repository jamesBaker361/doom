import retro
import pygame
import numpy as np
import os
import time
from PIL import Image
import argparse
import csv



if __name__=='__main__':

    states={
        'SonicTheHedgehog2-Genesis':['EmeraldHillZone.Act1','AquaticRuinZone.Act1','CasinoNightZone.Act1','HillTopZone.Act1'],
        'SuperMarioWorld-Snes':["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'],
        'CastlevaniaBloodlines-Genesis-v0':['Level1-1','Level2-1','Level3-1','Level4-1']
    }

    state_help=""
    for k,v in states.items():
        state_help+=f"{k} has states: "+",".join(v)+"\n"

    parser=argparse.ArgumentParser()
    parser.add_argument("--game",type=str,default='SonicTheHedgehog2-Genesis',help="one of \n SuperMarioWorld-Snes \n CastlevaniaBloodlines-Genesis-v0 \n SonicTheHedgehog2-Genesis ")
    parser.add_argument("--state",type=str,default='EmeraldHillZone.Act1',help=state_help)
    parser.add_argument("--save_dir",type=str,default="videos")

    args=parser.parse_args()

    # ---- SETTINGS ----
    save_parent=os.path.join(args.save_dir,args.game,args.state)
    os.makedirs(save_parent, exist_ok=True)
    n_games=len(os.listdir(save_parent))
    save_dir=os.path.join(save_parent , str(n_games))
    os.makedirs(save_dir,exist_ok=True)

    # Create environment with recording enabled
    env = retro.make(
        game=args.game,
        state=args.state,
        record=save_dir,
        use_restricted_actions=retro.Actions.ALL
    )

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

    # ---- Init pygame ----
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption(" Controller - Focus this window to play!")

    clock = pygame.time.Clock()

    obs = env.reset()
    done = False

    # Recorded list of full button vectors
    action_log = []

    print("Controls:")
    print(" AWASD = Move")
    print(" Q = Buttons (jump/spin)")
    print(" Enter = Start")
    print(" ESC = Quit")
    print("------------------")
    PATH="videos"
    os.makedirs(PATH,exist_ok=True)
    count=0
    action = np.zeros(len(BUTTONS), dtype=np.int8)
    obs, rew, terminated, truncated, info=env.step(action)
    env.reset()
    output_dict={
        key:[] for key in info
    }
    output_dict["action"]=[]
    output_dict["save_path"]=[]
    try:
        while not done:
            count+=1
            if count>=5000:
                break
            pressed = pygame.key.get_pressed()
            action = np.zeros(len(BUTTONS), dtype=np.int8)


            pressed_list=[]
            for key, button_name in KEY_TO_BUTTON.items():
                if pressed[key]:
                    pressed_list.append(button_name)
                    idx = button_index.get(button_name, None)
                    if idx is not None:
                        action[idx] = 1
            output_dict["action"].append('-'.join(pressed_list))
            if np.sum(action)<1:
                output_dict["action"].append("None")

            # Per-step events (quit, ESC, etc.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

            # Step emulator
            
            obs, rew, terminated, truncated, info=env.step(action)

            image=obs
            image = Image.fromarray(image).resize((256,256))
            save_path_image=os.path.join(save_dir,f"{count}.jpg")
            image.save(save_path_image)
            env.render()

            # Save user action
            action_log.append(action.copy())

            clock.tick(60)  # limit to 60 FPS for smooth control

            for key in info:
                output_dict[key].append(info[key])
            output_dict['save_path'].append(save_path_image)

    except KeyboardInterrupt:
        pass

    finally:
        env.close()
        pygame.quit()

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
