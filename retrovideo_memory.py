"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import gymnasium as gym
import numpy as np
#from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import retro


import gymnasium
import ale_py
from PIL import Image
import numpy as np
from datasets import Dataset,load_dataset
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback, CallbackList,CheckpointCallback
from PIL import Image
import os
import csv
import argparse
import random
import struct
import accelerate
import wandb


parser=argparse.ArgumentParser()
parser.add_argument("--game",type=str,default="SonicTheHedgehog2-Genesis")
parser.add_argument("--project_name",type=str,default="sonic_data")
parser.add_argument("--state",default=retro.State.DEFAULT)
parser.add_argument("--scenario", default="MetropolisZone.Act1")
parser.add_argument("--timesteps",type=int,default=10)
parser.add_argument("--record",action="store_true")
parser.add_argument("--save_dir",type=str,default="rl_videos")
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/sonic_rl")
parser.add_argument("--use_timelimit",action="store_true")
parser.add_argument("--max_episode_steps",type=int,default=50)
parser.add_argument("--image_saving",action="store_false")


CSV_NAME="actions.csv"
MODEL_SAVE_DIR="saved_rl_models"

from PIL import Image, ImageDraw, ImageFont

def pad_image_with_text(img:Image.Image, lines:list, font_size:int=20)->Image.Image:
    # Load original image
    width, height = img.size

    print("w,h",width,height)

    pad_height=len(lines)*font_size
    print("pad hieghts",pad_height)
    # Create new image with extra white padding at the bottom
    new_img = Image.new("RGB", (width, height + pad_height), color="white")
    new_img.paste(img, (0, 0))

    # Draw text in the padding area
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Windows/mac
    except:
        font = ImageFont.load_default()  # fallback

    line_spacing = pad_height // max(len(lines), 1)

    for i, line in enumerate(lines):
        y = height + i * line_spacing
        draw.text((10, y), line, fill="black", font=font)

    return new_img

'''

SNES street fighter 2
A: Forward (medium kick)

B: Short (light kick)

X: Strong (medium punch)

Y: Jab (light punch)

L: Fierce (hard punch)

R: Roundhouse (hard kick)


Genesis Street fighter 2

A-	Light Kick
B-	 Medium Kick
C-	 Hard Kick

X- Light Punch
Y- Medium Punch
Z- Hard Punch

Genesis-SNES
B=A
A=B
C=R
X=Y
Y=X
Z=L

'''

# SNES: ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
SNES_BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

# Genesis: ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
GENESIS_BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

SHARED_ACTION_MAP ={
    'NOOP':[],
    "UP":['UP'],
    'DOWN':['DOWN'],
    'LEFT':['LEFT'],
    'RIGHT':['RIGHT'],
}

GENESIS_BUTTON_BINDING={
    'BA':["B"],
    'AB':['A'],
    'CR':['C'],
    'XY':['X'],
    'YX':['Y'],
    'ZL':['Z']
}

SNES_BUTTON_BINDING={
    'BA':["A"],
    'AB':['B'],
    'CR':['R'],
    'XY':['Y'],
    'YX':['X'],
    'ZL':['L']
}

SNES_MAP={}
GENESIS_MAP={}

for console_map,button_binding in zip([SNES_MAP,GENESIS_MAP],[SNES_BUTTON_BINDING,GENESIS_BUTTON_BINDING]):
    for shared_k,shared_v in SHARED_ACTION_MAP.items():
        for button_k,button_v in button_binding.items():
            console_map[f"{shared_k}-{button_k}"]=shared_v+button_v
        console_map[f"{shared_k}"]=shared_v

print("snes_map",SNES_MAP)
print("genesis_map",GENESIS_MAP)


class Discretizer(gym.ActionWrapper):
    def __init__(self, env,console:str="Genesis"):
        super().__init__(env)
        if console=="Genesis":
            self.buttons=GENESIS_BUTTONS
            self.console_map=GENESIS_MAP
        elif console=="Snes":
            self.buttons=SNES_BUTTONS
            self.console_map=SNES_MAP
        self._actions=[]
        for action in self.console_map.keys():
            button_combo = self.console_map[action]
            arr = np.array([False] * len(self.buttons))
            for btn in button_combo:
                arr[self.buttons.index(btn)] = True
            self._actions.append(arr)
        
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()



class GenesisDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(GenesisDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()
    
class SNESDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for SNES games (e.g. Super Mario World, F-Zero).
    """
    def __init__(self, env):
        super(SNESDiscretizer, self).__init__(env)

        # SNES button layout in gym-retro
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

        # Define a reasonable set of discrete actions
        actions = [
            [],                     # NOOP
            ['LEFT'],              # Walk left
            ['RIGHT'],             # Walk right
            ['LEFT', 'B'],         # Run left
            ['RIGHT', 'B'],        # Run right
            ['B'],                 # Jump or run
            ['A'],                 # Spin jump or jump alt
            ['DOWN'],              # Duck/crouch
            ['UP'],                # Look up / enter pipe
            ['Y'],                 # Shoot / grab
            ['RIGHT', 'Y'],        # Shoot while moving
            ['RIGHT', 'B', 'Y'],   # Run + shoot + right
        ]

        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()

class FrameActionPerEpisodeLogger(BaseCallback):
    def __init__(self, save_freq: int,info_keys:list,accelerator:accelerate.Accelerator,dest_dataset:str,verbose: int = 0,image_saving:bool=True):
        super().__init__(verbose)
        self.save_freq = save_freq
        '''self.save_dir = save_dir
        self.frame_dir = os.path.join(save_dir, frame_dir)
        self.csv_path = os.path.join(save_dir,frame_dir, CSV_NAME)
        os.makedirs(self.frame_dir, exist_ok=True)'''
        self.episode_idx = 0
        self.frame_idx = 0  # frame index within episode
        self.info_keys=info_keys
        self.image_saving=image_saving
        self.output_dict={
            key:[] for key in ["episode", "frame_in_episode", "action","image"]+self.info_keys
        }
        self.accelerator=accelerator
        self.dest_dataset=dest_dataset
        

    def _on_step(self) -> bool:
        # Environment is vectorized; assume single environment
        

        if self.n_calls % self.save_freq == 0:
            # Save image
            frame = self.training_env.get_images()[0]
            if frame is not None and self.image_saving:
                image = Image.fromarray(frame)

            
            

            # Save action
            action = self.locals["actions"][0]


            self.output_dict["image"].append(image)
            self.output_dict["episode"].append(self.episode_idx)
            self.output_dict["frame_in_episode"].append(self.frame_idx)
            self.output_dict["action"].append(action)

            for key,value in self.locals["infos"][0].items():
                if key in self.output_dict:
                    self.output_dict[key].append(value)

            self.frame_idx += 1

            if self.frame_idx%100==0:
                Dataset.from_dict(self.output_dict).push_to_hub(self.dest_dataset)
        dones = self.locals["dones"]
        #print(self.locals["infos"])
        if dones[0]:
            accelerator.log({
                "reward":self.locals["rewards"][0]
            })

            Dataset.from_dict(self.output_dict).push_to_hub(self.dest_dataset)
            self.episode_idx += 1
            self.frame_idx = 0  # reset per episode
        return True


if __name__=="__main__":
    
    args=parser.parse_args()
    print(args)
    accelerator=accelerate.Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    gymnasium.register_envs(ale_py)

    env = retro.make(
                game=args.game,
                state=args.state,
                scenario=args.scenario,
                render_mode="rgb_array",
            )

    action = env.action_space.sample()

    # Take the step using the random action
    env.reset()
    step= env.step(action)
    print('info',step[-1])
    env.reset()

    info_keys=[k for k in step[-1].keys()]

    if args.record:
        env = gymnasium.wrappers.RecordVideo(
            env,
            episode_trigger=lambda num: num % 1 == 0,
            video_folder="saved-video-folder",
            name_prefix="video-",
        )

    console=args.game.split("-")[-1]
    env=Discretizer(env,console)
    action_space_size = env.action_space.n
    print("Action space size:", action_space_size)

    if args.use_timelimit:
        env=gym.wrappers.TimeLimit(env,args.max_episode_steps)

    


    callback = FrameActionPerEpisodeLogger(
        dest_dataset=args.dest_dataset,
        accelerator=accelerator,
        save_freq=1,           # Save every frame; increase if needed
        info_keys=info_keys,
        image_saving=args.image_saving
    )

    save_path=os.path.join(MODEL_SAVE_DIR,args.game,args.scenario)
    checkpoint_path=os.path.join(save_path,"checkpoints")
    try:
        model=PPO.load(save_path, env=env, verbose=1)
    except:
        model = PPO("CnnPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_path,
        name_prefix="ppo"
    )

    model.learn(args.timesteps,callback=CallbackList([checkpoint_callback, callback]))
    model.save(save_path)

    output_dict=callback.output_dict

    Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)

    print("all done :)")