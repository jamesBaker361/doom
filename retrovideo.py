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
from datasets import Dataset
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback
from PIL import Image
import os
import csv
import argparse
import random
import struct



parser=argparse.ArgumentParser()
parser.add_argument("--game",type=str,default="SonicTheHedgehog2-Genesis")
parser.add_argument("--state",default=retro.State.DEFAULT)
parser.add_argument("--scenario", default=None)
parser.add_argument("--timesteps",type=int,default=1)

CSV_NAME="actions.csv"

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


# Define variable memory references
VAR_MAP = {
    "x":         (16756744, ">H"),  # player world X
    "y":         (16756748, ">H"),  # player world Y
    "screen_x":  (16772608, ">H"),  # camera X
    "screen_y":  (16772612, ">H"),  # camera Y
    "screen_x_end": (16772810, ">H")
}

def read_variable(ram: bytes, address: int, fmt: str):
    # Map absolute address to RAM index (offset in 2KB RAM)
    print("ram type",type(ram))
    print("ram[0]",ram[0])
    ram_index = address % len(ram)
    return struct.unpack(fmt, ram[ram_index:ram_index + struct.calcsize(fmt)])[0]

def get_coords(env):
    ram = env.get_ram()
    return {
        name: read_variable(ram, addr, fmt)
        for name, (addr, fmt) in VAR_MAP.items()
    }

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
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
        print("a",a)
        return self._actions[a].copy()

class FrameActionPerEpisodeLogger(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, frame_dir:str,verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.frame_dir = os.path.join(save_dir, frame_dir)
        self.csv_path = os.path.join(save_dir,frame_dir, CSV_NAME)
        os.makedirs(self.frame_dir, exist_ok=True)
        self.episode_idx = 0
        self.frame_idx = 0  # frame index within episode

        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "frame_in_episode", "action","file"])

    def _on_step(self) -> bool:
        # Environment is vectorized; assume single environment
        print(self.locals["infos"])
        dones = self.locals["dones"]
        if dones[0]:
            self.episode_idx += 1
            self.frame_idx = 0  # reset per episode

        if self.n_calls % self.save_freq == 0:
            # Save image
            frame = self.training_env.get_images()[0]
            if frame is not None:
                filename = f"ep_{self.episode_idx:05d}_frame_{self.frame_idx:06d}.png"
                path = os.path.join(self.frame_dir, filename)
                img = Image.fromarray(frame)
                vec_env = self.model.get_env()
                retro_env = vec_env.envs[0].unwrapped
                #retro_env=
                coord_dict=get_coords(retro_env)
                print("coord dict",coord_dict)
                lines=[f"{key}={value}" for key,value in coord_dict.items()]
                img=pad_image_with_text(img,lines)
                img.save(path)

            # Save action
            action = self.locals["actions"][0]
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                print("action",action)
                writer.writerow([self.episode_idx, self.frame_idx, int(action),filename])

            self.frame_idx += 1

        return True

args=parser.parse_args()
gymnasium.register_envs(ale_py)

env = retro.make(
            game=args.game,
            state=args.state,
            scenario=args.scenario,
            render_mode="rgb_array",
        )

if args.game=="SonicTheHedgehog2-Genesis":
    env=SonicDiscretizer(env)
'''env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 1 == 0,
    video_folder="saved-video-folder",
    name_prefix="video-",
)'''

FOLDER_NAME=os.path.join("saved_retro_videos",args.game)
os.makedirs(FOLDER_NAME,exist_ok=True)

random_noun_list=[]
with open("random_nouns.txt","r") as file:
    for line in file:
        random_noun_list.append(line.strip())



frame_dir="-".join(random.sample(random_noun_list, 3))
print("frame dir",frame_dir)

callback = FrameActionPerEpisodeLogger(
    save_freq=1,           # Save every frame; increase if needed
    save_dir=FOLDER_NAME,
    frame_dir=frame_dir
)

model = PPO("CnnPolicy", env, verbose=1)

model.learn(args.timesteps,callback=callback)

print("all done :)")