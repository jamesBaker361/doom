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

from stable_baselines3.common.callbacks import BaseCallback, CallbackList,CheckpointCallback
from PIL import Image
import os
import csv
import argparse
import random
import struct



parser=argparse.ArgumentParser()
parser.add_argument("--game",type=str,default="SonicTheHedgehog2-Genesis")
parser.add_argument("--state",default=retro.State.DEFAULT)
parser.add_argument("--scenario", default="MetropolisZone.Act1")
parser.add_argument("--timesteps",type=int,default=10)
parser.add_argument("--record",action="store_true")

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
        return self._actions[a].copy()

class FrameActionPerEpisodeLogger(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, frame_dir:str,info_keys:list,verbose: int = 0,):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.frame_dir = os.path.join(save_dir, frame_dir)
        self.csv_path = os.path.join(save_dir,frame_dir, CSV_NAME)
        os.makedirs(self.frame_dir, exist_ok=True)
        self.episode_idx = 0
        self.frame_idx = 0  # frame index within episode
        self.info_keys=info_keys

        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "frame_in_episode", "action","file"]+self.info_keys)

    def _on_step(self) -> bool:
        # Environment is vectorized; assume single environment
        dones = self.locals["dones"]
        #print(self.locals["infos"])
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
                img.save(path)

            # Save action
            action = self.locals["actions"][0]
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                try:
                    row=[self.episode_idx, self.frame_idx, int(action),filename]+[self.locals["infos"][0][value] for value in self.info_keys]
                except TypeError as e:
                    print(action)
                    raise e
                writer.writerow(row)

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
    info_keys=["x","y","screen_x","screen_y","score","lives"]
else:
    info_keys=[]

if args.record:
    env = gymnasium.wrappers.RecordVideo(
        env,
        episode_trigger=lambda num: num % 1 == 0,
        video_folder="saved-video-folder",
        name_prefix="video-",
    )

FOLDER_NAME=os.path.join("saved_retro_videos",args.game,args.scenario)
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
    frame_dir=frame_dir,
    info_keys=info_keys
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

print("all done :)")