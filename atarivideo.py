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

parser=argparse.ArgumentParser()
parser.add_argument("--game",type=str,default="ALE/Alien-v5")
parser.add_argument("--mode",type=int,default=0)
parser.add_argument("--difficulty",type=int,default=0)
parser.add_argument("--timesteps",type=int,default=100)

CSV_NAME="actions.csv"
FRAME_DIR="frames"

class FrameActionPerEpisodeLogger(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.frame_dir = os.path.join(save_dir, FRAME_DIR)
        self.csv_path = os.path.join(save_dir, CSV_NAME)
        os.makedirs(self.frame_dir, exist_ok=True)
        self.episode_idx = 0
        self.frame_idx = 0  # frame index within episode

        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "frame_in_episode", "action","file"])

    def _on_step(self) -> bool:
        # Environment is vectorized; assume single environment
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
                img.save(path)

            # Save action
            action = self.locals["actions"][0]
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_idx, self.frame_idx, int(action),filename])

            self.frame_idx += 1

        return True

args=parser.parse_args()
gymnasium.register_envs(ale_py)

env = gymnasium.make(args.game, render_mode="rgb_array",
                     full_action_space=True,
                     difficulty=args.difficulty,
                     mode=args.mode)
'''env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 1 == 0,
    video_folder="saved-video-folder",
    name_prefix="video-",
)'''

FOLDER_NAME=args.game.split("/")[-1]

callback = FrameActionPerEpisodeLogger(
    save_freq=1,           # Save every frame; increase if needed
    save_dir=FOLDER_NAME
)

model = PPO("CnnPolicy", env, verbose=1)

model.learn(args.timesteps,callback=callback)

