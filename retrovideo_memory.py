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

import json
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

COMBO_LIST=[['LEFT'], ['RIGHT'], ['DOWN'], ['B'],['A']]

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
parser.add_argument("--hard_coded_steps",type=int,default=1000)
parser.add_argument("--schedule",nargs="*",type=int,default=[])
parser.add_argument("--repo_id",type=str,default="jlbaker361/sonic-rl-agent")


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


class FrameActionPerEpisodeLogger(BaseCallback):
    def __init__(self, save_freq: int,info_keys:list,
                 accelerator:accelerate.Accelerator,dest_dataset:str,
                 json_path:str,
                 verbose: int = 0,image_saving:bool=True,
                 episode_start:int=0,
                 steps_taken:int=0
                 ):
        super().__init__(verbose)
        self.save_freq = save_freq
        '''self.save_dir = save_dir
        self.frame_dir = os.path.join(save_dir, frame_dir)
        self.csv_path = os.path.join(save_dir,frame_dir, CSV_NAME)
        os.makedirs(self.frame_dir, exist_ok=True)'''
        self.episode_idx = episode_start
        self.frame_idx = 0  # frame index within episode
        self.info_keys=info_keys
        self.image_saving=image_saving
        try:
            self.output_dict=load_dataset(dest_dataset,split="train").to_dict()
            print("features",[k for k in self.output_dict.keys()])
        except:
            self.output_dict={
                key:[] for key in ["episode", "frame_in_episode", "action","image","action_combo"]+self.info_keys
            }
        self.accelerator=accelerator
        self.dest_dataset=dest_dataset
        self.cum_reward=0
        self.json_path=json_path
        self.steps_taken=steps_taken
        

    def _on_step(self) -> bool:
        self.steps_taken+=1
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
            self.output_dict["action_combo"].append(COMBO_LIST[action])

            for key,value in self.locals["infos"][0].items():
                
                if key in self.output_dict and key !="episode": #it adds its own episode field :(
                    self.output_dict[key].append(value)
            #print(", ".join([k for k in self.locals["infos"][0]]))
            self.frame_idx += 1

            '''if self.frame_idx%100==0:
                Dataset.from_dict(self.output_dict).push_to_hub(self.dest_dataset)'''
            '''if len(set([type(elem) for elem in self.output_dict["episode"]]))>1:
            print("hella types",self.frame_idx)
            print("self epsidoe idx",self.episode_idx)
            print("self locals",",".join([k for k in self.locals["infos"][0]]))'''
        dones = self.locals["dones"]
        #print(self.locals["infos"])
        if "rewards" in self.locals:
            accelerator.log({
                "reward":self.locals["rewards"][0]
            })
            self.cum_reward+=self.locals["rewards"][0]
            accelerator.log({
                "cum_reward":self.cum_reward
            })
        if dones[0]:
            self.cum_reward=0
            accelerator.log({
                "final_reward":self.locals["rewards"][0]
            })
            #print(self.output_dict["episode"])

            for k,v in self.output_dict.items():
                print(k,len(v))

            Dataset.from_dict(self.output_dict).push_to_hub(self.dest_dataset)
            self.episode_idx += 1
            with open(self.json_path,"w+") as file:
                json.dump({
                    "episode_start":self.episode_idx,
                    "steps_taken":self.steps_taken
                },file)
            self.frame_idx = 0  # reset per episode
        return True
    
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        print('buttons',buttons)
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()



class SonicDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=COMBO_LIST)
    
class MyWrapper(gym.Wrapper):
    def __init__(self, env,
                 #starting_x,
                 #starting_y,
                    length_schedule:list=[],
                    length_index:int=0):
        super().__init__(env)
        #self.starting_x=starting_x
        #self.starting_y=starting_y
        self.visited_y=set()
        self.rings=0
        self.visited_x=set()
        self.length_schedule=length_schedule
        self.length_index=length_index
        self.default_length=1000
        self.elapsed_steps=0
        self.current_score=0
        self.current_lives=-1
        
        
    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        self.elapsed_steps+=1

        rew=-1

        score=dict(info)["score"]
        if score> self.current_score:
            rew+=score-self.current_score
            self.current_score=score

        if self.current_lives==-1:
            self.current_lives=dict(info)["lives"]
        elif dict(info)["lives"]<self.current_lives:
            terminated=True
            self.current_lives=-1

        
        limit=self.default_length
        if len(self.length_schedule)>0:
            limit=self.length_schedule[min(len(self.length_schedule)-1,self.length_index)]
        if self.elapsed_steps>=limit:
            truncated=True
        return obs, rew, terminated, truncated, dict(info)
        
    def reset(self,seed=None,options=None):
        self.visited_y=set()
        self.rings=0
        self.visited_x=set()
        self.elapsed_steps=0
        self.length_index+=1
        self.current_score=0
        self.current_lives=-1
        return super().reset(seed=seed, options=options)



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
    
    original_reset=env.reset
    
    #starting_y=info["y"]

    action = env.action_space.sample()
    accelerator.print("action space",action,len(action))

    # Take the step using the random action
    env.reset()
    step= env.step(action)
    print('info',step[-1])
    #print(step)
    env.reset()
    
    
    info_keys=[k for k in step[-1].keys()]

    print("info keys",info_keys)

    if args.record:
        env = gymnasium.wrappers.RecordVideo(
            env,
            episode_trigger=lambda num: num % 1 == 0,
            video_folder="saved-video-folder",
            name_prefix="video-",
        )

    console=args.game.split("-")[-1]
    env=SonicDiscretizer(env)
    action = env.action_space.sample()
    action_space_size = env.action_space.n
    accelerator.print("discretized action ",env.action_space.sample())
    print("discretized Action space size:", action_space_size)
    
    


    save_path=os.path.join(MODEL_SAVE_DIR,args.repo_id)
    checkpoint_path=os.path.join(save_path,"checkpoints")
    json_path=os.path.join(save_path,"data.json")
    
    try:
        with open(json_path) as file:
            episode_start=json.load(file)["episode_start"]
    except:
        episode_start=0
        
    try:
        with open(json_path) as file:
            steps_taken=json.load(file)["steps_taken"]
    except:
        steps_taken=0
        
    print("episode_start",episode_start, "steps_taken",steps_taken, f"taking {args.timesteps-steps_taken} steps ")
    
    env=MyWrapper(env,
                  #starting_x,starting_y,
                  args.schedule,episode_start)
    try:
        model=PPO.load(save_path+".zip", env=env, verbose=1)
        print("successfully loaded")
    except Exception as e:
        model = PPO("CnnPolicy", env, verbose=1)
        print('didnt load',e)
        

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_path,
        name_prefix="ppo"
    )
    
    callback = FrameActionPerEpisodeLogger(
        dest_dataset=args.dest_dataset,
        json_path=json_path,
        accelerator=accelerator,
        save_freq=1,           # Save every frame; increase if needed
        info_keys=info_keys,
        image_saving=args.image_saving,
        episode_start=episode_start,
        steps_taken=steps_taken
    )

    model.learn(args.timesteps-steps_taken,callback=CallbackList([
        #checkpoint_callback, 
        callback]))
    model.save(save_path)
    
    output_dict=callback.output_dict
    hard_coded_steps=[]
    for q in range(args.hard_coded_steps):
        if q%50!=0:
            hard_coded_steps.append(COMBO_LIST.index(["RIGHT"]))
        else:
            hard_coded_steps.append(COMBO_LIST.index(["B"]))
            
    output_dict=callback.output_dict
    episode=output_dict["episode"][-1]+1
    for k,action in enumerate(hard_coded_steps):
        result=env.step(action)
        frame=result[0]
        info=result[-1]
        image=Image.fromarray(frame)
        output_dict["image"].append(image)
        output_dict["episode"].append(episode)
        output_dict["frame_in_episode"].append(k)
        output_dict["action"].append(action)
        output_dict["action_combo"].append(COMBO_LIST[action])
        for key,value in info.items():
            if key in output_dict:
                output_dict[key].append(value)
        

    gym.wrappers.TimeLimit

    for k, v in output_dict.items():
        print(k, len(v))
    Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)

    print("all done :)")