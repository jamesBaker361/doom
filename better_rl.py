#https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


import retro

import json
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation,GrayscaleObservation,ResizeObservation
import ale_py
from gymnasium.spaces import Box

import numpy as np
import time, datetime
import matplotlib.pyplot as plt
from agent_rl import Agent
from experiment_helpers.init_helpers import repo_api_init,default_parser,parse_args
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from datasets import load_dataset,Dataset
from extract_sprites import get_sprite_match


class MetricLogger:
    def __init__(self, save_dir,accelerator:Accelerator=None):
        self.accelerator=accelerator
        self.save_log = os.path.join(save_dir, "logs")
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
        if self.accelerator is not None:
            self.accelerator.log({
                f"moving_avg_{metric}":getattr(self, f"moving_avg_{metric}") for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]
            })

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
    
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip,repo_id:str,game:str,state:str):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.score=None
        self.lives=None
        self.game=game
        self.state=state
        self.repo_id=repo_id
        try:
            self.data_dict=load_dataset(repo_id,split="train").to_dict()
            if len(self.data_dict["episode"])>0:
                self.current_episode=1+max(self.data_dict["episode"])
        except:
            self.current_episode=0
            self.data_dict={
                "game":[],
                "state":[],
                "image":[],
                "episode":[],
                "overlay":[], # these will be none so we can separate template mmatching from rl training
                "use_overlay":[] # these will be none so we can separate template mmatching from rl training
            }

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            score=info["score"]
            lives=info["lives"]
            if self.score is None:
                self.score=score
                self.lives=lives
            else:
                if score>self.score:
                    reward+=(score-self.score)
                    self.score=score
                    #print("score =",self.score)
                if lives<self.lives:
                    #print("lives = ",lives)
                    done=True
            total_reward += reward
            if done:
                
                break
        self.data_dict["game"].append(self.game)
        self.data_dict["state"].append(self.state)
        self.data_dict["episode"].append(self.current_episode)
        self.data_dict["image"].append(Image.fromarray(obs))
        self.data_dict["overlay"].append(None)
        self.data_dict["use_overlay"].append(None)
        
        if done:
            self.lives=None
            self.score=None
            self.current_episode+=1
            Dataset.from_dict(self.data_dict).push_to_hub(self.repo_id)
        return obs, total_reward, done, trunk, info


'''class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation'''



    
COMBO_LIST=[['LEFT'], ['RIGHT'], ['DOWN'],['UP'] ,['B'],['A']]

def main(args):
    hf_api, accelerator,device=repo_api_init(args)
    GAME='SonicTheHedgehog2-Genesis'
    SCENARIO='MetropolisZone.Act1'
    env = retro.make(
                game=GAME,
                #state=args.state,
                scenario=SCENARIO,
                render_mode="rgb_array",
            )
    
    env.reset()
    action = env.action_space.sample()
    print("action space",action,len(action))
    next_state, reward, done, trunc, info = env.step(action)
    print(f"next_state.shape {next_state.shape},\n reward {reward},\n done {done},\n info {info}")
    
    stack_size=4
    h=next_state.shape[0]//2
    w=next_state.shape[1]//2
    
    # Apply Wrappers to environment
    sprite_dir=os.path.join("sprite_from_sheet",GAME)
    env = SkipFrame(env, skip=15)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=(h,w))
    env = FrameStackObservation(env, stack_size=4)
    
    env=Discretizer(env,COMBO_LIST)
    action = env.action_space.sample()
    env.reset()
    action = env.action_space.sample()
    print("action space",action)
    next_state, reward, done, trunc, info = env.step(action)
    print(f"next_state.shape {next_state.shape},\n reward {reward},\n done {done},\n info {info}")
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = os.path.join("checkpoints",GAME)
    os.makedirs(save_dir,exist_ok=True)
    
    mario = Agent(state_dim=(stack_size,h,w), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir,accelerator)

    episodes = 6
    for e in range(episodes):

        state, info= env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done:
                break

        logger.log_episode()

        if (e % 2 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
        
if __name__=='__main__':
    parser=default_parser()

    print_details()
    start=time.time()
    args=parse_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")