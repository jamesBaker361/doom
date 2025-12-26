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
from gymnasium.wrappers import FrameStackObservation, TransformObservation,GrayscaleObservation
import ale_py
from gymnasium.spaces import Box

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
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
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


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation



    
COMBO_LIST=[['LEFT'], ['RIGHT'], ['DOWN'],['UP'] ,['B'],['A']]

if __name__=='__main__':
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
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
    
    
    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStackObservation(env, stack_size=4, new_step_api=True)
    else:
        env = FrameStackObservation(env, stack_size=4)
    
    env=Discretizer(env,COMBO_LIST)
    action = env.action_space.sample()
    env.reset()
    action = env.action_space.sample()
    print("action space",action,len(action))
    next_state, reward, done, trunc, info = env.step(action)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
    
    