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
import ale_py

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
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")