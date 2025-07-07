import gymnasium
import ale_py
from PIL import Image
import numpy as np
from datasets import Dataset

gymnasium.register_envs(ale_py)

env = gymnasium.make("ALE/Alien-v5", render_mode="rgb_array")
'''env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 1 == 0,
    video_folder="saved-video-folder",
    name_prefix="video-",
)'''
all_episode_frame_list=[]
all_episode_action_list=[]
for episode in range(4):
    obs, info = env.reset()
    episode_over = False
    episode_frame_sequence=[Image.fromarray(obs)]
    episode_action_sequence=[0]  #0=NOOP
    while not episode_over:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_frame_sequence.append(Image.fromarray(obs))
        episode_action_sequence.append(action)

        episode_over = terminated or truncated
    all_episode_frame_list.append(episode_frame_sequence)
    all_episode_action_list.append(episode_action_sequence)

env.close()

data_map={
    "name":[],
    "episode_frame_sequence":[],
    "episode_action_sequence":[]
}

for i,(episode_frame_sequence,episode_action_sequence) in enumerate(zip(all_episode_frame_list,all_episode_action_list)):
    data_map["name"].append(f"video_{i}")
    data_map["episode_action_sequence"].append(episode_action_sequence)
    data_map["episode_frame_sequence"].append(episode_frame_sequence)

Dataset.from_dict(data_map).push_to_hub("jlbaker361/alien")