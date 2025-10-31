from datasets import load_dataset, Dataset
import random

base_data=load_dataset("jlbaker361/sonic-vae-preprocessed",split="train")
episodes=list(set(base_data["episode"]))
n_episodes=len(episodes)
retain_episodes=n_episodes//10
valid=set(random.sample(episodes,retain_episodes))
print(n_episodes,retain_episodes)
new_data=base_data.filter(lambda x: x["episode"] in valid)
print(len(new_data))
new_data.push_to_hub("jlbaker361/sonic-vae-preprocessed-0.1")