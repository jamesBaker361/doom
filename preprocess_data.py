from datasets import load_dataset, Dataset
import torch.nn.functional as F
from diffusers import DiffusionPipeline
import torch

data=load_dataset("jlbaker361/sonic-vae",split="train")
image_processor=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").image_processor
n_actions=len(set(data["action"]))
episode_set=set()
data=data.select([z for z in range(20)])
data=data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})
data=data.map(lambda x: {"action":F.one_hot(torch.Tensor(x["action"]).long(),n_actions)})

data.push_to_hub("jlbaker361/sonic-vae-preprocessed")

print("done :o")