from datasets import load_dataset, Dataset
import torch.nn.functional as F
from diffusers import DiffusionPipeline

data=load_dataset("jlbaker361/sonic-vae",split="train")
image_processor=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").image_processor
n_actions=len(set(data["action"]))
episode_set=set()
data=data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})
data=data.map(lambda x: {"action":F.one_hot(x["action"],n_actions)})

data.push_to_hub("jlbaker361/sonic-vae-preprocessed")