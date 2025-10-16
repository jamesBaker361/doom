from datasets import load_dataset, Dataset
import torch.nn.functional as F
from diffusers import DiffusionPipeline
import torch

data=load_dataset("jlbaker361/sonic-vae",split="train")
image_processor=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").image_processor
n_actions=max(set(data["action"]))+1
print(set(data["action"]),n_actions)
episode_set=set()
#data=data.select([z for z in range(2)])
#data=data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})

def one_hot(x):
    action=int(x["action"])
    zeros=[0 for _ in range(n_actions)]
    zeros[action]=1
    return {"action":torch.tensor(zeros)}

'''def f(x):
    try:
        y=F.one_hot(torch.Tensor(int(x["action"])).long(),n_actions)
    except Exception as e:
        print("error",x["action"],n_actions)
        raise e
    return {"action":y}'''

data=data.map(lambda x: one_hot(x))

data.push_to_hub("jlbaker361/sonic-vae-preprocessed")

print("done :o")