from datasets import load_dataset
from PIL import Image
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--src_dataset",type=str,default="jlbaker361/SuperMarioWorld-Snes_DonutPlains1_rl_data")
parser.add_argument("--episode_list",type=int,nargs="*",default=[1])

args=parser.parse_args()

data=load_dataset(args.src_dataset,split="train")
data_name=args.src_dataset.split("/")[-1]
episode_set=set([e for e in data["episode"]])
for e in args.episode_list:
    if e in episode_set:
        episode_data=data.filter(lambda row: row["episode"]==e)["image"]
        episode_data[0].save(f"{data_name}_episode_{e}.gif",save_all=True,append_images=episode_data[1:],optimize=False,duration=len(episode_data)/4)
        
if -1 in args.episode_list:
    biggest=max(episode_set)
    episode_data=data.filter(lambda row: row["episode"]==biggest)["image"]
    episode_data[0].save(f"{data_name}_episode_final.gif",save_all=True,append_images=episode_data[1:],optimize=False,duration=len(episode_data)/4)