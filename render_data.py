import argparse
from datasets import load_dataset,Dataset

parser=argparse.ArgumentParser()
parser.add_argument("--src",type=str,default="jlbaker361/discrete_EmeraldHillZone.Act15000000")
parser.add_argument("--dest",type=str,default="jlbaker361/discrete_EmeraldHillZone.Act15000000-render")

args=parser.parse_args()
src=load_dataset(args.src,split="train")
dest_dict={
    f:[] for f in src.features
}
dest_dict["past_image"]=[]
for i in range(1,len(src)):
    if src[i]["episode"]==src[i+1]["episode"]:
        for f in src.features:
            dest_dict[f].append(src[i][f])
        dest_dict["past_image"].append(src[i-1])
        
Dataset.from_dict(dest_dict).push_to_hub(args.dest)