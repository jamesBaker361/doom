from datasets import Dataset,load_dataset, concatenate_datasets
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument("--dataset_list",nargs="*",type=str)
parser.add_argument("--dest_dataset_vae",type=str,default="jlbaker361/sonic-vae")
parser.add_argument("--dest_dataset_unet",type=str,default="jlbaker361/sonic-unet")

args=parser.parse_args()

data=load_dataset(args.dataset_list[0],split="train")
for row in data:
    break

unet_big_data=Dataset.from_dict({key:[] for key in row})
vae_big_data=Dataset.from_dict({key:[] for key in row})

print(unet_big_data)


for name in args.dataset_list:
    data=load_dataset(name,split="train")
    vae_data=data.filter(lambda row: row["episode"]%2==0).map(lambda ex: {"episode":name+"-"+str(ex["episode"])})
    unet_data=data.filter(lambda row: row["episode"]%2==1).map(lambda ex: {"episode":name+"-"+str(ex["episode"])})
    print(name, len(vae_data["image"]), len(unet_data["image"]))

    unet_big_data=concatenate_datasets([unet_big_data,unet_data])
    vae_big_data=concatenate_datasets([vae_big_data,vae_data])

unet_big_data.push_to_hub(args.dest_dataset_unet)
vae_big_data.push_to_hub(args.dest_dataset_vae)