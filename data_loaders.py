import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch
import random
import csv
from gpu_helpers import *
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers import AutoencoderKL
from datasets import load_dataset
from constants import *
import numpy as np
import torch.nn.functional as F

NULL_ACTION=35 #this is the "button" pressed for null frames ()

def find_earliest_less_than(arr, target):
    left, right = 0, len(arr) - 1
    result = None

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < target:
            result = arr[mid]      # candidate found
            right = mid - 1        # but try to find earlier one
        else:
            left = mid + 1

    return result

class ImageDatasetHF(Dataset):
    def __init__(self,src_dataset:str,
                 image_processor:VaeImageProcessor,
                 process:bool=False,
                 skip_num:int=1):
        super().__init__()
        data=load_dataset(src_dataset,split="train")["image"]
        if process:
            _image_list=[self.image_processor.preprocess(image)[0] for image in data]
        else:
            _image_list=data
        self.image_list=_image_list[::skip_num]
        self.image_processor=image_processor

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image=torch.tensor(self.image_list[index])
        
        return {
            "image":image
        }
    
class WorldModelDatasetHF(Dataset):
    def __init__(self,src_dataset:str,image_processor:VaeImageProcessor,max_sequence_length,metadata_key_list:list=[],
                 process:bool=False,
                 vae:AutoencoderKL=None):
        super().__init__()
        self.data=load_dataset(src_dataset,split="train")
        #self.data=self.data.select(range(0,len(self.data),skip_num))
        self.image_processor=image_processor
        self.start_index_list=[]
        self.n_actions=len(set(self.data["action"]))
        episode_set=set()
        if process:
            self.data=self.data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})
            self.data=self.data.map(lambda x: {"action":F.one_hot(x["action"],self.n_actions)})
        for i,row in enumerate(self.data):
            if row["episode"] not in episode_set:
                episode_set.add(row["episode"])
                self.start_index_list.append(i)
        self.start_index_list.append(i)
        self.metadata_key_list=metadata_key_list
        self.max_sequence_length=max_sequence_length
        '''if vae is not None:
            self.data.map(lambda x:)'''
        for key in metadata_key_list:
            self.data=self.data.map(lambda x: {key:torch.tensor(x[key])})

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        #return super().__getitem__(index)
        end_index=find_earliest_less_than(self.start_index_list,index)
        output_dict={"image":torch.tensor(self.data["image"][index:end_index]),
                     "action":torch.tensor(self.data["action"][index:end_index])}
        for key in self.metadata_key_list:
            output_dict[key]=self.data[key][index:end_index]
        output_dict["metadata"]=torch.cat([output_dict[key] for key in self.metadata_key_list ])
        for k,v in output_dict.items():
            shape=v[0].size()
            output_dict[k]+=[torch.zeros(shape) for _ in range(self.max_sequence_length)]
        output_dict["stop"]=end_index-index
        return output_dict


class FlatImageFolder(Dataset):
    def __init__(self, folder, transform=None,skip_frac=0):
        paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.paths=[p for p in paths if random.random()>skip_frac]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    
class FlatImageFolderFromHF(Dataset):
    def __init__(self, folder, transform=None,skip_frac=0):
        paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.paths=[p for p in paths if random.random()>skip_frac]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    

class MovieImageFolder(Dataset):

    def __init__(self, folder, vae, image_processor, lookback: int):
        super().__init__()
        self.lookback = lookback
        csv_file = os.path.join(folder, "actions.csv")

        # Load CSV into a list of dicts (like a DataFrame)
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)  # Each row is a dict with keys like "file", "episode", etc.

        self.posterior_list = []
        for f, row in enumerate(self.data):
            before=find_cuda_objects()
            file = row["file"]
            pil_image = Image.open(os.path.join(folder, file))
            pt_image = image_processor.preprocess(pil_image)
            posterior = vae.encode(pt_image.to(vae.device)).latent_dist.parameters.cpu().detach()
            
            self.posterior_list.append(posterior)
            torch.cuda.empty_cache()
            if f == 0:
                self.zero_posterior = torch.zeros(DiagonalGaussianDistribution(posterior).sample().size()).squeeze(0)
                print("zero posterior",self.zero_posterior.size())
            after=find_cuda_objects()
            delete_unique_objects(before,after)
        self.output_dict_list = []
        for index in range(len(self.posterior_list)):
            episode = self.data[index]["episode"]
            start = index - self.lookback
            posterior_indices = []
            skip_num = 0

            for i in range(start, index):
                if i < 0 or self.data[i]["episode"] != episode:
                    posterior_indices.append(NULL_ACTION)
                    skip_num += 1
                else:
                    posterior_indices.append(torch.Tensor([self.data[i]["action"]]))

            output_dict = {
                "posterior_indices": posterior_indices,
                "skip_num": skip_num
            }

            # Add all other metadata columns (except 'file')
            for key, value in self.data[index].items():
                if key != "file":
                    try:
                        output_dict[key] = int(value)
                    except ValueError:
                        output_dict[key] = value

            self.output_dict_list.append(output_dict)

    def __len__(self):
        return len(self.posterior_list)

    def __getitem__(self, index):
        output_dict= self.output_dict_list[index]
        posterior_indices=output_dict["posterior_indices"]
        tiny_posterior_list=[]
        posterior_indices=[int(i.item()) for i in posterior_indices]
        for i in posterior_indices:
            if i==NULL_ACTION:
                tiny_posterior_list.append(self.zero_posterior)
            else:
                tiny_posterior_list.append(DiagonalGaussianDistribution(torch.tensor(self.posterior_list[i])).sample().squeeze(0))
        output_dict["posterior"]=torch.cat(tiny_posterior_list)
        return output_dict
    

class MovieImageFolderFromHF(MovieImageFolder):
    def __init__(self, hf_path, lookback,prior=False):
        self.lookback=lookback
        self.data=load_dataset(hf_path,split="train")
        print("column names ",self.data.column_names)
        self.output_dict_list=[]
        self.posterior_list=self.data["posterior"]
        #self.data = self.data.remove_columns("posterior_list")

        for f,row in enumerate(self.data):
            if f == 0:
                posterior=row["posterior"]
                try:
                    self.zero_posterior = torch.zeros(DiagonalGaussianDistribution(posterior).sample().size()).squeeze(0)
                except TypeError:
                    self.zero_posterior = torch.zeros(DiagonalGaussianDistribution(torch.tensor(posterior)).sample().size()).squeeze(0)
                print("zero posterior",self.zero_posterior.size())
            output_dict={}
            
            episode = row["episode"]
            start = f- self.lookback
            posterior_indices = []
            skip_num = 0

            for i in range(start, f):
                if i < 0 or self.data[i]["episode"] != episode:
                    posterior_indices.append(torch.Tensor([NULL_ACTION]))
                    skip_num += 1
                else:
                    posterior_indices.append(torch.Tensor([self.data[i]["action"]]))

            output_dict = {
                "posterior_indices": posterior_indices,
                "skip_num": skip_num
            }
            for key,value in row.items():
                if key!="posterior_list":
                    if key =="image":
                        value=np.array(value)
                    output_dict[key]=value
                    

            if prior and len(self.output_dict_list)>0:
                prior_output_dict=self.output_dict_list[-1]
                for key,value in prior_output_dict.items():
                    output_dict[PRIOR_PREFIX+key]=value

            self.output_dict_list.append(output_dict)

class SequenceDatasetFromHF(Dataset):
    def __init__(self,hf_path,lookback,prior=False):
        super().__init__()
        self.lookback=lookback
        self.data=load_dataset(hf_path,split="train")
        print("column names ",self.data.column_names)
        self.output_dict_list=[]

        for f,row in enumerate(self.data):
            output_dict={}
            
            episode = row["episode"]
            start = f- self.lookback
            action_sequence = []
            skip_num = 0
            for i in range(start, f+1):
                if i < 0 or self.data[i]["episode"] != episode:
                    action_sequence.append(torch.Tensor([NULL_ACTION]))
                    skip_num += 1
                else:
                    #print(self.data[i]["action"],self.data[i]["frame_in_episode"])
                    action_sequence.append(torch.Tensor([self.data[i]["action"]]))

            #print(action_sequence)
            output_dict = {
                "action_sequence": torch.stack(action_sequence).int().squeeze(-1),
                "skip_num": skip_num
            }

            for key,value in row.items():
                output_dict[key]=value

            if prior:
                if len(self.output_dict_list)>0:
                    prior_output_dict=self.output_dict_list[-1]
                    for key,value in prior_output_dict.items():
                        if key.find(PRIOR_PREFIX)==-1:
                            output_dict[PRIOR_PREFIX+key]=value
                else:
                    keys=[k for k in output_dict.keys()]
                    for k in keys:
                        output_dict[PRIOR_PREFIX+k]=output_dict[k]

            self.output_dict_list.append(output_dict)

    def __len__(self):
        return len(self.output_dict_list)
    

    def __getitem__(self, index):
        return self.output_dict_list[index]