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
    
class RenderingModelDatasetHF(Dataset):
    def __init__(self,src_dataset:str,image_processor:VaeImageProcessor,max_sequence_length,metadata_key_list:list=[],
                 process:bool=False,
                 vae:AutoencoderKL=None):
        super().__init__()
        self.data=load_dataset(src_dataset,split="train")
        #self.data=self.data.select(range(0,len(self.data),skip_num))
        self.image_processor=image_processor
        self.start_index_list=[]
        #self.n_actions=len(set(self.data["action"]))
        episode_set=set()
        if process:
            self.n_actions=len(set(self.data["action"]))
            self.data=self.data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})
            self.data=self.data.map(lambda x: {"action":F.one_hot(x["action"],self.n_actions)})
        else:
            self.n_actions=len(self.data["action"][0])
        
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
        return len(self.data)-1
    
    def __getitem__(self, index):
        '''#return super().__getitem__(index)
        end_index=find_earliest_less_than(self.start_index_list,index)
        output_dict={"image":torch.tensor(self.data["image"][index]),
                     "action":torch.tensor(self.data["action"][index])}
        if len(self.metadata_key_list)>0:
            for key in self.metadata_key_list:
                output_dict[key]=self.data[key][index:end_index]
            output_dict["metadata"]=torch.cat([output_dict[key] for key in self.metadata_key_list ])
        print(output_dict)
        for k,v in output_dict.items():
            shape=v[0].size()
            output_dict[k]+=[torch.zeros(shape) for _ in range(self.max_sequence_length)]
        output_dict["stop"]=end_index-index
        return output_dict'''
        image=torch.tensor(self.data["image"][index])
        if self.vae is not None:
            image=self.vae.encode(image).latent_dist.sample()
            
        next_image=torch.tensor(self.data["image"][index+1])
        if self.vae is not None:
            next_image=self.vae.encode(next_image).latent_dist.sample()
        output_dict={
            "image":image,
            "next_image":next_image
        }
        for key in self.metadata_key_list:
            output_dict[key]=self.data[key][index]
            
        return index
    
    
class VelocityPositionDatasetHF(Dataset):
    def __init__(self,src_dataset:str):
        super().__init__()
        self.data=load_dataset(src_dataset,split="train")
        self.start_index_list=[]
        episode_set=set()
        self.initial_velocity_x=[]
        self.initial_velocity_y=[]
        for i,row in enumerate(self.data):
            if row["episode"] not in episode_set:
                episode_set.add(row["episode"])
                self.start_index_list.append(i)
                self.initial_velocity_x.append(0)
                self.initial_velocity_y.append(0)
            else:
                prior_row=self.data[i-1]
                d_x=row["x"]-prior_row["x"]
                d_y=row["y"]-prior_row["y"]
                self.initial_velocity_x.append(d_x)
                self.initial_velocity_y.append(d_y)
        self.start_index_list.append(i)
        
    def __len__(self):
        return len(self.data)-1
    
    def __getitem__(self, index)->dict:
        output= {
            k:torch.tensor(self.data[k][index]) for k in ["action","image","x","y"]
        }
        output["vi_x"]=self.initial_velocity_x[index]
        output["vi_y"]=self.initial_velocity_y[index]
        
        output["xf"]=self.data["x"][index+1]
        output["yf"]=self.data["y"][index+1]
        
        output["vf_x"]=self.initial_velocity_x[index+1]
        output["vf_y"]=self.initial_velocity_y[index+1]
        
        return output
