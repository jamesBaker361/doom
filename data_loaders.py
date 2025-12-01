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
import datasets

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
        dataset=load_dataset(src_dataset,split="train")
        try:
            dataset=dataset.cast_column("image",datasets.Image())
        except:
            pass
        data=dataset["image"]
        self.image_processor=image_processor
        if process:
            _image_list=[self.image_processor.preprocess(image)[0] for image in data]
        else:
            _image_list=data
        self.image_list=_image_list[::skip_num]
        

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image=torch.tensor(self.image_list[index])
        
        return {
            "image":image
        }
    
class RenderingModelDatasetHF(Dataset):
    def __init__(self,src_dataset:str,image_processor:VaeImageProcessor,
               #  max_sequence_length,
                 metadata_key_list:list=[],
                 process:bool=False,
                 vae:AutoencoderKL=None):
        super().__init__()
        self.data=load_dataset(src_dataset,split="train")
        try:
            self.data=self.data.cast_column("image",datasets.Image())
        except:
            pass
        #self.data=self.data.select(range(0,len(self.data),skip_num))
        self.image_processor=image_processor
        self.start_index_list=[]
        #self.n_actions=len(set(self.data["action"]))
        episode_set=set()
        if process:
            self.n_actions=len(set(self.data["action"]))
            if image_processor is not None:
                self.data=self.data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})
            self.data=self.data.map(lambda x: {"action":F.one_hot(torch.tensor(x["action"]),self.n_actions)})
        else:
            self.n_actions=len(self.data["action"][0])
        self.vae=vae
        for i,row in enumerate(self.data):
            if row["episode"] not in episode_set:
                episode_set.add(row["episode"])
                self.start_index_list.append(i)
        self.start_index_list.append(i)
        self.metadata_key_list=metadata_key_list
        #self.max_sequence_length=max_sequence_length
        '''if vae is not None:
            self.data.map(lambda x:)'''
        for key in metadata_key_list+["action"]:
            self.data=self.data.map(lambda x: {key:torch.tensor(x[key])})
            
        # ------------------------------------------------------------------ #
        #                   PRECOMPUTE VALID IMAGE PAIRS                     #
        # ------------------------------------------------------------------ #
        self.images = []
        self.next_images = []
        self.other_metadata = {k: [] for k in metadata_key_list+["action"]}

        # Episode tracking
        episodes = self.data["episode"]
        N = len(self.data)

        for i in range(N - 1):
            # Skip if next frame is from a new episode
            if episodes[i] != episodes[i + 1]:
                continue

            self.images.append(self.data["image"][i])
            self.next_images.append(self.data["image"][i + 1])

            for k in metadata_key_list+["action"]:
                self.other_metadata[k].append(self.data[k][i])

        # number of valid transitions
        self.length = len(self.images)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image = torch.tensor(self.images[index])
        next_image = torch.tensor(self.next_images[index])

        # Optional VAE encoding
        if self.vae is not None:
            image = self.vae.encode(image).latent_dist.sample()
            next_image = self.vae.encode(next_image).latent_dist.sample()

        out = {
            "image": image,
            "next_image": next_image,
        }

        for k in self.metadata_key_list+["action"]:
            out[k] = torch.tensor(self.other_metadata[k][index])

        return out
    
    
class VelocityPositionDatasetHF(Dataset):
    def __init__(self,src_dataset:str,image_processor:VaeImageProcessor=None,process:bool=False):
        super().__init__()
        self.data=load_dataset(src_dataset,split="train")
        try:
            self.data=self.data.cast_column("image",datasets.Image())
        except:
            pass
        
        self.start_index_list=[]
        episode_set=set()
        self.initial_velocity_x=[]
        self.initial_velocity_y=[]
        
        if process:
            self.n_actions=len(set(self.data["action"]))
            if image_processor is not None:
                self.data=self.data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})
            self.data=self.data.map(lambda x: {"action":F.one_hot(torch.tensor(x["action"]),self.n_actions)})
        else:
            self.n_actions=len(self.data["action"][0])
        
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
