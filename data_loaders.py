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
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from datasets import load_dataset

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
                    posterior_indices.append(-1)
                    skip_num += 1
                else:
                    posterior_indices.append(i)

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
        print("posterior_indices",posterior_indices)
        for i in posterior_indices:
            if i==-1:
                tiny_posterior_list.append(self.zero_posterior)
            else:
                tiny_posterior_list.append(DiagonalGaussianDistribution(torch.tensor(self.posterior_list[i])).sample().squeeze(0))
        output_dict["posterior"]=torch.cat(tiny_posterior_list)
        return output_dict
    

class MovieImageFolderFromHF(MovieImageFolder):
    def __init__(self, hf_path, lookback):
        self.lookback=lookback
        self.data=load_dataset(hf_path,split="train")
        print("column names ",self.data.column_names)
        self.output_dict_list=[]
        self.posterior_list=self.data["posterior_list"]
        #self.data = self.data.remove_columns("posterior_list")

        for f,row in enumerate(self.data):
            if f == 0:
                posterior=row["posterior_list"]
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
                    posterior_indices.append(-1)
                    skip_num += 1
                else:
                    posterior_indices.append(i)

            output_dict = {
                "posterior_indices": posterior_indices,
                "skip_num": skip_num
            }
            for key,value in row.items():
                if key!="posterior_list":
                    output_dict[key]=value

            self.output_dict_list.append(output_dict)

class SequenceDatasetFromHF(Dataset):
    def __init__(self,hf_path,lookback):
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

            for i in range(start, f):
                if i < 0 or self.data[i]["episode"] != episode:
                    action_sequence.append(-1)
                    skip_num += 1
                else:
                    action_sequence.append(i)

            output_dict = {
                "action_sequence": action_sequence,
                "skip_num": skip_num
            }

            for key,value in row.items():
                output_dict[key]=value

            self.output_dict_list.append(output_dict)

    def __len__(self):
        return len(self.output_dict_list)
    

    def __getitem__(self, index):
        return self.output_dict_list[index]