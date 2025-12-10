from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers import AutoencoderKL
from datasets import load_dataset
import datasets
import torch

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

class SequenceGameDatasetHF(Dataset):
    def __init__(self, src_dataset, image_processor, metadata_key_list=[], 
                 sequence_length:int=1,
                 process=False,
                 
                 vae=None):
        super().__init__()
        self.data = load_dataset(src_dataset, split="train")

        try:
            self.data = self.data.cast_column("image", datasets.Image())
        except Exception as e:
            print("map error ",e)

        self.image_processor = image_processor
        self.metadata_key_list = metadata_key_list
        self.vae = vae

        # preprocess metadata if needed
        if process:
            self.n_actions = len(set(self.data["action"]))
            '''if image_processor is not None:
                self.data = self.data.map(
                    lambda x: {"image": image_processor.preprocess(x["image"])[0]},
                    batched=False
                )'''
            '''self.data = self.data.map(
                lambda x: {"action": F.one_hot(torch.tensor(x["action"]), self.n_actions)},
                batched=False
            )'''
        else:
            self.n_actions = len(self.data["action"][0])

        # -------------------------------------------- #
        #         BUILD ONLY INDEX PAIRS (cheap)        #
        # -------------------------------------------- #

        self.index_list = []
        self.seqence_length=sequence_length
        episodes = self.data["episode"]
        N = len(self.data)

        for i in range(sequence_length,N ):
            # skip crossing episodes
            if episodes[i] != episodes[i - sequence_length]:
                continue
            self.index_list.append(i)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        i = self.index_list[idx]
        row = self.data[i]

        '''row = self.data[i]
        past_row = self.data[i - ]

        img = row["image"]
        past_img = past_row["image"]

        if self.image_processor:
            img = self.image_processor.preprocess(img)[0]
            past_img = self.image_processor.preprocess(past_img)[0]'''
        sequence=[self.data[i-j] for j in range(self.seqence_length)]
        sequence=self.image_processor.preprocess(sequence)
        out = {"sequence":sequence}
        
        # metadata
        for k in self.metadata_key_list: # ["action"]:
            out[k] = torch.tensor(row[k])
            
        out["action"]=F.one_hot(torch.tensor(row["action"]),self.n_actions)
        return out
        

class ImageDatasetHF(Dataset):
    def __init__(self,src_dataset:str,
                 image_processor:VaeImageProcessor,
                 skip_num:int=1):
        super().__init__()
        dataset=load_dataset(src_dataset,split="train")
        
        dataset=dataset.cast_column("image",datasets.Image())
        data=dataset["image"]
        self.image_processor=image_processor
        if image_processor is not None:
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
    def __init__(self, src_dataset, image_processor, metadata_key_list=[], process=False, vae=None):
        super().__init__()
        self.data = load_dataset(src_dataset, split="train")

        try:
            self.data = self.data.cast_column("image", datasets.Image())
        except Exception as e:
            print("map error ",e)

        self.image_processor = image_processor
        self.metadata_key_list = metadata_key_list
        self.vae = vae

        # preprocess metadata if needed
        if process:
            self.n_actions = len(set(self.data["action"]))
            '''if image_processor is not None:
                self.data = self.data.map(
                    lambda x: {"image": image_processor.preprocess(x["image"])[0]},
                    batched=False
                )'''
            '''self.data = self.data.map(
                lambda x: {"action": F.one_hot(torch.tensor(x["action"]), self.n_actions)},
                batched=False
            )'''
        else:
            self.n_actions = len(self.data["action"][0])

        # -------------------------------------------- #
        #         BUILD ONLY INDEX PAIRS (cheap)        #
        # -------------------------------------------- #

        self.index_list = []

        episodes = self.data["episode"]
        N = len(self.data)

        for i in range(1,N ):
            # skip crossing episodes
            if episodes[i] != episodes[i - 1]:
                continue
            self.index_list.append(i)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        i = self.index_list[idx]

        row = self.data[i]
        past_row = self.data[i - 1]

        img = row["image"]
        past_img = past_row["image"]

        if self.image_processor:
            img = self.image_processor.preprocess(img)[0]
            past_img = self.image_processor.preprocess(past_img)[0]

        out = {"image": img, "past_image": past_img}
        
        # metadata
        for k in self.metadata_key_list: # ["action"]:
            out[k] = torch.tensor(row[k])
            
        out["action"]=F.one_hot(torch.tensor(row["action"]),self.n_actions)
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
