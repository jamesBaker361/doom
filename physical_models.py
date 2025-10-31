import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
import time
import torch.nn.functional as F
from PIL import Image
import random
import wandb
import numpy as np
import random
from gpu_helpers import *
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from data_loaders import VelocityPositionDatasetHF


from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)

class Newtonian(torch.nn.Module):
    #given metadata and embedding , predict net forces on sonic using network, 
    # also have parameters (learnable) to represent g, friction, etc
    # might need a regularization term too
    # were going to model the world as if the rigid body is 1 kg
    # so there's internal force (running?) 
    # normal force (which isnt always a thing)
    # gravity (always downwards)
    # coefficient of friction with air (probably 0)
    # external forces (like from an enemy)
    # and all this is used to calculate direction of net velocity
    def __init__(self,hidden_layer_dim_list:list,embedding_dim:int, action_dim:int, *args, **kwargs):
        super().__init__()
        input_dim=embedding_dim+action_dim
        layers=[]
        dim_list=[input_dim]+hidden_layer_dim_list+[5]
        for k,dim in enumerate(dim_list[:-2]):
            layers.append(torch.nn.Linear(dim, dim_list[k+1]))
            
        self.layers=torch.nn.ModuleList(layers)
        self.g=torch.nn.Parameter([1])
        self.mu_air=torch.nn.Parameter([1])
        self.mu_ground=torch.nn.Parameter([1])
        self.mass=1.
        
    def forward(self,vi_x,vi_y,xi,yi,img_embedding,action):
        predicted=self.layers(torch.concat([img_embedding,action]))
        fx_internal,fy_internal, fx_external,fy_external,theta_f=predicted.chunk(5,dim=1)

        mg=self.mass*self.g
        
        
        
        fx=fx_internal+fx_external+(2*np.cos(theta_f)*np.sin(theta_f)-self.mu_ground*(np.cos(theta_f)**2))*mg
        
        fy=fy_internal+fy_external+(-1+(np.cos(theta_f)**2)-(np.sin(theta_f)**2))*mg
        
        vf_x=vi_x+fx
        vf_y=vi_y+fy
        
        xf=xi+vf_x
        yf=yi+vf_y
        
        return vf_x,vf_y,xf,yf

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    if accelerator.is_main_process or state.num_processes==1:
        try:
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    
    data_loader=VelocityPositionDatasetHF("jlbaker361/sonic-vae-preprocessed-0.1")
    
    for batch in data_loader:
        break
    
    print(batch)


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")