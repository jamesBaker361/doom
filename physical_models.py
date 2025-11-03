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
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,AutoencoderKL
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from data_loaders import VelocityPositionDatasetHF
from torch.utils.data import random_split, DataLoader


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
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--image_encoder",type=str,help="one of vae, vqvae, trained",default="vae")
parser.add_argument("--n_layers_encoder",type=int,default=4)
parser.add_argument("--epochs",type=int,default=2)
parser.add_argument("--limit",type=int,default=10)

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
    def __init__(self,hidden_layer_dim_list:list,embedding_dim:int, action_dim:int,
                 image_encoder: torch.nn.Module=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim=embedding_dim+action_dim
        layers=[]
        dim_list=[input_dim]+hidden_layer_dim_list+[5]
        for k,dim in enumerate(dim_list[:-2]):
            layers.append(torch.nn.Linear(dim, dim_list[k+1]))
            layers.append(torch.nn.LeakyReLU())
            
        self.layers=torch.nn.Sequential(*layers)
        self.module_list=torch.nn.ModuleList([self.layers])
        self.g=torch.nn.Parameter(torch.randn([1]))
        self.mu_air=torch.nn.Parameter(torch.randn([1]))
        self.mu_ground=torch.nn.Parameter(torch.randn([1]))
        self.mass=1.
        self.image_encoder=image_encoder
        
    def forward(self,vi_x,vi_y,xi,yi,img,action):
        if self.image_encoder is not None:
            img_embedding=self.image_encoder(img)
        print("img",img_embedding.size(),"action",action.size())
        predicted=self.layers(torch.concat([img_embedding,action],dim=-1))
        fx_internal,fy_internal, fx_external,fy_external,theta_f=predicted.chunk(5,dim=1)

        mg=self.mass*self.g
        
        
        
        fx=fx_internal+fx_external+(2*np.cos(theta_f)*np.sin(theta_f)-self.mu_ground*(np.cos(theta_f)**2))*mg
        
        fy=fy_internal+fy_external+(-1+(np.cos(theta_f)**2)-(np.sin(theta_f)**2))*mg
        
        vf_x=vi_x+fx
        vf_y=vi_y+fy
        
        xf=xi+vf_x
        yf=yi+vf_y
        
        return vf_x,vf_y,xf,yf
    
class ImageEncoder(torch.nn.Module):
    def __init__(self,n_layers:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers=[torch.nn.Conv2d(3,4,4,2),torch.nn.LeakyReLU(),torch.nn.BatchNorm2d(4)]
        dim=4
        for _ in range(n_layers):
            layers+=[torch.nn.Conv2d(dim,2*dim,4,2),torch.nn.LeakyReLU(),torch.nn.BatchNorm2d(2*dim)]
            dim*=2
            
        layers.append(torch.nn.Flatten())
            
        self.layers=torch.nn.Sequential(*layers)
        self.module_list=torch.nn.ModuleList([self.layers])
        
    def forward(self,img:torch.Tensor)->torch.Tensor:
        return self.layers(img)
    
class VAEWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").vae
        self.flatten=torch.nn.Flatten()
        self.layer_list=torch.nn.ModuleList([self.vae,self.flatten])
        
    def forward(self,img:torch.Tensor)->torch.Tensor:
        if len(img.size())==3:
            img=img.unsqueeze(0)
        latents=self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
        return self.flatten(latents)

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    with accelerator.autocast():
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
        
        dataset=VelocityPositionDatasetHF("jlbaker361/sonic-vae-preprocessed-500")
        
        for batch in dataset:
            break
        
        action_dim=batch["action"].size()[-1]
        
        params=[]
        
        if args.image_encoder=="trained":
            image_encoder=ImageEncoder(args.n_layers_encoder).to(device)
            params+=[p for p in image_encoder.parameters()]
            accelerator.print("encoder params len ",len(params))
        elif args.image_encoder=="vae":
            image_encoder=VAEWrapper()
            
        image_embedding_dim=image_encoder(batch["image"]).size()[-1]
        
        test_size=int(len(dataset)//10)
        train_size=int(len(dataset)-2*test_size)

        
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(42)

        # Split the dataset
        train_dataset, test_dataset,val_dataset = random_split(dataset, [train_size, test_size,test_size], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        
        hidden_layer_dim_list=[256,128,64,32]
        
        model=Newtonian(hidden_layer_dim_list,image_embedding_dim,action_dim,image_encoder)
        params+=[p for p in model.parameters()]
        accelerator.print("model params",len(params))
        optimizer=torch.optim.AdamW(params,args.lr)
        
        
        start_epoch=1
        for e in range(start_epoch+1,args.epochs+1):
            loss_list=[]
            start=time.time()
            for b,batch in enumerate(train_loader):
                if b==args.limit:
                    break
                with accelerator.accumulate():
                    image=batch["image"]
                    action=batch["action"]
                    if e==start_epoch and b==0:
                        accelerator.print("image",image.device,image.dtype,image.size())
                        
                    vf_x,vf_y,xf,yf=model(batch["vi_x"],batch["vi_y"],batch["x"],batch["y"],image,action)
                    
                    vx_loss=F.mse_loss(vf_x.float(),batch["vf_x"].float())
                    vy_loss=F.mse_loss(vf_y.float(),batch["vf_y"].float())
                    x_loss=F.mse_loss(xf.float(),batch["xf"].float())
                    y_loss=F.mse_loss(yf.float(),batch["yf"].float())
                
                    total_loss=vx_loss+vy_loss+x_loss+y_loss
                    
                    loss_list.append(total_loss.cpu().detach().numpy())
                    
                    accelerator.backward(total_loss)
                    optimizer.step()
                    optimizer.zero_grad()
            end=time.time()
            accelerator.print(f"epoch {e} elapsed {end-start}")
            accelerator({
                "loss":np.mean(loss_list)
            })          
                
                
                    
                
                    
        


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