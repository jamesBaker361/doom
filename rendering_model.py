import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from unet_helpers import prepare_metadata,forward_with_metadata,set_metadata_embedding
from constants import VAE_WEIGHTS_NAME

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
from accelerate.utils import set_seed
import time
import torch.nn.functional as F
from PIL import Image
import random
import wandb
import numpy as np
import random
from torch.utils.data import random_split, DataLoader
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderKL
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from experiment_helpers.loop_decorator import optimization_loop



from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi
from data_loaders import VelocityPositionDatasetHF,RenderingModelDatasetHF

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--repo_id",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--save_dir",type=str,default="weights")
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--load_hf",action="store_true")
parser.add_argument("--val_interval",type=int,default=10)
parser.add_argument("--action",type=str,default="embedding",help="encoder or embedding")
parser.add_argument("--dataset",type=str,default="jlbaker361/discrete_HillTopZone.Act1100")
parser.add_argument("--vae_checkpoint",type=str,default="jlbaker361/sonic-vae")

class ActionEncoder(torch.nn.Module):
    def __init__(self,input_dim:int,output_dim:int,n_layers:int,n_actions:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim=input_dim
        self.output_dim=output_dim
        dim_step=(self.output_dim-input_dim)//n_layers
        layer_dims=[input_dim]+[input_dim+(k* dim_step) for k in range(1,n_layers)]+[output_dim]
        print("layer dims action encoder",layer_dims)
        self.n_actions=n_actions
        
        layer_list=[]
        
        for n,dim in enumerate(layer_dims[:-1]):
            layer_list.append(torch.nn.Linear(dim,layer_dims[n+1]))
            layer_list.append(torch.nn.Dropout1d(0.1))
            layer_list.append(torch.nn.LeakyReLU())
            
        self.layers=torch.nn.Sequential(*layer_list)
        self.module_list=torch.nn.ModuleList([self.layers])
        
    def forward(self,x):
        x=F.one_hot(x,self.n_actions)
        return self.layers(x)

            

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(123)
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
            api.create_repo(args.repo_id,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.repo_id,exist_ok=True)

    
    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    unet=pipe.unet
    accelerator.print("len params before metadata ",len([p for p in unet.parameters()]))
    accelerator.print("len weight dict before metadata ",len(unet.state_dict()))
    unet=set_metadata_embedding(unet,2)
    accelerator.print("len params after metadata",len([p for p in unet.parameters()]))
    accelerator.print("len weight dict after metadata ",len(unet.state_dict()))
    vae=pipe.vae
    image_processor=pipe.image_processor

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    #dataset=??????

    dataset=RenderingModelDatasetHF(args.dataset,image_processor,["x","y"],True,
                                    None)
    n_actions=dataset.n_actions
    test_size=int(len(dataset)//4)
    train_size=int(len(dataset)-2*test_size)
    
    for batch in dataset:
        break
    
    print(batch["action"])
    
    action_dim=batch["action"].size()[-1]
    image_shape=batch["image"].unsqueeze(0).size()
    accelerator.print("image shape",image_shape)
    
    # Split the dataset
    train_dataset, test_dataset,val_dataset = random_split(dataset, [train_size, test_size,test_size],)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    for batch in train_loader:
        break

    save_subdir=os.path.join(args.save_dir,args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)

    WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
    
    ACTION_WEIGHTS_NAME="action_pytorch_model.safetensors"
    CONFIG_NAME="config.json"
    
    save_path=os.path.join(save_subdir,WEIGHTS_NAME)
    action_path=os.path.join(save_subdir,ACTION_WEIGHTS_NAME)
    config_path=os.path.join(save_subdir,CONFIG_NAME)
    
    
    
    DIM_PER_TOKEN=768
    N_TOKENS=4
    
    embedding_dim=DIM_PER_TOKEN*N_TOKENS
    
    if args.action=="embedding":
        action_encoder=torch.nn.Embedding(action_dim,embedding_dim)
    elif args.action=="encoder":
        action_encoder=ActionEncoder(action_dim,embedding_dim,3,n_actions)
    
    #vae=AutoencoderKL.from_pretrained(args.vae_checkpoint)

    start_epoch=1
    try:
        if args.load_hf:
            pretrained_weights_path=api.hf_hub_download(args.repo_id,WEIGHTS_NAME,force_download=True)
            pretrained_action_path=api.hf_hub_download(args.repo_id,ACTION_WEIGHTS_NAME,force_download=True)
            pretrained_config_path=api.hf_hub_download(args.repo_id,CONFIG_NAME,force_download=True)
            unet.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)
            action_encoder.load_state_dict(torch.load(pretrained_action_path,weights_only=True))
            with open(pretrained_config_path,"r") as f:
                data=json.load(f)
            start_epoch=data["start_epoch"]+1
    except Exception as e:
        accelerator.print(e)

    params=[p for p in unet.parameters()]+[p for p in action_encoder.parameters()]
    optimizer=torch.optim.AdamW(params,args.lr)
    
    optimizer,unet,action_encoder,train_loader,test_loader,val_loader = accelerator.prepare(optimizer,unet,action_encoder,train_loader,test_loader,val_loader)

    state_memory={
        "start_epoch":start_epoch
    }
    def save():
        #state_dict=???
        state_dict=unet.state_dict()
        state_dict_action=action_encoder.state_dict()
        print("state dict len",len(state_dict))
        torch.save(state_dict,save_path)
        torch.save(action_path,state_dict_action)
        e=state_memory["start_epoch"]
        state_memory["start_epoch"]+=1
        with open(config_path,"w+") as config_file:
            data={"start_epoch":e}
            json.dump(data,config_file, indent=4)
            pad = " " * 2048  # ~1KB of padding
            config_file.write(pad)
        print(f"saved {save_path}")
        try:
            api.upload_file(path_or_fileobj=save_path,
                                    path_in_repo=WEIGHTS_NAME,
                                    repo_id=args.repo_id)
            api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                    repo_id=args.repo_id)
            api.upload_file(path_or_fileobj=action_path,path_in_repo=ACTION_WEIGHTS_NAME,repo_id=args.repo_id)
            print(f"uploaded {args.repo_id} to hub")
        except Exception as e:
            accelerator.print("failed to upload")
            accelerator.print(e)

    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training):
        action=batch["action"]
        x=batch["x"]
        y=batch["y"]
        
        past_image=batch["past_image"]
        image=batch["image"]
        
        metadata=prepare_metadata(x,y)
        
        past_image=vae.encode(past_image).latent_dist.sample()
        image=vae.encode(image).latent_dist.sample()
        
        if training:
            with accelerator.accumulate(params):
                action_embedding=action_encoder(action)
                predicted=forward_with_metadata(unet,sample=past_image,
                                                encoder_hidden_states=action_embedding,
                                                metadata=metadata)
                loss=F.mse_loss(predicted.float(),image.float())
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        else:
            action_embedding=action_encoder(action)
            predicted=forward_with_metadata(unet,sample=past_image,
                                            encoder_hidden_states=action_embedding,
                                            metadata=metadata)
            loss=F.mse_loss(predicted.float(),image.float())
        return loss.cpu().detach().item()
        


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