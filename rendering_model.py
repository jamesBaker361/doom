import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from unet_helpers import prepare_metadata,forward_with_metadata,set_metadata_embedding,inference_metadata
from constants import VAE_WEIGHTS_NAME
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

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
from experiment_helpers.data_helpers import split_data
from experiment_helpers.image_helpers import concat_images_horizontally



from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi,hf_hub_download
from data_loaders import VelocityPositionDatasetHF,RenderingModelDatasetHF
from experiment_helpers.init_helpers import default_parser,repo_api_init

parser=default_parser()
parser.add_argument("--action",type=str,default="embedding",help="encoder or embedding")
parser.add_argument("--dataset",type=str,default="jlbaker361/discrete_HillTopZone.Act1100")
parser.add_argument("--vae_checkpoint",type=str,default="jlbaker361/sonic-vae")
parser.add_argument("--num_inference_steps",type=int,default=4)

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
    api,accelerator,device=repo_api_init(args)


    
    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    unet=pipe.unet
    accelerator.print("len params before metadata ",len([p for p in unet.parameters()]))
    accelerator.print("len weight dict before metadata ",len(unet.state_dict()))
    unet=set_metadata_embedding(unet,2)
    accelerator.print("len params after metadata",len([p for p in unet.parameters()]))
    accelerator.print("len weight dict after metadata ",len(unet.state_dict()))
    vae=pipe.vae.to(device)
    image_processor=pipe.image_processor
    unet.to(device)
    scheduler=FlowMatchEulerDiscreteScheduler.from_config(json.loads(open(hf_hub_download(
        "stabilityai/stable-diffusion-3-medium-diffusers","scheduler/scheduler_config.json")).read()))
    ddim_scheduler=DDIMScheduler()

    #dataset=??????

    dataset=RenderingModelDatasetHF(args.dataset,image_processor,["x","y"],True,
                                    None)
    n_actions=dataset.n_actions
    
    
    
    # Split the dataset
    train_loader,test_loader,val_loader=split_data(dataset,0.8,args.batch_size)
    for batch in train_loader:
        break
    
    action_dim=batch["action"].size()[-1]
    image_shape=batch["image"].size()
    accelerator.print("image shape",image_shape)
    batch_size=image_shape[0]

    save_subdir=os.path.join(args.save_dir,args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)



    DIM_PER_TOKEN=768
    #N_TOKENS=4
    
    embedding_dim=DIM_PER_TOKEN #*N_TOKENS
    
    if args.action=="embedding":
        action_encoder=torch.nn.Embedding(action_dim,embedding_dim).to(device)
    elif args.action=="encoder":
        action_encoder=ActionEncoder(action_dim,embedding_dim,3,n_actions).to(device)
    
    #vae=AutoencoderKL.from_pretrained(args.vae_checkpoint)

    

    params=[p for p in unet.parameters()]+[p for p in action_encoder.parameters()]
    optimizer=torch.optim.AdamW(params,args.lr)
    
    optimizer,unet,action_encoder,train_loader,test_loader,val_loader,scheduler,ddim_scheduler = accelerator.prepare(optimizer,
                    unet,action_encoder,train_loader,test_loader,val_loader,scheduler,ddim_scheduler)

    
    save,load=save_and_load_functions({
        "pytorch_weights.safetensors":unet,
        "action_pytorch_weights.safetensors":action_encoder
    },save_subdir,api,args.repo_id)
    
    start_epoch=load(True)

    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training,misc_dict):
        action=batch["action"]
        x=batch["x"]
        y=batch["y"]
        
        past_image=batch["past_image"]
        image=batch["image"]
        
        bsz=image.size()[0]
        
        
        
        if training:
            metadata=prepare_metadata(x,y)
        
            past_image=vae.encode(past_image).latent_dist.sample()*vae.config.scaling_factor
            image=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor
            
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()
            
            
            # model input scaling
            past_image = ddim_scheduler.add_noise(image,past_image,timesteps)
            with accelerator.accumulate(params):
                with accelerator.autocast():
                    action_embedding=action_encoder(action)
                    predicted=forward_with_metadata(unet,sample=past_image,
                                                    timestep=timesteps,
                                                    encoder_hidden_states=action_embedding,
                                                    metadata=metadata).sample
                    loss=F.mse_loss(predicted.float(),image.float())
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        else:
            decoded=inference_metadata(unet,action,action_encoder,vae,args.num_inference_steps,
                                       scheduler,past_image,x,y,device)
            loss=F.mse_loss(decoded.float(),image.float())
            _batch_size=decoded.size()[0]
            predicted_images=image_processor.postprocess(predicted,do_denormalize= [True]*_batch_size)
            initial_images=image_processor.postprocess(image,do_denormalize= [True]*_batch_size)
            start=0
            if "batch_num" in misc_dict:
                start=misc_dict["batch_num"]*batch_size
            mode="test"
            if "mode" in misc_dict:
                mode=misc_dict["mode"]
            for k,(real,reconstructed) in enumerate(zip(initial_images,predicted_images)):
                    concatenated_image=concat_images_horizontally([real,reconstructed])
                    accelerator.log({
                        f"image_{k+start}_{mode}":wandb.Image(concatenated_image)
                    })
        return loss.cpu().detach().item()
    
    batch_function()
        


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