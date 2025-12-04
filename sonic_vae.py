import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from experiment_helpers.init_helpers import repo_api_init
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.image_helpers import concat_images_horizontally
from experiment_helpers.data_helpers import split_data

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
import time
import torch.nn.functional as F
from PIL import Image
from diffusers.models.autoencoders.vq_model import VQModel
from constants import VAE_WEIGHTS_NAME
import random
import wandb
import numpy as np
import random
from gpu_helpers import *
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,AutoencoderKL
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from data_loaders import ImageDatasetHF
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoProcessor, CLIPModel
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import random_split, DataLoader
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi,hf_hub_download,upload_folder
import requests
from accelerate.utils import set_seed

set_seed(123)

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--repo_id",type=str,default="jlbaker361/vae-model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--image_size",default=256,type=int)
parser.add_argument("--image_folder_paths",nargs="+")
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--val_interval",type=int,default=10)
parser.add_argument("--skip_frac",type=float,default=1.0)
parser.add_argument("--use_hf_training_data",action="store_true")
parser.add_argument("--hf_data_path",type=str,default="")
parser.add_argument("--save_dir",type=str,default="sonic_vae_saved")
parser.add_argument("--src_dataset",type=str,default="jlbaker361/discrete_AquaticRuinZone.Act1100")
parser.add_argument("--load_hf",action="store_true")
parser.add_argument("--encoder_type",type=str,default="vae")
parser.add_argument("--skip_num",type=int,default=100)
parser.add_argument("--process_data",action="store_true")
parser.add_argument("--load_locally",action="store_true")


def main(args):
    e=0
    api,accelerator,device=repo_api_init(args)


    

    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    image_processor=pipe.image_processor
    

    if args.encoder_type=="vae":
        autoencoder=pipe.vae.to(device)
        
    elif args.encoder_type=="vqvae":
        autoencoder=VQModel()
    
    CONFIG_NAME="config.json"
    
    accelerator.print(autoencoder.config)
    
    save_subdir=os.path.join(os.getcwd(),args.save_dir,args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)

    dataset=ImageDatasetHF(args.src_dataset, image_processor,args.skip_num)
    accelerator.print("dataset len",len(dataset))


    train_loader,test_loader,val_loader=split_data(dataset,0.9,args.batch_size)

    

    
    params=[p for p in autoencoder.parameters()]

    optimizer=torch.optim.AdamW(params,args.lr)

    train_loader,autoencoder,optimizer=accelerator.prepare(train_loader,autoencoder,optimizer)

    

    for initial_batch in train_loader:
        break

    initial_batch=initial_batch["image"].to(device)
    batch_size=initial_batch.size()[0]

    
    
    

    save,load=save_and_load_functions({
        "pytorch_weights.safetensors":autoencoder,
        #"action_pytorch_weights.safetensors":action_encoder
    },save_subdir,api,args.repo_id)
    
    start_epoch=load(True)
    @optimization_loop(accelerator,train_loader,args.epochs,args.val_interval,
                       args.limit,val_loader,test_loader,save,start_epoch)
    def process_batch(batch,training,misc_dict):
        batch=batch["image"]
        if training:
            with accelerator.accumulate(params):
                with accelerator.autocast():
                    predicted=autoencoder(batch).sample
                    loss=F.mse_loss(predicted.float(),batch.float())
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        else:
            predicted=autoencoder(batch).sample
            loss=F.mse_loss(predicted.float(),batch.float())
            _batch_size=predicted.size()[0]
            predicted_images=image_processor.postprocess(predicted,do_denormalize= [True]*_batch_size)
            initial_images=image_processor.postprocess(initial_batch,do_denormalize= [True]*_batch_size)
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
            
        return loss.cpu().detach().numpy()
        
    process_batch()

        



                






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