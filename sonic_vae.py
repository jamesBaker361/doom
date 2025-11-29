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
parser.add_argument("--name",type=str,default="jlbaker361/vae-model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--image_size",default=256,type=int)
parser.add_argument("--image_folder_paths",nargs="+")
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--image_interval",type=int,default=10)
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

def concat_images_horizontally(images)-> Image.Image:
    """
    Concatenate a list of PIL.Image objects horizontally.

    Args:
        images (List[PIL.Image]): List of PIL images.

    Returns:
        PIL.Image: A new image composed of the input images concatenated side-by-side.
    """
    # Resize all images to the same height (optional)
    heights = [img.height for img in images]
    min_height = min(heights)
    resized_images = [
        img if img.height == min_height else img.resize(
            (int(img.width * min_height / img.height), min_height),
            Image.LANCZOS
        ) for img in images
    ]

    # Compute total width and max height
    total_width = sum(img.width for img in resized_images)
    height = min_height

    # Create new blank image
    new_img = Image.new('RGB', (total_width, height))

    # Paste images side by side
    x_offset = 0
    for img in resized_images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

def main(args):
    e=0
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
            print("init error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)
        


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    

    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    image_processor=pipe.image_processor
    

    if args.encoder_type=="vae":
        autoencoder=pipe.vae.to(device)
        
    elif args.encoder_type=="vqvae":
        autoencoder=VQModel()
    
    CONFIG_NAME="config.json"
    
    accelerator.print(autoencoder.config)
    
    save_subdir=os.path.join(os.getcwd(),args.save_dir,args.name)
    os.makedirs(save_subdir,exist_ok=True)
    
    
    
    
    def load_model(repo_id: str):
        """
        Loads VAE from HuggingFace in correct architecture + resume epoch.
        """
        # 1. Download the folder
        vae = AutoencoderKL.from_pretrained(repo_id)

        # 2. Read training metadata (start_epoch)
        index_path = hf_hub_download(repo_id, CONFIG_NAME)
        with open(index_path, "r") as f:
            data = json.load(f)

        if "training" in data and "start_epoch" in data["training"]:
            start_epoch = data["training"]["start_epoch"] + 1
        else:
            start_epoch = 1  # fresh training

        print(f"[OK] Loaded VAE from {repo_id}, resume at epoch {start_epoch}")

        return vae, start_epoch
    
    def load_model_locally(subdir:str):
        vae = AutoencoderKL.from_pretrained(subdir)

        # 2. Read training metadata (start_epoch)
        index_path = os.path.join(subdir, CONFIG_NAME)
        with open(index_path, "r") as f:
            data = json.load(f)

        if "training" in data and "start_epoch" in data["training"]:
            start_epoch = data["training"]["start_epoch"] + 1
        else:
            start_epoch = 1  # fresh training

        print(f"[OK] Loaded VAE from {subdir}, resume at epoch {start_epoch}")

        return vae, start_epoch


    start_epoch=1
    if args.load_hf:
        try:
            autoencoder,start_epoch=load_model(args.name)
        except requests.exceptions.HTTPError:
            accelerator.print("not found couldnt load")
            
    if args.load_locally:
        autoencoder,start_epoch=load_model_locally(save_subdir)
        


    accelerator.print("start epoch: ",start_epoch)


    dataset=ImageDatasetHF(args.src_dataset, image_processor,args.process_data,args.skip_num)
    accelerator.print("dataset len",len(dataset))


    test_size=16
    train_size=len(dataset)-test_size

    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    

    
    params=[p for p in autoencoder.parameters()]

    optimizer=torch.optim.AdamW(params,args.lr)

    train_loader,autoencoder,optimizer=accelerator.prepare(train_loader,autoencoder,optimizer)

    

    for initial_batch in train_loader:
        break

    initial_batch=initial_batch["image"].to(device)

    
    
    

    def save_model(vae: AutoencoderKL, epoch: int, repo_id: str, save_dir: str = "vae_ckpt"):
        """
        Save AutoEncoderKL in HF-pretrained format + store start_epoch safely.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save full HF format (weights + config)
        vae.save_pretrained(save_dir)

        # 2. Add training metadata (e.g., epoch) into model_index.json
        index_path = os.path.join(save_dir, CONFIG_NAME)
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index_data = json.load(f)
        else:
            index_data = {}

        index_data["training"] = {"start_epoch": epoch}

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=4)

        accelerator.print(f"[OK] Saved VAE checkpoint to {save_dir}")

        # 3. Upload entire folder to HuggingFace repo
        
        upload_folder(
            folder_path=save_dir,
            repo_id=repo_id,
            path_in_repo=".",   # root of the repo
        )
        accelerator.print(f"[OK] Uploaded to HuggingFace: {repo_id}")

    with accelerator.autocast():    

        for e in range(start_epoch,args.epochs+1):
            start=time.time()
            loss_buffer=[]
            for b,batch in enumerate(train_loader):
                if b==args.limit:
                    break
                '''if b%args.skip_num!=0:
                    continue'''

                with accelerator.accumulate(params):
                    batch=batch["image"]
                    batch=batch.to(device)
                    #accelerator.print("batch size",batch.size())
                    predicted=autoencoder(batch).sample
                    loss=F.mse_loss(predicted.float(),batch.float())
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_buffer.append(loss.cpu().detach().item())

            end=time.time()

            accelerator.print(f"\t epoch {e} elapsed {end-start}")

            accelerator.log({
                    "loss_mean":np.mean(loss_buffer),
                    "loss_std":np.std(loss_buffer),
                })
            save_model(autoencoder,e,args.name,save_subdir)
            if e%args.image_interval==1:
                with torch.no_grad():
                    predicted_batch=autoencoder(initial_batch).sample
                    batch_size=predicted_batch.size()[0]
                    predicted_images=image_processor.postprocess(predicted_batch,do_denormalize= [True]*batch_size)
                    initial_images=image_processor.postprocess(initial_batch,do_denormalize= [True]*batch_size)
                    print('type(predicted_images)',type(predicted_images))
                    for k,(real,reconstructed) in enumerate(zip(initial_images,predicted_images)):
                        concatenated_image=concat_images_horizontally([real,reconstructed])
                        accelerator.log({
                            f"image_{k}":wandb.Image(concatenated_image)
                        })

    with torch.no_grad():
        save_model(autoencoder,e,args.name,save_subdir)
        for initial_batch in test_loader:
            initial_batch=initial_batch["image"].to(device)
            predicted_batch=autoencoder(initial_batch).sample
            batch_size=predicted_batch.size()[0]
            predicted_images=image_processor.postprocess(predicted_batch,do_denormalize= [True]*batch_size)
            initial_images=image_processor.postprocess(initial_batch,do_denormalize= [True]*batch_size)
            for k,(real,reconstructed) in enumerate(zip(initial_images,predicted_images)):
                concatenated_image=concat_images_horizontally([real,reconstructed])
                accelerator.log({
                    f"test_image_{k}":wandb.Image(concatenated_image)
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