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
from data_loaders import FlatImageFolder,MovieImageFolderFromHF
from torch.utils.data import ConcatDataset, DataLoader
from diffusers import AutoencoderKL
from transformers import AutoProcessor, CLIPModel
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import random_split, DataLoader
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

    with accelerator.autocast():

        transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
        ])

        dataset_list=[]
        for path in args.image_folder_paths:
            dataset_list.append(FlatImageFolder(path,transform=transform,skip_frac=args.skip_frac))

        combined_dataset = ConcatDataset(dataset_list)

        test_size=8
        train_size=len(combined_dataset)-test_size

        
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(42)

        # Split the dataset
        train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size], generator=generator)


        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        autoencoder=AutoencoderKL.from_pretrained("digiplay/DreamShaper_7",subfolder="vae").to(device)
        params=[p for p in autoencoder.parameters()]

        optimizer=torch.optim.AdamW(params,args.lr)

        train_loader,autoencoder,optimizer=accelerator.prepare(train_loader,autoencoder,optimizer)

        image_processor=VaeImageProcessor()

        for initial_batch in train_loader:
            break

        save_subdir=os.path.join(args.save_dir,args.name)
        os.makedirs(save_subdir,exist_ok=True)
        WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
        save_path=os.path.join(save_subdir,WEIGHTS_NAME)
        def save():
            state_dict=autoencoder.state_dict()
            print("state dict len",len(state_dict))
            torch.save(state_dict,save_path)
            try:
                api.upload_file(path_or_fileobj=save_path,
                                path_in_repo=WEIGHTS_NAME,
                                repo_id=args.name)
                accelerator.print(f"uploaded {args.name} to hub")
            except:
                accelerator.print("failed to upload")

        for e in range(1,args.epochs+1):
            start=time.time()
            loss_buffer=[]
            for b,batch in enumerate(train_loader):
                if b==args.limit:
                    break
                with accelerator.accumulate(params):
                    batch=batch.to(device)
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
            save()
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
            save()
            n=0
            for initial_batch in test_loader:
                predicted_batch=autoencoder(initial_batch).sample
                batch_size=predicted_batch.size()[0]
                predicted_images=image_processor.postprocess(predicted_batch,do_denormalize= [True]*batch_size)
                initial_images=image_processor.postprocess(initial_batch,do_denormalize= [True]*batch_size)
                for k,(real,reconstructed) in enumerate(zip(initial_images,predicted_images)):
                    concatenated_image=concat_images_horizontally([real,reconstructed])
                    accelerator.log({
                        f"test_image_{n}":wandb.Image(concatenated_image)
                    })
                    n+=1

        



                






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