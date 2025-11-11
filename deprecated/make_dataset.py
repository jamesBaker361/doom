'''
this module is for uploading a hf dataset from a saved folder
'''

import torch
import huggingface_hub
from datasets import Dataset
import argparse
from accelerate import Accelerator
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,AutoencoderKL,UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from diffusers.image_processor import VaeImageProcessor
import os
import pandas as pd
from PIL import Image
from huggingface_hub import create_repo,HfApi
import torch
import datetime

with torch.no_grad():

    parser=argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint",type=str,default="")
    parser.add_argument("--folder",type=str,default="sonic_videos_10/SonicTheHedgehog2-Genesis/EmeraldHillZone.Act1/gelly-religiousness-brazos/")
    parser.add_argument("--upload_path",type=str,default="jlbaker361/sonic10")
    parser.add_argument("--no_image",action="store_true")

    args=parser.parse_args()

    print(args)

    accelerator=Accelerator()
    vae=AutoencoderKL.from_pretrained("digiplay/DreamShaper_7",subfolder="vae").to(accelerator.device)
    image_processor=VaeImageProcessor(vae_scale_factor=8)

    if len(args.vae_checkpoint)>0:
        api=HfApi()

        pretrained_weights_path=api.hf_hub_download(args.vae_checkpoint,"diffusion_pytorch_model.safetensors",force_download=True)


        vae.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)

    csv_file = os.path.join(args.folder, "actions.csv")
    df=pd.read_csv(csv_file)

    columns=df.columns
    print(columns)

    output_dict=df.to_dict("list")
    if args.no_image !=True:
        posterior_list = []
        image_list=[]

        for n,file in enumerate(output_dict["file"]):
            with Image.open(os.path.join(args.folder, file)) as pil_image:
                pt_image = image_processor.preprocess(pil_image)
                posterior = vae.encode(pt_image.to(vae.device)).latent_dist.parameters.cpu().detach()
                
                posterior_list.append(posterior)
                image_list.append(pil_image)
        #output_dict["posterior_list"]=posterior_list
            output_dict["image"]=image_list
            output_dict["posterior"]=posterior_list

            if n%1000==1:
                truncated_output_dict={
                    key:value[:n] for key,value in output_dict.items()
                }
                Dataset.from_dict(truncated_output_dict).push_to_hub(args.upload_path)
                print(n, datetime.datetime.now())

    accelerator.print("all done :) ")
