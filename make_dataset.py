import torch
import huggingface_hub
import datasets
import argparse
from accelerate import Accelerator
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,AutoencoderKL,UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from diffusers.image_processor import VaeImageProcessor
import os
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument("--vae_checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--folder",type=str,default="sonic_videos_10/SonicTheHedgehog2-Genesis/EmeraldHillZone.Act1/gelly-religiousness-brazos/")
parser.add_argument("--upload_path",type=str,default="jlbaker361/sonic10")

args=parser.parse_args()

print(args)

accelerator=Accelerator()

vae=AutoencoderKL.from_pretrained(args.vae_checkpoint,subfolder="vae")
image_processor=VaeImageProcessor(vae_scale_factor=8)

csv_file = os.path.join(args.folder, "actions.csv")
df=pd.read_csv(csv_file)


accelerator.print("all done :) ")
