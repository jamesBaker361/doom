from training.data_loaders import MovieImageFolder
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch

folder="sonic_videos_10/SonicTheHedgehog2-Genesis/EmeraldHillZone.Act1/gelly-religiousness-brazos/"
vae=AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="vae")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae=vae.to(device)

image_processor=VaeImageProcessor(vae_scale_factor=8)
data=MovieImageFolder(folder,vae,image_processor,4)
print("done")