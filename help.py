from training.data_loaders import MovieImageFolder
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

folder="sonic_videos_10/SonicTheHedgehog2-Genesis/EmeraldHillZone.Act1/gelly-religiousness-brazos/"
vae=AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="vae")

image_processor=VaeImageProcessor(vae_scale_factor=8)
#data=MovieImageFolder(folder,vae,image_processor)