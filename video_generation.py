from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch.nn import Embedding, Module
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,AutoencoderKL,UNet2DConditionModel
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import retrieve_timesteps
from torch import Tensor
from typing import List

def generate_frames(unet:UNet2DConditionModel,
                    lookback:int,
                    action_embedding:Embedding,
                    scheduler:LCMScheduler,
                    skip_num:int,
                    action_list:list,
                    nonzero_latent:Tensor,
                    num_inference_steps:int,
                    n_frames:int,
                    recurrent_model:Module)-> List[Tensor]:
    
    frame_list=nonzero_latent #B x 4C x H x W
    for f in range(n_frames):
        next_action=recurrent_model(action_list)
        batch=frame_list[:,-4*lookback:,:,:]
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, unet.device, timesteps, original_inference_steps=None
        )
        for t, step in enumerate(timesteps):
            


    