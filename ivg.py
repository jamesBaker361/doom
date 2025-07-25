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
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,AutoencoderKL,UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance

from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi
from data_loaders import MovieImageFolder,MovieImageFolderFromHF
from torch.utils.data import DataLoader
from peft import LoraConfig
from diffusers.image_processor import VaeImageProcessor


parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
#parser.add_argument("--use_hf_training_data",action="store_true")
parser.add_argument("--hf_training_data",type=str,default="jlbaker361/sonic_100")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--folder",type=str,default="sonic_videos_10/SonicTheHedgehog2-Genesis/EmeraldHillZone.Act1/gelly-religiousness-brazos/")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--epochs",type=int,default=4)
parser.add_argument("--lookback",type=int,default=4)
parser.add_argument("--metadata_keys",nargs="*")
parser.add_argument("--use_lora",action="store_true")
parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
parser.add_argument("--n_actions",type=int,default=35,help="number of action embeddings that can be learned ")
parser.add_argument("--n_action_tokens",type=int,default=2,help="amount of text tokens to be learned for each action")
parser.add_argument("--drop_context_frames_probability",type=float,default=0.1)
parser.add_argument("--use_prior",action="store_true")


def main(args):
    args.metadata_keys=sorted(args.metadata_keys)
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
    accelerator.print("torch dtype",torch_dtype)
    with accelerator.autocast():

        #pipeline=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        #pipeline=DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        accelerator.print("pipeline loaded")
        '''
        vae loading path?
        '''
        vae=AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="vae")
        vae.requires_grad_(False)
        accelerator.print('vae')
        image_processor=VaeImageProcessor(vae_scale_factor=8)
        accelerator.print('image_processor')
        unet=UNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="unet")
        accelerator.print("unet")

        accelerator.print(4*args.lookback,unet.conv_in.out_channels,
                                    unet.conv_in.kernel_size,
                                    unet.conv_in.stride,
                                    unet.conv_in.padding)
        unet.conv_in=torch.nn.Conv2d(4*args.lookback,unet.conv_in.out_channels,
                                    kernel_size=unet.conv_in.kernel_size,
                                    stride=unet.conv_in.stride,
                                    padding=unet.conv_in.padding)
        accelerator.print("made conv in")
        scheduler=LCMScheduler()
        if args.use_lora:
            unet.requires_grad_(False)
            unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],)

            unet.add_adapter(unet_lora_config)
        unet.conv_in.requires_grad_(True)
        unet.class_embedding=torch.nn.Embedding(10,unet.time_embedding.linear_2.out_features,device=accelerator.device)
        params=[p for p in unet.parameters() if p.requires_grad]
        print("params",len(params))
        dataset=MovieImageFolderFromHF(args.hf_training_data,args.lookback,args.use_prior)
        loader=DataLoader(dataset,args.batch_size,shuffle=True)
        action_embedding=torch.nn.Embedding(args.n_actions,768*args.n_action_tokens,device=accelerator.device)
        accelerator.print(f" each embedding = 768 * {args.n_action_tokens} ={768*args.n_action_tokens} ")
        params+=[p for p in action_embedding.parameters()]

        for batch in loader:
            break

        print("posertior size",batch["posterior"].size())

        optimizer=torch.optim.AdamW(params,args.lr)

        optimizer,unet,loader,action_embedding=accelerator.prepare(optimizer,unet,loader,action_embedding)

        '''@torch.no_grad()
        def logging(unet,loader):'''

        


        for e in range(1,args.epochs+1):
            start=time.time()
            loss_buffer=[]
            for b,batch in enumerate(loader):
                with accelerator.accumulate(params):
                    latent=batch["posterior"].to(device)
                    action=batch["action"]
                    if e==1 and b==0:
                        accelerator.print("latent",latent.size())
                    skip_num=batch["skip_num"]
                    (B,C,H,W)=latent.size()
                    num_chunks=C//4
                    latent_chunks = latent.view(B, num_chunks, 4, H, W)
                    noise_chunks = torch.randn_like(latent_chunks)

                    # Generate timesteps:
                    # - one value for each chunk except the last → shape (B, num_chunks - 1)
                    # - one value for the last chunk only → shape (B,)
                    if random.random()<args.drop_context_frames_probability:
                        drop=True
                    else:
                        drop=False

                    if drop:
                        main_timesteps=torch.zeros((B,), device=latent.device)
                    else:
                        main_timesteps = torch.randint(
                            0, int(scheduler.config.num_train_timesteps * 0.7), (B,), device=latent.device
                        )

                    with torch.no_grad():
                        class_labels=main_timesteps//100
                        class_labels=class_labels.int()
                    last_timestep = torch.randint(
                        0, scheduler.config.num_train_timesteps, (B,), device=latent.device
                    )

                    # Run per chunk
                    noised_latent_chunks = []

                    for i in range(num_chunks):
                        latent_i = latent_chunks[:, i]      # (B, 4, H, W)
                        noise_i = noise_chunks[:, i]

                        if i == num_chunks - 1:
                            t_i = last_timestep             # (B,)
                        else:
                            t_i = main_timesteps      # (B,)

                        if drop and i != num_chunks-1:
                            noised_i=torch.zeros(noise_i.size(),device=device)
                        else:
                            noised_i = scheduler.add_noise(latent_i, noise_i, t_i)  # (B, 4, H, W)
                        noised_latent_chunks.append(noised_i)

                    # Reassemble
                    noised_latent_chunks = torch.stack(noised_latent_chunks, dim=1)           # (B, num_chunks, 4, H, W)
                    noised_latent = noised_latent_chunks.view(B, C, H, W)              # (B, C, H, W)
                    noise=noise_chunks.view(B,C,H,W)
                    # Reshape to blocks of 4 channels
                    '''latent_blocks = latent.view(B, C // 4, 4, H, W)

                    # Sample noise and timesteps
                    noise_blocks = torch.randn_like(latent_blocks)
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, 
                        (B, C // 4), 
                        device=device
                    )

                    # Apply add_noise to each 4-channel block individually
                    noised_latent_blocks = []
                    for i in range(C // 4):
                        latent_i = latent_blocks[:, i]         # (B, 4, H, W)
                        noise_i = noise_blocks[:, i]
                        t_i = timesteps[:, i]                  # (B,)

                        # scheduler.add_noise expects (B, C, H, W) and (B,)
                        noised_i = scheduler.add_noise(latent_i, noise_i, t_i)
                        noised_latent_blocks.append(noised_i)

                    # Stack along channel block dimension → (B, C//4, 4, H, W)
                    noised_latent_blocks = torch.stack(noised_latent_blocks, dim=1)

                    # Reshape back to (B, C, H, W)
                    noised_latent = noised_latent_blocks.view(B, C, H, W)
                    noise=noise_blocks.view(B,C,H,W)'''

                    if scheduler.config.prediction_type == "epsilon":
                        target = noise[:, - 4:, :, :] 
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(noised_latent[:,  - 4:, :, :] , noise[:, - 4:, :, :] , last_timestep)
                    encoder_hidden_states=action_embedding(action).reshape(B,2 ,-1)
                    if b==0 and e==1:
                        print('noised_latent.size()',noised_latent.size())
                        print('noised_latent[:,  - 4:, :, :].size()',noised_latent[:,  - 4:, :, :].size())
                        print('noise[:, - 4:, :, :].size()',noise[:, - 4:, :, :].size())
                        print("encodr hiden states",encoder_hidden_states.size())
                        print("class labels",class_labels)
                        print("noise[:, - 4:, :, :] ",noise[:, - 4:, :, :] .sum())
                        print("noise sum", noise.sum())
                        print("drop",drop)

                    
                    model_pred=unet(noised_latent,last_timestep,encoder_hidden_states=encoder_hidden_states,
                                    class_labels=class_labels,
                                    return_dict=False)[0] #somehow condiiton on main_timesteps ???
                    if b==0 and e==1:
                        print("model pred size",model_pred.size())
                        print("target size",target.size())

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_buffer.append(loss.cpu().detach().item())
            end=time.time()
            elapsed=end-start
            accelerator.print(f"\t epoch {e} elapsed {elapsed}")
            accelerator.log({
                "loss_mean":np.mean(loss_buffer),
                "loss_std":np.std(loss_buffer),
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