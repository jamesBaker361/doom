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
from torch.utils.data import DataLoader, random_split, TensorDataset
from torcheval.metrics import PeakSignalNoiseRatio

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
from lpips import LPIPS


parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="ivg")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--name",type=str,default="jlbaker361/test-ivg",help="name on hf")
#parser.add_argument("--use_hf_training_data",action="store_true")
parser.add_argument("--hf_training_data",type=str,default="jlbaker361/sonic_chemical_50000_encoded") #replace chemical with hilltop, aqua, casino, emerald, hilltop
parser.add_argument("--vae_checkpoint",type=str,default="jlbaker361/sonic-vae50000-ChemicalPlantZone.Act11") 
#replace ChemicalPlantZone.Act11 with MetropolisZone.Act11, CasinoNightZone.Act11, AquaticRuinZone.Act11, ChemicalPlantZone.Act11, HillTopZone.Act11, EmeraldHillZone.Act11
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
parser.add_argument("--save_dir",type=str,default="ivg_models")
parser.add_argument("--load",action="store_true")
parser.add_argument("--train_frac",type=float,default=0.8)
parser.add_argument("--val_frac",type=float,default=0.1)
parser.add_argument("--validation_interval",type=int,default=10)
parser.add_argument("--limit",type=int,default=-1)

WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
CONFIG_NAME="config.json"

def concat_images_horizontally(images):
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
    save_dir=os.path.join(args.save_dir,args.name.split("/")[1])
    print("save dir",save_dir)
    os.makedirs(save_dir,exist_ok=True)
    unet_save_path=os.path.join(save_dir,"unet",WEIGHTS_NAME)
    os.makedirs(os.path.join(save_dir,"unet"),exist_ok=True)
    action_embedding_save_path=os.path.join(save_dir,"action_embedding",WEIGHTS_NAME)
    os.makedirs(os.path.join(save_dir,"action_embedding"),exist_ok=True)
    config_path=os.path.join(save_dir,CONFIG_NAME)
    if args.metadata_keys is not None:
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

        pretrained_weights_path=api.hf_hub_download(args.vae_checkpoint,"diffusion_pytorch_model.safetensors",force_download=True)
        vae.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)
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
        
        unet.class_embedding=torch.nn.Embedding(10,unet.time_embedding.linear_2.out_features,device=accelerator.device)
        unet.class_embedding.requires_grad_(True)
        #the class embedding says which of the 10 discrete noise levels we apply to the past data
        
        
        dataset=MovieImageFolderFromHF(args.hf_training_data,args.lookback,args.use_prior)
        train_frac=int(args.train_frac*len(dataset))
        val_frac=int(args.val_frac * len(dataset))
        test_frac=len(dataset)-(train_frac+val_frac)
        print("train, val, test",[train_frac,val_frac,test_frac])
        dataset,val_dataset,test_dataset=random_split(dataset,[train_frac,val_frac,test_frac])
        loader=DataLoader(dataset,args.batch_size,shuffle=True)
        val_loader=DataLoader(val_dataset,args.batch_size)
        test_loader=DataLoader(test_dataset,args.batch_size)
        action_embedding=torch.nn.Embedding(args.n_actions,768*args.n_action_tokens,device=accelerator.device)
        accelerator.print(f" each embedding = 768 * {args.n_action_tokens} ={768*args.n_action_tokens} ")
        

        start_epoch=1
        if args.load:
            past_unet_state_dict=torch.load(unet_save_path, weights_only=True)
            past_action_embedding_state_dict=torch.load(action_embedding_save_path,weights_only=True)
            unet.load_state_dict(past_unet_state_dict)
            action_embedding.load_state_dict(past_action_embedding_state_dict)
            with open(config_path,"r") as f:
                data=json.load(f)
                start_epoch=data["start_epoch"]
            print("loaded from local checkpoint")


        
        '''for name, module in unet.attn_processors.items():
            for param in module.attn.parameters():
                param.requires_grad = True'''
        unet.conv_in.requires_grad_(True)
        unet.class_embedding.requires_grad_(True)
        action_embedding.requires_grad_(True)
        params=[p for p in unet.parameters() if p.requires_grad]
        params+=[p for p in action_embedding.parameters()]
        for batch in loader:
            break
        print("params",len(params))
        print("posertior size",batch["posterior"].size())

        optimizer=torch.optim.AdamW(params,args.lr)

        optimizer,unet,loader,action_embedding=accelerator.prepare(optimizer,unet,loader,action_embedding)

        '''@torch.no_grad()
        def logging(unet,loader):'''

        


        for e in range(start_epoch,args.epochs+1):
            start=time.time()
            loss_buffer=[]
            for _b,batch in enumerate(loader):
                if _b==args.limit:
                    break
                with accelerator.accumulate(params):
                    latent=batch["posterior"].to(device)
                    action=batch["action"]
                    if e==1 and _b==0:
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

                    if scheduler.config.prediction_type == "epsilon":
                        target = noise[:, - 4:, :, :] 
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(noised_latent[:,  - 4:, :, :] , noise[:, - 4:, :, :] , last_timestep)
                    encoder_hidden_states=action_embedding(action).reshape(B,args.n_action_tokens ,-1)
                    if _b==0 and e==start_epoch:
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
                    if _b==0 and e==1:
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
            val_loss_buffer=[]
            if e % args.validation_interval ==0:
                start=time.time()
                with torch.no_grad():
                    for _b,batch in enumerate(val_loader):
                        if _b==args.limit:
                            break
                        latent=batch["posterior"].to(device)
                        action=batch["action"]
                        skip_num=batch["skip_num"]
                        (B,C,H,W)=latent.size()
                        num_chunks=C//4
                        latent_chunks = latent.view(B, num_chunks, 4, H, W)
                        noise_chunks = torch.randn_like(latent_chunks)
                    
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

                        if scheduler.config.prediction_type == "epsilon":
                            target = noise[:, - 4:, :, :] 
                        elif scheduler.config.prediction_type == "v_prediction":
                            target = scheduler.get_velocity(noised_latent[:,  - 4:, :, :] , noise[:, - 4:, :, :] , last_timestep)
                        encoder_hidden_states=action_embedding(action).reshape(B,args.n_action_tokens ,-1)

                        model_pred=unet(noised_latent,last_timestep,encoder_hidden_states=encoder_hidden_states,
                                    class_labels=class_labels,
                                    return_dict=False)[0] #somehow condiiton on main_timesteps ???

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        val_loss_buffer.append(loss.cpu().detach().item())

                        decoded_target=vae.decode(target)
                        decoded_pred=vae.decode(model_pred)
                        do_denormalize=B*[True]
                        decoded_target_pil=image_processor.postprocess(decoded_target,do_denormalize=do_denormalize)
                        decoded_pred_pil=image_processor.postprocess(decoded_pred,do_denormalize=do_denormalize)

                        for i, (real,fake) in enumerate(zip(decoded_target_pil,decoded_pred_pil)):
                            concat=concat_images_horizontally([real,fake])
                            accelerator.log({
                                f"{_b}_{i}_validation":wandb.Image(concat)
                            })
                    end=time.time()
                    elapsed=end-start
                    accelerator.print(f"\t validation epoch {e} elapsed {elapsed}")
                    accelerator.log({
                        "val_loss_mean":np.mean(val_loss_buffer),
                        "val_loss_std":np.std(val_loss_buffer),
                    })

        with torch.no_grad():
            test_loss_buffer=[]
            psnr_buffer=[]
            psnr_metric=PeakSignalNoiseRatio()
            lpips_buffer=[]
            loss_fn_alex = LPIPS(net='alex') # best forward scores
            for _b, batch in enumerate(test_loader):
                if _b==args.limit:
                    break
                latent=batch["posterior"].to(device)
                action=batch["action"]
                skip_num=batch["skip_num"]
                (B,C,H,W)=latent.size()
                num_chunks=C//4
                latent_chunks = latent.view(B, num_chunks, 4, H, W)
                noise_chunks = torch.randn_like(latent_chunks)

                main_timesteps=torch.zeros((B,), device=latent.device)


                class_labels=main_timesteps//100
                class_labels=class_labels.int()

                last_timestep = torch.randint(
                    0, scheduler.config.num_train_timesteps, (B,), device=latent.device
                )

                # Run per chunk
                noised_latent_chunks = []

                for i in range(num_chunks):
                    latent_i = latent_chunks[:, i]   # (B, 4, H, W)
                    noise_i = noise_chunks[:, i]

                    if i == num_chunks - 1:
                        t_i = last_timestep    # (B,)
                    else:
                        t_i = main_timesteps    # (B,)

                    if drop and i != num_chunks-1:
                        noised_i=torch.zeros(noise_i.size(),device=device)
                    else:
                        noised_i = scheduler.add_noise(latent_i, noise_i, t_i)  # (B, 4, H, W)
                    noised_latent_chunks.append(noised_i)

                # Reassemble
                noised_latent_chunks = torch.stack(noised_latent_chunks, dim=1)           # (B, num_chunks, 4, H, W)
                noised_latent = noised_latent_chunks.view(B, C, H, W)              # (B, C, H, W)
                noise=noise_chunks.view(B,C,H,W)

                if scheduler.config.prediction_type == "epsilon":
                    target = noise[:, - 4:, :, :] 
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(noised_latent[:,  - 4:, :, :] , noise[:, - 4:, :, :] , last_timestep)
                encoder_hidden_states=action_embedding(action).reshape(B,args.n_action_tokens ,-1)

                model_pred=unet(noised_latent,last_timestep,encoder_hidden_states=encoder_hidden_states,
                            class_labels=class_labels,
                            return_dict=False)[0] #somehow condiiton on main_timesteps ???

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                test_loss_buffer.append(loss.cpu().detach().item())

                decoded_target=vae.decode(target)
                decoded_pred=vae.decode(model_pred)
                do_denormalize=B*[True]
                decoded_target_pil=image_processor.postprocess(decoded_target,do_denormalize=do_denormalize)
                decoded_pred_pil=image_processor.postprocess(decoded_pred,do_denormalize=do_denormalize)

                for i, (real,fake) in enumerate(zip(decoded_target_pil,decoded_pred_pil)):
                    concat=concat_images_horizontally([real,fake])
                    accelerator.log({
                        f"{_b}_{i}_test":wandb.Image(concat)
                    })

                for real_tensor,fake_tensor in zip(decoded_target,decoded_pred):
                    psnr_metric.update(fake_tensor,real_tensor)
                    psnr=psnr_metric.compute().cpu().item()
                    psnr_buffer.append(psnr)

                lpips_loss=loss_fn_alex(real_tensor,fake_tensor)
                lpips_loss=lpips_loss.squeeze(1).squeeze(1).squeeze(1).cpu().detach().numpy().tolist()
                lpips_buffer=lpips_buffer+lpips_loss

            test_metrics={
                    "test_loss_mean":np.mean(test_loss_buffer),
                    "test_loss_std":np.std(test_loss_buffer),
                    "psnr_mean":np.mean(psnr_buffer),
                    "psnr_std":np.std(psnr_buffer),
                }
            accelerator.log(test_metrics)
            

                




            unet_state_dict={name: param for name, param in unet.named_parameters() if param.requires_grad}
            print("state dict len",len(unet_state_dict))
            torch.save(unet_state_dict,unet_save_path)
            action_embedding_state_dict=action_embedding.state_dict()
            print("state dict len",len(action_embedding_state_dict))
            torch.save(action_embedding_state_dict,action_embedding_save_path)
            config={
                "start_epoch":e+1
            }
            with open(config_path,"w+") as config_file:
                json.dump(config,config_file, indent=4)


    

                



    


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