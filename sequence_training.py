import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split
from torch.utils.data import Subset
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

from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi
from sequence_models import BasicRNN, BasicTransformer,BasicCNN
from data_loaders import SequenceDatasetFromHF

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--sequence_dataset",type=str,default="jlbaker361/sonic_hilltop_sequence")
parser.add_argument("--sequence_length",type=int,default=16,help="how many past actions to use to predict")
parser.add_argument("--embedding_dim",type=int,default=256)
parser.add_argument("--model_type",type=str,default="transformer or rnn")
parser.add_argument("--vocab_size",type=int,default=36)
parser.add_argument("--nhead",type=int,default=8)
parser.add_argument("--num_layers",type=int,default=10)
parser.add_argument("--metadata_keys",nargs="*")
parser.add_argument("--rnn_hidden_size",type=int,default=128)
parser.add_argument("--lookback",type=int,default=32)
parser.add_argument("--epochs",type=int,default=50)
parser.add_argument("--train_fraction",type=float,default=0.9)
parser.add_argument("--batch_size",type=int,default=16)
parser.add_argument("--test_interval",type=int,default=10)
parser.add_argument("--limit",type=int,default=-1)

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
        model_type=args.model_type
        if model_type=="transformer":
            model=BasicTransformer(
                args.embedding_dim,
                args.vocab_size,
                args.nhead,
                args.num_layers,
                len(args.metadata_keys)
            )
        elif model_type=="rnn":
            model=BasicRNN(
                args.embedding_dim,
                args.vocab_size,
                args.rnn_hidden_size,
                args.num_layers,
                len(args.metadata_keys)
            )
        elif model_type=="cnn":
            model= BasicCNN(
                args.embedding_dim,
                args.vocab_size,
                args.num_layers,
                len(args.metadata_keys)
            )

        model=model.to(accelerator.device)

        data=SequenceDatasetFromHF(args.sequence_dataset,args.lookback)

        if args.limit!=-1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = Subset(data, indices[:args.limit])

        train_size = int(args.train_fraction * len(data))
        test_size = len(data) - train_size

        # Split the dataset
        train_dataset, test_dataset = random_split(data, [train_size, test_size])

        train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

        parameters=[p for p in model.parameters()]
        optimizer=torch.optim.AdamW(parameters)

        model,optimizer,train_loader,test_loader=accelerator.prepare(model,optimizer,train_loader,test_loader)

        start_epoch=1
        for e in range(start_epoch,args.epochs+1):
            loss_buffer=[]
            start=time.time()
            with accelerator.accumulate(parameters):
                for b, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    action=batch["action_sequence"]
                    if e==start_epoch and b==0:
                        accelerator.print("action",action.size())

                    
                    batches=[batch[k] for k in args.metadata_keys]
                    print(batches)
                    try:
                        target= torch.cat([batch[k] for k in args.metadata_keys],dim=0) #turn b x 1 or into b x n
                        print("target 0 cat", target.size())
                    except:
                        print("0 didnt work")

                    try:
                        target= torch.stack([batch[k] for k in args.metadata_keys],dim=1) #turn b x 1 or into b x n
                        print("target 1 stack", target.size())
                    except:
                        print("1 stack didnt work")


                    target= torch.cat([batch[k] for k in args.metadata_keys],dim=-1) #turn b x 1 or into b x n
                    predicted=model(action)

                    if e==start_epoch and b==0:
                        accelerator.print('target.size()',target.size())
                        accelerator.print("predicted size",predicted.size())

                    loss=F.mse_loss(predicted.float(),target.float(),reduction="mean")

                    accelerator.backward(loss)

                    optimizer.step()

                    loss_buffer.append(loss.cpu().detach().numpy())
            end=time.time()
            elapsed=end-start
            accelerator.print(f"epoch {e} elapsed {elapsed} seconds")
            accelerator.log({
                "loss":np.mean(loss_buffer)
            })

            if e % args.test_interval == 1:
                #testing frequently because we are cheating :)
                test_loss_buffer=[]
                with torch.no_grad():
                    for b,batch in enumerate(test_loader):
                        action=batch["action_sequence"]

                        target= torch.cat([batch[k] for k in args.metadata_keys],dim=1) #turn b x 1 or into b x n
                        predicted=model(action)

                        test_loss=F.mse_loss(predicted.float(),target.float(),reduction="mean")

                        test_loss_buffer.append(test_loss.cpu().detach().numpy())

                accelerator.log({
                    "test_loss":np.mean(test_loss_buffer)
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