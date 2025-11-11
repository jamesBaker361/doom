#https://medium.com/@lukasbierling/recurrent-state-space-models-pytorch-implementation-ba5d7e063d11
from torch import nn
import torch
from typing import Tuple
import torch.nn.functional as F
import math

class EncoderCNN(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 2048, input_shape: Tuple[int, int] = (128, 128),n_layers:int=5):
        super(EncoderCNN, self).__init__()
        '''self.conv0=nn.Conv2d(in_channels,16,  kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)'''
        channel_list=[in_channels]+[2**i for i in range(4, 4+n_layers+1)]
        self.conv_list=[nn.Conv2d(channel_list[i], channel_list[i+1],kernel_size=3, stride=2, padding=1) for i in range(len(channel_list)-1)]

        self.fc1 = nn.Linear(self._compute_conv_output((in_channels, input_shape[0], input_shape[1])), embedding_dim)

        '''self.bn0=nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)'''
        self.bn_list=[nn.BatchNorm2d(c) for c in channel_list[1:]]

    def _compute_conv_output(self, shape: Tuple[int, int, int]):
        with torch.no_grad():
            x = torch.randn(1, shape[0], shape[1], shape[2])
            for layer in self.conv_list:
                x=layer(x)

            return x.shape[1] * x.shape[2] * x.shape[3]


    def forward(self, x):
        for conv,bn in zip(self.conv_list,self.bn_list):
            x=conv(x)
            x=torch.relu(x)
            x=bn(x)
           
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class DecoderCNN(nn.Module):
    def __init__(self, hidden_size: int, state_size: int,  embedding_size: int,
                 use_bn: bool = True, output_shape: Tuple[int, int] = (3, 128, 128),n_layers:int=5):
        super(DecoderCNN, self).__init__()


        self.output_shape = output_shape

        self.embedding_size = embedding_size

        channel_list=[output_shape[0]]+[2**i for i in range(4, 4+n_layers+1)]
        channel_list=channel_list[::-1]

        self.fc1 = nn.Linear(hidden_size + state_size, embedding_size)
        dim=channel_list[0]
        self.dim=dim
        print(embedding_size, int(dim * (output_shape[1] // math.sqrt(dim)) * (output_shape[2] // math.sqrt(dim))) )
        self.fc2 = nn.Linear(embedding_size, int(dim * (output_shape[1] // math.sqrt(dim)) * (output_shape[2] // math.sqrt(dim))) )
        

        self.convt_list=[nn.ConvTranspose2d(channel_list[i],channel_list[i+1],kernel_size=3, stride=2, padding=1, output_padding=1) for i in range(len(channel_list)-1)]
        self.bn_list=[nn.BatchNorm2d(c) for c in channel_list[1:]]

        '''self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv4 = nn.ConvTranspose2d(32, output_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)'''

        self.use_bn = use_bn


    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        x = x.view(-1, self.dim, self.output_shape[1] // math.sqrt(self.dim), self.output_shape[2] // math.sqrt(self.dim))

        for convt,bn in zip(self.convt_list, self.bn_list):
            x=convt(x)
            if self.use_bn:
                x=bn(x)
            x=torch.relu(x)

        return x    
    

class DynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, state_dim: int, embedding_dim: int, 
                 rnn_layer: int = 1,
                 metadata_embedding_dim:int=0,metadata_dim:int=0):
        super(DynamicsModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.use_metadata=False
        #metadata embedding
        if metadata_embedding_dim>0:
            self.metadata_project=nn.Linear(metadata_dim,metadata_embedding_dim)
            self.use_metadata=True
        
        # Can be any recurrent network
        self.rnn = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rnn_layer)])
        
        # Projection layer to make efficient use of concatenated inputs
        self.project_state_action = nn.Linear(action_dim + state_dim+metadata_embedding_dim, hidden_dim)
        
        # Return mean and log-variance of the normal distribution
        self.prior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_action = nn.Linear(hidden_dim + action_dim, hidden_dim)
        
        # Return mean and log-variance of the normal distribution
        self.posterior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_obs = nn.Linear(hidden_dim + embedding_dim+metadata_embedding_dim, hidden_dim)

        self.state_dim = state_dim

        self.act_fn = nn.ReLU()

    def forward(self, prev_hidden: torch.Tensor, 
                prev_state: torch.Tensor, 
                actions: torch.Tensor,
                stop:list,
                obs: torch.Tensor = None, 
                metadata:torch.Tensor=None,
                #dones: torch.Tensor = None
                ):
        """
        Forward pass of the dynamics model for one time step.
        :param prev_hidden: Previous hidden state of the RNN: (batch_size, hidden_dim)
        :param prev_state: Previous stochastic state: (batch_size, state_dim)
        :param action: One hot encoded actions: (sequence_length, batch_size, action_dim)
        :param obs: This is the encoded observation from the encoder, not the raw observation!: (sequence_length, batch_size, embedding_dim)
        :param dones: Terminal states of the environment
        :return: 
        """
        B, T, _ = actions.size() # They are crucial to to infernece without access to observations

        hiddens_list = []
        posterior_means_list = []
        posterior_logvars_list = []
        prior_means_list = []
        prior_logvars_list = []
        prior_states_list = []
        posterior_states_list = []
        
        # (B, 1, hidden_dim)
        hiddens_list.append(prev_hidden.unsqueeze(1))  
        prior_states_list.append(prev_state.unsqueeze(1))
        posterior_states_list.append(prev_state.unsqueeze(1))

        for t in range(T - 1):
            ### Combine the state and action ###
            if self.use_metadata:
                metadata_t=metadata[:,t,:]
            action_t = actions[:, t, :]
            obs_t = obs[:, t, :] if obs is not None else torch.zeros(B, self.embedding_dim, device=actions.device)
            state_t = posterior_states_list[-1][:, 0, :] if obs is not None else prior_states_list[-1][:, 0, :]
            #state_t = state_t #if dones is None else state_t * (1 - dones[:, t, :])
            hidden_t = hiddens_list[-1][:, 0, :]
            
            state_action = torch.cat([state_t, action_t], dim=-1)
            if self.use_metadata:
                state_action= torch.cat([state_t, action_t,metadata_t], dim=-1)
            state_action = self.act_fn(self.project_state_action(state_action))

            ### Update the deterministic hidden state ###
            for i in range(len(self.rnn)):
                hidden_t = self.rnn[i](state_action, hidden_t)

            ### Determine the prior distribution ###
            hidden_action = torch.cat([hidden_t, action_t], dim=-1)
            hidden_action = self.act_fn(self.project_hidden_action(hidden_action))
            prior_params = self.prior(hidden_action)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            ### Sample from the prior distribution ###
            prior_dist = torch.distributions.Normal(prior_mean, torch.exp(F.softplus(prior_logvar)))
            prior_state_t = prior_dist.rsample()

            ### Determine the posterior distribution ###
            # If observations are not available, we just use the prior
            if obs is None:
                posterior_mean = prior_mean
                posterior_logvar = prior_logvar
            else:
                hidden_obs = torch.cat([hidden_t, obs_t], dim=-1)
                if self.use_metadata:
                    hidden_obs = torch.cat([hidden_t, obs_t,metadata_t], dim=-1)
                hidden_obs = self.act_fn(self.project_hidden_obs(hidden_obs))
                posterior_params = self.posterior(hidden_obs)
                posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)

            ### Sample from the posterior distribution ###
            posterior_dist = torch.distributions.Normal(posterior_mean, torch.exp(F.softplus(posterior_logvar)))
            
            # Make sure to use rsample to enable the gradient flow
            # Otherwise you could also use code the reparameterization trick by hand
            posterior_state_t = posterior_dist.rsample()

            ### Store results in lists (instead of in-place modification) ###
            posterior_means_list.append(posterior_mean.unsqueeze(1))
            posterior_logvars_list.append(posterior_logvar.unsqueeze(1))
            prior_means_list.append(prior_mean.unsqueeze(1))
            prior_logvars_list.append(prior_logvar.unsqueeze(1))
            prior_states_list.append(prior_state_t.unsqueeze(1))
            posterior_states_list.append(posterior_state_t.unsqueeze(1))
            hiddens_list.append(hidden_t.unsqueeze(1))

        # Convert lists to tensors using torch.cat()
        hiddens = torch.cat(hiddens_list, dim=1)
        prior_states = torch.cat(prior_states_list, dim=1)
        posterior_states = torch.cat(posterior_states_list, dim=1)
        prior_means = torch.cat(prior_means_list, dim=1)
        prior_logvars = torch.cat(prior_logvars_list, dim=1)
        posterior_means = torch.cat(posterior_means_list, dim=1)
        posterior_logvars = torch.cat(posterior_logvars_list, dim=1)

        print("hiddens size",hiddens.size())
        print('hidden t size',hiddens.size())

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars
    
class PhysicalDynamicsModel(nn.Module):
    #given metadata and embedding, predict net forces on sonic using network, 
    # also have parameters (learnable) to represent g, friction, etc
    # might need a regularization term too
    # were going to model the world as if the rigid body is 1 kg
    # so there's internal force (running?) 
    # normal force (which isnt always a thing)
    # gravity (always downwards)
    # coefficient of friction with air (probably 0)
    # external forces (like from an enemy)
    # and all this is used to calculate direction of net velocity
    def __init__(self,hidden_dim: int, action_dim: int, state_dim: int, embedding_dim: int, 
                 rnn_layer: int = 1,
                 metadata_embedding_dim:int=0,metadata_dim:int=0, *args, **kwargs):
        super().__init__()
        self.hidden_dim=hidden_dim
        
        
    

class RSSM:
    def __init__(self,
                 encoder: EncoderCNN,
                 decoder: DecoderCNN,
                 #reward_model: RewardModel,
                 dynamics_model: nn.Module,
                 hidden_dim: int,
                 state_dim: int,
                 device: str = "mps"):
        """
        Recurrent State-Space Model (RSSM) for learning dynamics models.

        Args:
            encoder: Encoder network for deterministic state
            prior_model: Prior network for stochastic state
            decoder: Decoder network for reconstructing observation
            sequence_model: Recurrent model for deterministic state
            hidden_dim: Hidden dimension of the RNN
            latent_dim: Latent dimension of the stochastic state
            action_dim: Dimension of the action space
            obs_dim: Dimension of the encoded observation space


        """
        super(RSSM, self).__init__()

        self.dynamics = dynamics_model
        self.encoder = encoder
        self.decoder = decoder
        #self.reward_model = reward_model

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        #shift to device
        self.dynamics.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        #self.reward_model.to(device)


    def generate_rollout(self, actions: torch.Tensor, hiddens: torch.Tensor = None, states: torch.Tensor = None,
                         obs: torch.Tensor = None, stop: list=[],metadata:torch.Tensor=None):

        if hiddens is None:
            hiddens = torch.zeros(actions.size(0), self.hidden_dim).to(actions.device)

        if states is None:
            states = torch.zeros(actions.size(0), self.state_dim).to(actions.device)

        dynamics_result = self.dynamics(hiddens, states, actions,stop,obs,metadata)
        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = dynamics_result

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars

    def train(self):
        self.dynamics.train()
        self.encoder.train()
        self.decoder.train()
        #self.reward_model.train()

    def eval(self):
        self.dynamics.eval()
        self.encoder.eval()
        self.decoder.eval()
        #self.reward_model.eval()

    def encode(self, obs: torch.Tensor):
        return self.encoder(obs)

    def decode(self, state: torch.Tensor):
        return self.decoder(state)

    '''def predict_reward(self, h: torch.Tensor, s: torch.Tensor):
        return self.reward_model(h, s)'''

    def parameters(self):
        return list(self.dynamics.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters()) # + list(self.reward_model.parameters())

    def save(self, path: str):
        torch.save({
            "dynamics": self.dynamics.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
           # "reward_model": self.reward_model.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        #self.reward_model.load_state_dict(checkpoint["reward_model"])

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
from data_loaders import WorldModelDatasetHF
from torch.utils.data import random_split, DataLoader
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from transformers import AutoProcessor, CLIPModel
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
parser.add_argument("--src_dataset",type=str,default="jlbaker361/sonic-vae-preprocessed")
parser.add_argument("--hidden_dim",type=int,default=512)
parser.add_argument("--state_dim",type=int,default=512)
parser.add_argument("--action_dim",type=int,default=128)
parser.add_argument("--embedding_dim",type=int,default=256)
parser.add_argument("--metadata_key_list",nargs="*")
parser.add_argument("--use_metadata",action="store_true")
parser.add_argument("--metadata_embedding_dim",type=int,default=128)
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--n_layers",type=int,default=4)
parser.add_argument("--rnn_layer",type=int,default=2)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--epochs",type=int,default=10)

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

    if args.use_metadata and args.metadata_key_list==None:
        args.metadata_key_list=["score","lives","screen_x","screen_y","x","y"]

    if not args.use_metadata:
        args.metadata_key_list=[]

    dataset=WorldModelDatasetHF(args.src_dataset,DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").image_processor,args.metadata_key_list)

    test_size=args.batch_size * 2
    train_size=len(dataset)-test_size

    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    encoder=EncoderCNN(3,args.embedding_dim,(args.size,args.size),args.n_layers)
    decoder=DecoderCNN(args.hidden_dim,args.state_dim,args.embedding_dim,output_shape=[3,args.size,args.size],n_layers=args.n_layers)
    dynamics=DynamicsModel(args.hidden_dim,args.action_dim,args.state_dim,args.embedding_dim,args.rnn_layer,args.metadata_embedding_dim,len(args.metadata_key_list))

    rssm=RSSM(encoder,decoder,dynamics, args.hidden_dim, args.state_dim,device=device)

    params=[p for p in rssm.parameters()]

    optimizer=torch.optim.Adam(params,args.lr)
    
    rssm,optimizer,train_loader,test_loader=accelerator.prepare(rssm,optimizer,train_loader,test_loader)

    start_epoch=1
    for e in range(start_epoch,args.epochs+1):
        for k, batch in enumerate(train_loader):
            with accelerator.accumulate(params):
                if k==args.limit:
                    break
                batch_size=batch["image"].size()[0]
                stop_index=max(batch["stop"])
                metadata=None
                obs=batch["image"][:stop_index].to(device)
                stop=batch["stop"]
                actions=batch["action"][:stop_index].to(device)
                if args.use_metadata:
                    metadata=batch["metadata"][:stop_index].to(device)

                hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars=rssm.generate_rollout(actions=actions,
                                                                                                                                            stop=stop,obs=obs,metadata=metadata)
                
                hiddens_reshaped = hiddens.reshape(batch_size * stop_index, -1)
                posterior_states_reshaped = posterior_states.reshape(batch_size * stop_index, -1)
                decoded_obs = rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
                decoded_obs = decoded_obs.reshape(batch_size, stop_index, *obs.size()[-3:])

                reconstruction_loss =F.mse_loss(obs,decoded_obs)
                
                prior_dist = Normal(prior_means, torch.exp(prior_logvars))
                posterior_dist = Normal(posterior_means, torch.exp(posterior_logvars))

                kl_loss=kl_divergence(posterior_dist, prior_dist).mean()

                loss=kl_loss+reconstruction_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                





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