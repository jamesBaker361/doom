from torch import nn
import torch
import copy
from diffusers import AutoencoderKL,AutoencoderDC

class PrintModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        print(x.size())
        return x
    
class AgentNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)

class ConvAgentNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + LeakyReLU) x 3 -> flatten -> (dense + LeakyReLU) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(4),
            nn.Flatten(),
        )
        
        result=self.online_conv(torch.zeros((1,c,h,w)))
        dim=1
        for r in result.size():
            dim*=r
            
        print(f" {(c,h,w)} -> {dim} ")
        self.online_dense=nn.Sequential(
            nn.Linear(int(dim), 1024),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        )
        
        self.online=torch.nn.Sequential(
            #PrintModule(),
            self.online_conv,
            #PrintModule(),
            self.online_dense)

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
            
        self.module_list=torch.nn.ModuleList([self.online,self.target])
        
    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)



class AEKLAgentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder=AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",subfolder="vae")
        self.encoder.requires_grad_(False)
        c, h, w = input_dim
        self.online_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        
        result=torch.zeros((1,c,h,w))
        result=self.encoder.encode(result).latent_dist.sample()
        result=self.online_conv(result)
        dim=1
        for r in result.size():
            dim*=r
            
        print(f" {(c,h,w)} -> {dim} ")
        
        self.online_dense=nn.Sequential(
            nn.Linear(int(dim), 1024),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        )
        
        self.online=torch.nn.Sequential(
            #PrintModule(),
            self.online_conv,
            #PrintModule(),
            self.online_dense)

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
            
        self.module_list=torch.nn.ModuleList([self.encoder,self.online,self.target])
        
    def forward(self, input, model):
        input=self.encoder.encode(input).latent_dist.sample()
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)
        
        
class AEDCAgentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
        self.encoder.requires_grad_(False)
        c, h, w = input_dim
        self.online_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        
        result=torch.zeros((1,c,h,w))
        result=self.encoder.encode(result).latent
        result=self.online_conv(result)
        dim=1
        for r in result.size():
            dim*=r
            
        print(f" {(c,h,w)} -> {dim} ")
        
        self.online_dense=nn.Sequential(
            nn.Linear(int(dim), 1024),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        )
        
        self.online=torch.nn.Sequential(
            #PrintModule(),
            self.online_conv,
            #PrintModule(),
            self.online_dense)

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
            
        self.module_list=torch.nn.ModuleList([self.encoder,self.online,self.target])
        
    def forward(self, input, model):
        input=self.encoder.encode(input).latent
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)