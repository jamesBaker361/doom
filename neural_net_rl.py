from torch import nn
import torch
import copy

class AgentNet(nn.Module):
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
            nn.MaxPool2d(2),
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
            nn.Linear(int(dim), 512),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(int(dim), 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(int(dim), 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        )
        
        self.online=torch.nn.Sequential(self.online_conv,self.online_dense)

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
