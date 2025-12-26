from torch import nn
import torch
import copy

class AgentNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        result=self.online_conv(torch.zeros((1,c,h,w)))
        dim=1
        for r in result.size():
            dim*=r
            
        print(f" {(c,h,w)} -> {dim} ")
        self.online_dense=nn.Sequential(
            nn.Linear(int(dim), 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
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
