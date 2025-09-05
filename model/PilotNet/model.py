import torch

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from model.PilotNet.WRN import BlockStack, ConvNeXt, ResnetBlock

class PilotNetStatic(nn.Module):
    def __init__(self, num_waypoints: int, num_cmd: int, droprate = 0.1):
        super().__init__()
        
        self.residual1 = BlockStack(ResnetBlock, 3, 64, 3, droprate, 3)
        self.residual2 = BlockStack(ResnetBlock, 64, 128, 3, droprate, 3)
        self.residual3 = BlockStack(ResnetBlock, 128, 256, 3, droprate, 3)
        self.residual4 = BlockStack(ResnetBlock, 256, 512, 3, droprate, 3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.flatten = nn.Flatten()
        
        self.fc_stem = nn.Sequential(
            nn.LazyLinear(2048),
            nn.LeakyReLU()
        )
        
        cmd_branch = []
        for _ in range(num_cmd):
            cmd_branch += [nn.Sequential(
                nn.Linear(2048, 1024),
                nn.LeakyReLU(),
                nn.Dropout(droprate),
                
                
                nn.Linear(1024, 256),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Linear(64, num_waypoints * 2),
            )]
        self.cmd_branch = nn.ModuleList(cmd_branch)
        
    def forward(self, x, branch: int = 1):
        out = self.residual1(x)
        out = self.residual2(out)
        out = self.residual3(out)
        out = self.residual4(out)
        
        out = self.conv1(out)
        out = self.conv2(out)
        
        out = self.flatten(out)
        
        out = self.fc_stem(out)
        out = self.cmd_branch[branch](out)
        
        return out

def single_epoch_training_static(model: PilotNetStatic, loader: DataLoader, criterion: nn, optimizer: optim, device: torch.device = torch.device('cpu'), scheduler: lr_scheduler = None):
    model.train()

    for x_batch, y_batch in loader:
        model.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        
        optimizer.step()

class PilotNetDynamic(nn.Module):
    def __init__(self, num_waypoints: int, num_cmd: int, droprate = 0.1):
        super().__init__()

        self.residual1 = BlockStack(ResnetBlock, 3, 64, 3, droprate, 3)
        self.residual2 = BlockStack(ResnetBlock, 64, 128, 3, droprate, 3)
        self.residual3 = BlockStack(ResnetBlock, 128, 256, 3, droprate, 3)
        self.residual4 = BlockStack(ResnetBlock, 256, 512, 3, droprate, 3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.flatten = nn.Flatten()
        
        self.fc_stem = nn.Sequential(
            nn.LazyLinear(2048),
            nn.LeakyReLU()
        )

        self.fc_ctrl = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU()
        ) # Steer, throttle, speed, brake
        
        cmd_branch = []
        for _ in range(num_cmd):
            branch = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.LeakyReLU(),
                    nn.Dropout(droprate)
                ),
                nn.Sequential(
                    nn.Linear(1024 + 32, 256),
                    nn.LeakyReLU(),
                    nn.Dropout(droprate),

                    nn.Linear(256, 64),
                    nn.LeakyReLU(),
                    nn.Dropout(droprate),

                    nn.Linear(64, num_waypoints * 2)
                ) 
            ])
            cmd_branch += [branch]
        self.cmd_branch = nn.ModuleList(cmd_branch)
        
    def forward(self, x: torch.Tensor, branch: int = 0, ctrl: torch.Tensor = torch.tensor([0, 0, 0, 0], dtype = torch.float)[None, ...]):
        out = self.residual1(x)
        out = self.residual2(out)
        out = self.residual3(out)
        out = self.residual4(out)
        
        out = self.conv1(out)
        out = self.conv2(out)
        
        out = self.flatten(out)
        
        out = self.fc_stem(out)
        out = self.cmd_branch[branch][0](out)
        if ctrl.device != out.device:
            ctrl = ctrl.to(out.device)
        out = torch.concat([out, self.fc_ctrl(ctrl)], dim = 1)
        out = self.cmd_branch[branch][1](out)

        return out