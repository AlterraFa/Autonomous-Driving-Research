import os, sys
import torch

from tqdm.auto import tqdm
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from model.PilotNet.WRN import BlockStack, ConvNeXt, ResnetBlock

from typing import Literal
from utils.messages.logger import Logger

log = Logger()
class PilotNetStatic(nn.Module):
    def __init__(self, mode: Literal["steer", "waypoint"] = "steer", *input_shape, num_waypoints: int = 1, num_cmd: int = 1, droprate = 0.1):
        super().__init__()
        self.input_metadata = {}
        for i, shape in enumerate(input_shape):
            self.input_metadata.update({f"I{i}": shape})

        self.output_names = [mode] 

        self.residual1 = BlockStack(ResnetBlock, 3, 48, 2, droprate, 3)
        self.residual2 = BlockStack(ResnetBlock, 48, 72, 2, droprate, 3)
        self.residual3 = BlockStack(ResnetBlock, 72, 96, 2, droprate, 3)
        self.residual4 = BlockStack(ResnetBlock, 96, 128, 2, droprate, 3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 192, 3, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.flatten = nn.Flatten()
        
        self.fc_stem = nn.Sequential(
            nn.LazyLinear(256),
            nn.LeakyReLU()
        )

        self.mode = mode
        if mode == "steer":
            self.output_dim = 1
        elif mode == "waypoint":
            self.output_dim = 2 * num_waypoints
        cmd_branch = []
        for _ in range(num_cmd):
            cmd_branch += [nn.Sequential(
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Dropout(droprate),
                
                
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Linear(128, 96),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Linear(96, self.output_dim),
                nn.Identity() if self.mode == "waypoint" else nn.Tanh()
            )]
        self.cmd_branch = nn.ModuleList(cmd_branch)
        
    def forward(self, I0, branch: int = -1):
        out = self.residual1(I0)
        out = self.residual2(out)
        out = self.residual3(out)
        out = self.residual4(out)

        out = self.conv1(out)
        out = self.conv2(out)
        
        out = self.flatten(out)
        out = self.fc_stem(out)

        if torch.onnx.is_in_onnx_export():
            outputs = [b(out) for b in self.cmd_branch]
            return torch.stack(outputs, dim=1)  # shape: (B, num_branches, output_dim)

        if not self.training:
            out = self.cmd_branch[branch + 1](out)
            if self.mode == "waypoint":
                return out.view(-1, self.output_dim // 2, 2)  # (B, num_wp, 2)
            else:
                return out
        else:
            final_out = torch.zeros((I0.size(0), self.output_dim),
                                    device=I0.device, dtype=out.dtype)
            for b in torch.unique(branch):
                idxs = (branch == b).nonzero(as_tuple=True)[0]
                out_b = self.cmd_branch[int(b) + 1](out[idxs])
                final_out[idxs] = out_b

            if self.mode == "waypoint":
                return final_out.view(-1, self.output_dim // 2, 2)
            else:
                return final_out
    
    @staticmethod
    def postprocessor(raw_out: dict, data):
        control_type = list(raw_out.keys())[0]
        log.INFO(message = f"Using control type: [bold]{control_type}[/]", once = True)
        return raw_out[control_type][:, data + 1][0, 0]

def single_epoch_training_static(model: PilotNetStatic, mode: Literal["steer", "waypoint"], loader: DataLoader, criterion: nn, optimizer: optim, l1 = 0.0, l2 = 0.0):
    model.train()
    device = next(model.parameters()).device

    trainBar = tqdm(loader, desc = "Train", position = 1, leave = False)
    trainMetrics = {"Total": 0, "Supervised": 0}
    for images, true_waypoints, controls, turn_signals in trainBar:
        optimizer.zero_grad(set_to_none = True)

        images       = images.to(device)
        gt           = true_waypoints.to(device) if mode == "waypoint" else controls[:, 0].unsqueeze(1).to(device)
        turn_signals = turn_signals.to(device)

        pred = model(images, turn_signals)        
        
        weightParams = [p for n, p in model.named_parameters()
                        if p.requires_grad and "weight" in n]
        l1Norm = sum(p.abs().mean() for p in weightParams)
        l2Norm = sum(p.pow(2.0).mean() for p in weightParams)
        
        supervised_loss = criterion(pred, gt)

        loss = supervised_loss + \
            l1Norm * l1 + \
            l2Norm * l2

        loss.backward()
        optimizer.step()

        trainMetrics["Total"]      += loss.item()
        trainMetrics["Supervised"] += supervised_loss.item()
        
        trainBar.set_postfix({
            "T": f"{trainMetrics['Total']/ (trainBar.n+1):.3f}",
            "S": f"{trainMetrics['Supervised']/(trainBar.n+1):.3f}",
        })

    trainMetrics["Total"]       /= len(loader)
    trainMetrics["Supervised"]  /= len(loader)
    
    del images, true_waypoints, turn_signals, pred, supervised_loss, gt
    torch.cuda.empty_cache()

    return trainMetrics

def single_epoch_val_static(model: PilotNetStatic, mode: Literal["steer", "waypoint"], loader: DataLoader, criterion: nn, l1 = 0, l2 = 0):
    model.eval()
    device = next(model.parameters()).device

    valBar = tqdm(loader, desc = "Val", position = 2, leave = False)
    valMetrics = {"Total": 0, "Cost": 0}
    with torch.no_grad():
        for images, true_waypoints, controls, turn_signals in valBar:

            images       = images.to(device)
            
            
            gt           = true_waypoints.to(device) if mode == "waypoint" else controls[:, 0].unsqueeze(1).to(device)
            turn_signals = turn_signals.to(device)

            pred = model(images, turn_signals)        
            supervised_loss = criterion(pred, gt)

            weightParams = [p for n, p in model.named_parameters()
                            if p.requires_grad and "weight" in n]
            l1Norm = sum(p.abs().mean() for p in weightParams)
            l2Norm = sum(p.pow(2.0).mean() for p in weightParams)
            loss = supervised_loss + \
                l1Norm * l1 + \
                l2Norm * l2

            valMetrics["Cost"]  += supervised_loss.item()
            valMetrics["Total"] += loss.item()
            valBar.set_postfix({
                "Cost": f"{valMetrics['Cost'] / (valBar.n+1):.3f}",
                "Total": f"{valMetrics['Total'] / (valBar.n+1):.3f}"
            })

    valMetrics['Cost'] /= len(loader)
    valMetrics['Total'] /= len(loader)

    del images, true_waypoints, turn_signals, pred, supervised_loss, gt
    torch.cuda.empty_cache()

    return valMetrics

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