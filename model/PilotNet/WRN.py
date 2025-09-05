import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.layers.drop import DropPath

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0): 
        # Adding group conv won't work well since there's no mixing mechanism in this
        # I need seperate Resnext block with a 1x1->3x3->1x1 not a 3x3->3x3
        super(ResnetBlock, self).__init__()
        
        self.dropout = dropout
        
        self.bn1 = nn.BatchNorm2d(in_channels, momentum = 0.01)
        self.LeakyReLU1 = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum = 0.01)
        self.LeakyReLU2 = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.InIsOut = in_channels == out_channels
        self.shortcutCompat = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, bias = False) if not self.InIsOut else nn.Identity() 
        
    def forward(self, x):
        if not self.InIsOut:
            x = self.LeakyReLU1(self.bn1(x))
        else:
            out = self.LeakyReLU1(self.bn1(x))

        
        out = self.LeakyReLU2(self.bn2(self.conv1(out if self.InIsOut else x)))
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training = self.training)
            
        out = self.conv2(out)
        
        return self.shortcutCompat(x) + out
    

class ConvNeXt(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0, scaleInit: float = 1e-6):
        super(ConvNeXt, self).__init__()
        
        if in_channels != out_channels or stride != 1:
            self.reduction = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = stride, bias = False),
                Rearrange("b c h w -> b h w c"),
                nn.LayerNorm(out_channels),
                Rearrange("b h w c -> b c h w"),
            )
        else:
            self.reduction = nn.Identity()

        self.compute    = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 7, stride = 1, padding = 3, groups = out_channels, bias = True),
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels),
            Rearrange("b h w c -> b c h w"),
        )
        self.layerScale = nn.Parameter(scaleInit * torch.ones(out_channels), 
                                        requires_grad=True)
        self.dropPath   = DropPath(dropout)
            
    def forward(self, x: torch.Tensor):
        x = self.reduction(x)
        
        out = self.compute(x)
        out = self.layerScale.view(1, -1, 1, 1) * out
        
        out = self.dropPath(out)
        
        return x + out


class BlockStack(nn.Module):
    def __init__(self, blockType: nn.Module, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        super(BlockStack, self).__init__() 
        self.group = self.make_group(blockType, in_channels, out_channels, stride, dropout, numBlock)
        
    def make_group(self, blockType: nn.Module, in_channels: int, out_channels: int, stride: int, dropout: float, numBlock: int):
        layers = []
        for idx in range(numBlock):
            if idx == 0:
                layers.append(blockType(in_channels, out_channels, stride, dropout)) # This changes both resolution and channels
            else:
                layers.append(blockType(out_channels, out_channels, 1, dropout))
        
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.group(x)

        

class WRN(nn.Module):
    def __init__(self, blockType: nn.Module, depth: int, widenFact: int, numClasses: int, patchSz: int = 0, dropout: float = 0.0):
        super(WRN, self).__init__()
        assert (depth - 4) % 6 == 0
        numBlock = (depth - 4) // 6

        channelDepth = [16, 16 * widenFact, 32 * widenFact, 64 * widenFact]
        strides = [1, 2, 2]

        self.patchSz = patchSz
        if patchSz != 0:
            self.stem = nn.Conv2d(3, channelDepth[0], kernel_size = patchSz, stride = patchSz) # The actual patchify layer
        else:
            self.stem = nn.Conv2d(3, channelDepth[0], kernel_size = 3, stride = 1) # Almost similar to patchify, but its overlapping kernel => no
        
        self.largeGroup = nn.ModuleList(
            [BlockStack(blockType, channelDepth[i], channelDepth[i + 1], strides[i], dropout, numBlock) for i in range(3)]
        )

        self.bn = nn.BatchNorm2d(channelDepth[-1], momentum = 0.01)
        self.LeakyReLU = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        self.fc = nn.Linear(channelDepth[-1], numClasses)

        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        
    def forward(self, x):
        if self.patchSz != 0 and not torch.jit.is_tracing():
            assert x.shape[-1] % self.patchSz == 0, f"Patch size is enabled but is not divisible by input shape. Redo the model"
        
        x = self.stem(x)
        for group in self.largeGroup:
            x = group(x)
        x = self.LeakyReLU(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def summary(self):
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_MB = total_params * 4 / (1024 ** 2)  # Assuming 32-bit float = 4 bytes
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"Approximate Model Size: {total_MB:.2f} MB")
        