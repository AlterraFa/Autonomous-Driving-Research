import os, sys
import yaml

import torch
import torch.nn as nn


class WaypointPredictor(nn.Module):
    def __init__(self, ):
        super().__init__()
    ...


if __name__ == "__main__":
    target_wrp = torch.load("./JEPA_ACT/Experiment/pretraining/run4/weights/best_target.pt", map_location = "cuda", weights_only = False)
    target_enc = target_wrp.backbone
    device = torch.device('cuda')
    
    dummy_inp = torch.rand((1, 3, 16, 224, 224)).to(device)
    
    print(target_enc(dummy_inp).transpose(1, 2).shape)