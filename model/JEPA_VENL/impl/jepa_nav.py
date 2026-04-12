import torch
import numpy as np
import cv2

from torch import nn
from model.JEPA_VENL.impl.transformer import ViTEncode
from model.JEPA_VENL.impl.venl import SingleVENL
from model.JEPA_VENL.data_utils.jepa_nav.image_transform import Normalization, Augment, v2
from utils.messages.logger import Logger

from typing import Any

class UnifiedJEPANav(nn.Module):
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["log"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.log = Logger()
    
    def __init__(self, backbone: ViTEncode, readout: SingleVENL, full_finetune: bool = False):
        super().__init__()
        self.log = Logger()
        
        self.backbone = backbone
        self.readout  = readout
        
        # -- Register metadata for tensorrt compiling
        self.input_metadata: dict = self.readout.input_metadata.copy()
        self.input_metadata["I0"] = (1, 3, *self.backbone.base_size)
        self.output_names         = self.readout.output_names 

        for param in self.backbone.parameters():
            param.requires_grad = full_finetune
        
        self.full_finetune = full_finetune
        self.initialized = False
        self._override_shape_check = False
    
        
        self.log.WARNING(f"Full finetune mode is {'[green]enabled[/]' if full_finetune else '[red]disabled[/]'}")
        
    def initialize_module(self, I0: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor):
        if self.initialized == False:
            self.initialized = True
            self.readout.initialized = True
            self.readout._override_shape_check = True

            self.forward(I0, MU, MR) 
            self.readout._init_weights()
            
            self.log.INFO("Layer initialized")
        else:
            self.log.WARNING("Layer already initialized")
    
    def train(self, mode=True):
        """
        Override train mode to keep backbone in eval mode.
        """
        super().train(mode)
        if mode:
            if self.full_finetune: self.backbone.train()
            else                 : self.backbone.eval()
        return self
    
    def _match_shape(self, actual_shape, expected_shape):
        if len(actual_shape) != len(expected_shape):
            return False

        for i in range(1, len(expected_shape)):
            dim_expected = expected_shape[i]
            dim_actual = actual_shape[i]

            if dim_expected is Any: continue
            
            if dim_expected is None: continue

            if dim_expected != dim_actual: return False
        
        return True
    
    def _shape_security(self, argnames, local_var):
        for name in argnames[1: ]: # skip self
            tensor = local_var[name]
            expected_shape = self.input_metadata.get(name)
            # -- No shape specified
            if expected_shape is None: 
                self.log.WARNING(f"Layer `{name}` has no input metadata specified", once = True)
                continue 
            if not self._match_shape(tensor.shape, expected_shape):
                expected_str = str([
                    "Any" if x is Any else x for x in expected_shape[1:]
                ])
                
                self.log.ERROR(
                    f"Input tensor '{name}' has shape {list(tensor.shape)[1:]}, "
                    f"expected {expected_str}", 
                    exit_code = 12
                )
        
    def forward(self, I0: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor):
        argcount = self.forward.__code__.co_argcount
        argnames = self.forward.__code__.co_varnames[: argcount]

        if self.initialized == False:
            self.log.ERROR(f"Modules not initialized", exit_code = -1)
        
        if not torch.onnx.is_in_onnx_export() and not self._override_shape_check:
            self._shape_security(argnames, locals())
        
        x = self.backbone(I0)
        x = self.readout(x, MU, MR)

        return x
    
    @staticmethod
    def postprocessor(raw_out: dict, data):
        return tuple([output[0] for output in raw_out.values()])

    def preprocessor(self, **images):
        
        missing_keys = [key for key in self.input_metadata.keys() if key not in images]
        if missing_keys:
            self.log.ERROR(f"Missing keys: {missing_keys}", exit_code = 2)
        
        if not hasattr(self, "main_transform"):
            import yaml
            with open(self.config_path, "r") as f:
                args = yaml.safe_load(f)
            
            augment = args['data']['augmentations']
            map_size = args['model']['map_shape']
            self.main_transform = Augment(
                dimension = augment['image_size'],
                crop = augment['crop'],
                normalization = augment['normalization']
            )
            self.aux_transform  = Normalization(size = map_size)
            self.to_tensor = v2.ToTensor()

        MU = cv2.resize(images["MU"], (self.input_metadata["MU"][3], self.input_metadata["MU"][2]))[..., None]
        MR = cv2.resize(images["MR"], (self.input_metadata["MR"][3], self.input_metadata["MR"][2]))
        I0 = cv2.cvtColor(images['I0'], cv2.COLOR_RGB2BGR)

            
        I0 = self.to_tensor(I0)
        MU = self.to_tensor(MU)
        MR = self.to_tensor(MR)
        
        I0 = self.main_transform(I0).unsqueeze(0)
        MU = self.aux_transform(MU).unsqueeze(0)
        MR = self.aux_transform(MR).unsqueeze(0)
        
        # print(I0.shape)
        # print(MU.shape)
        # print(MR.shape)
        
        return (I0, MU, MR)
        