import cv2
import torch
import queue
import threading
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import warnings


from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from model.PilotNet.WRN import BlockStack, ResnetBlock


from pathlib import Path
from utils.messages.logger import Logger

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

class VENL(nn.Module):
    def __init__(self, droprate: float = 0.1):
        self.log = Logger()
        super().__init__()

        # Might change to WRN to improve performance
        self.multicam_backbone: nn.ModuleList[nn.Sequential] = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(in_channels = 3, out_channels = 24, kernel_size = 5, stride = 2),
                nn.GELU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
                nn.GELU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 3, stride = 2),
                nn.GELU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 2),
                nn.GELU(),
                nn.Dropout(droprate),

                nn.Flatten()
            ]) for _ in range(3)
        ])

        # Shallow network to prevent translational + rotational invariance
        self.unrouted_backbone: nn.Sequential = nn.Sequential(*[
            nn.Conv2d(in_channels = 1, out_channels = 24, kernel_size = 5, stride = 2),
            nn.GELU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
            nn.GELU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 3, stride = 2),
            nn.GELU(),
            nn.Dropout(droprate),

            nn.Flatten()
        ])

        self.routed_backbone: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 24, kernel_size = 5, stride = 2),
            nn.GELU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
            nn.GELU(),
            nn.Dropout(droprate),

            nn.Flatten()
        )

        self.feature_downsize = nn.Sequential(
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Dropout(droprate),

            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Dropout(droprate),
        )

        self.fusion_projector = nn.Sequential(
            nn.LazyLinear(100),
            nn.GELU(),
            nn.Dropout(droprate)
        )
    
        self.initialized = False
        self.droprate = droprate

    def _init_weights(self):
        """Custom weight initialization for all submodules."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["log"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.log = Logger()

    @classmethod
    def steer(cls, camera_shape = (80, 200), map_shape = (50, 50), components: int = 3, droprate = 0.1) -> "VENL":
        self = cls(droprate = droprate)
        self.components = components
        self.log.INFO("Using steer mode")

        self.gmm_head = nn.Linear(200, 3 * components) # 3 gaussian parameters * number of modes
        self.determ_head = nn.Linear(100, 1)
        self.input_metadata = {
            "I0": (1, 3, *camera_shape),
            "I1": (1, 3, *camera_shape),
            "I2": (1, 3, *camera_shape),
            "MU": (1, 1, *map_shape),
            "MR": (1, 3, *map_shape),
        }
        self.output_names = ["steer", "weights", "muy", "sigma"]
        
        return self

    @classmethod
    def waypoint(cls, camera_shape = (80, 200), map_shape = (50, 50), num_waypoints = 1, components: int = 3, droprate = 0.1) -> "VENL":
        self = cls(droprate = droprate)
        self.num_waypoints = num_waypoints
        self.components = components
        self.log.INFO("Using waypoint mode")

        self.gmm_head = nn.Linear(200, components * (1 + num_waypoints * 4)) # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation
        self.determ_head = nn.Linear(100, num_waypoints * 2)
        self.input_metadata = {
            "I0": (1, 3, *camera_shape),
            "I1": (1, 3, *camera_shape),
            "I2": (1, 3, *camera_shape),
            "MU": (1, 1, *map_shape),
            "MR": (1, 3, *map_shape),
        }
        self.output_names = ["waypoint", "weights", "muy", "sigma"]

        return self

    def initialize_module(self, I0: torch.Tensor, I1: torch.Tensor, I2: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor):
        if self.initialized == False:
            self.initialized = True

            self.forward(I0, I1, I2, MU, MR)
            self._init_weights()
            self.log.INFO("Layer initialized")
        else:
            self.log.WARNING("Layer already initialized")
        
        

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, I2: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor) -> torch.Tensor:
        argcount = self.forward.__code__.co_argcount
        argnames = self.forward.__code__.co_varnames[: argcount]

        if self.initialized == False:
            self.log.ERROR(f"Modules not initialized", exit_code = -1)
        
        if not torch.onnx.is_in_onnx_export():
            for name in argnames[1: ]: # skip self
                tensor = locals()[name]
                expected_shape = self.input_metadata.get(name)
                if expected_shape[1:] != tuple(tensor.shape)[1:]:
                    self.log.ERROR(f"Input tensor {name} has shape {tensor.shape[1:]}, expected {expected_shape[1:]}", exit_code = 12)

        # features of multicam setup
        f0 = self.multicam_backbone[0](I0)
        f1 = self.multicam_backbone[1](I1)
        f2 = self.multicam_backbone[2](I2)
        # features of unrouted map
        fmu = self.unrouted_backbone(MU)

        # Concatenation of left, front, right and map features on a single vector
        features_cat = torch.cat([f1, f0, f2, fmu], dim=1) # TENSORRT DOES NOT SUPPORT HSTACK OR VSTACK

        out = self.feature_downsize(features_cat)
        if self.training:
            dropmask = torch.rand(MR.shape[0], device = MR.device) < 0.25
            MR[dropmask] = MU[dropmask].repeat(1, 3, 1, 1)  # randomly drop MR during training
        routed_features = self.routed_backbone(MR)

        gmm_out = self.gmm_head(out)
        determ_in = torch.cat([out, routed_features], dim = 1)
        determ_out = self.determ_head(self.fusion_projector(determ_in))

        if self.output_names[0] == 'waypoint':
            return determ_out.view(-1, self.num_waypoints, 2), *self.extract_gparams(gmm_out)
        else:
            return determ_out, *self.extract_gparams(gmm_out)

    def extract_gparams(self, gmm_params: torch.Tensor):
        if not hasattr(self, "num_waypoints"):
            # predetermined 3 parameters correspond to 3 chunks 
            weights, muy_weights, sigma_weights = torch.chunk(gmm_params, 3, 1)
            weights = torch.softmax(weights, dim=1) 
            muy     = muy_weights                       
            sigma   = torch.exp(sigma_weights)
            return weights, muy, sigma
        else:
            weights, muy_weights, sigma_weights = torch.split(
                gmm_params, 
                [
                    self.components, 
                    self.components * self.num_waypoints * 2, 
                    self.components * self.num_waypoints * 2
                ],  # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation per components
                dim=1
            )
            weights = torch.softmax(weights, dim=1).unsqueeze(-1)
            muy     = muy_weights.view(-1, self.components, self.num_waypoints, 2)
            sigma   = torch.exp(sigma_weights).view(-1, self.components, self.num_waypoints, 2)  # (batch, modes, waypoints, dim)
            return weights, muy, sigma

    @staticmethod
    def postprocessor(raw_out: dict, data):
        return tuple([output[0] for output in raw_out.values()])

    def preprocessor(self, I0: torch.Tensor, I1: torch.Tensor, I2: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor):

        H, W, _    = I0.shape
        x_top_left = 250; x_top_right = W - x_top_left
        y_hor      = 390; y_bot         = 680
        I0 = I0[y_hor: y_bot, x_top_left: x_top_right]
        I0 = cv2.resize(I0, (self.input_metadata["I0"][3], self.input_metadata["I0"][2]))[..., :3]
        I1 = I1[..., :3]
        I2 = I2[..., :3]

        MU = cv2.resize(MU, (self.input_metadata["MU"][3], self.input_metadata["MU"][2]))[..., None]
        MR = cv2.resize(MR, (self.input_metadata["MR"][3], self.input_metadata["MR"][2]))[..., ::-1]

        return (I0, I1, I2, MU, MR)


    def gaussian_function(self, sample, parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        weights, muy, sigma = parameters
        if not hasattr(self, "num_waypoints"):
            try:
                probs_per_components = self._univariate(sample, muy, sigma)
                return weights * probs_per_components  # Return GMM probability per mode with weights
            except Exception as e:
                self.log.ERROR(
                    f"Please check the parameters if it is in the format of univariate or if the sample format is correct. Sample: [bold]{sample.shape}[/], Paramters: [bold]{muy.shape}[/]",
                    full_traceback = e,
                    exit_code = 12

                )
        else:
            try:
                _, branch, *_ = sample.shape
                probs_per_components = self._multivariate(sample, muy, sigma)
                weights = weights.unsqueeze(1).expand(-1, branch, -1, -1)
                return weights * probs_per_components  # returns joint probability of x, y per mode per waypoint
            except Exception as e:
                self.log.ERROR(
                    f"Please check the parameters if it is in the format of multivariate or if the sample format is correct. Sample: [bold]{sample.shape}[/], Parameters: [bold]{muy.shape}[/]",
                    full_traceback = e,
                    exit_code = 12
                )


    @staticmethod
    def _univariate(sample, muy, sigma):
        return (1 / (2 * torch.pi * sigma ** 2) ** 0.5) * torch.exp(-(sample - muy) ** 2 / (2 * sigma ** 2))


    @staticmethod
    def _multivariate(sample, muy, sigma):
        """Format for sample must be (B, wp, 2)"""
        _, N, *_      = muy.shape
        _, branch, *_ = sample.shape
        sample = sample.unsqueeze(2).expand(-1, -1, N, -1, -1)
        muy    = muy.unsqueeze(1).expand(-1, branch, -1, -1, -1)
        sigma  = sigma.unsqueeze(1).expand(-1, branch, -1, -1, -1)

        # joint probability distribution between x and y => norm const is prod while exp term is sum
        norm_const = (1.0 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * sigma)).prod(dim=-1)
        exp_term = torch.exp(-0.5 * (((sample - muy) / sigma) ** 2).sum(dim=-1))
        return norm_const * exp_term