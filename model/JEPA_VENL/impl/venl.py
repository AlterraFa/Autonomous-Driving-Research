import torch
import torch.nn as nn
import torch.nn.init as init
import warnings

from utils.messages.logger import Logger
from typing import Any
from torch.nn import functional as F

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

class GatedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Output 1 score per token
        )

    def forward(self, x):
        attn_weights = self.attention_net(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_features = torch.sum(x * attn_weights, dim=1)
        return weighted_features
    
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, num_queries=6, hidden_dim=256):
        super().__init__()
        self.waypoint_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.proj = nn.LazyLinear(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, z):
        B = z.shape[0]
        z_proj = self.proj(z)
        queries = self.waypoint_queries.expand(B, -1, -1)
        
        out, _ = self.cross_attn(queries, z_proj, z_proj)
        return out.flatten(1)
        
class SingleVENL(nn.Module):
    def __init__(self, droprate: float = 0.0, map_droprate: float = 0.0):
        self.log = Logger()
        super().__init__()

        self.emb_pooling = GatedAttentionPooling(hidden_dim = 368)
        # self.emb_spatial = SpatialFeatureExtractor(hidden_dim = 360)

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
            nn.LazyLinear(1024),
            nn.GELU(),
            nn.Dropout(droprate),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(droprate),

            nn.Linear(512, 200),
            nn.GELU(),
            nn.Dropout(droprate),
        )

        self.fusion_projector = nn.Sequential(
            nn.LazyLinear(256),
            nn.GELU(),
            nn.Dropout(droprate),
            
            nn.Linear(256, 100),
            nn.GELU(),
            nn.Dropout(droprate)
        )
    
        self.initialized  = False
        self.droprate     = droprate
        self.map_droprate = map_droprate
        self._override_shape_check = False

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
    def steer(cls, map_shape = (50, 50), components: int = 3, droprate = 0.0, map_droprate = 0.0) -> "SingleVENL":
        self = cls(droprate = droprate, map_droprate = map_droprate)
        self.components = components
        self.log.INFO("Using steer mode")

        self.gmm_head = nn.Linear(200, 3 * components) # 3 gaussian parameters * number of modes
        self.determ_head = nn.Sequential(
            nn.Linear(100, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.input_metadata = {
            "MU": (1, 1, *map_shape),
            "MR": (1, 3, *map_shape),
        }
        self.output_names = ["steer", "weights", "muy", "sigma"]
        
        return self
    

    @classmethod
    def waypoint(cls, map_shape = (50, 50), num_waypoints = 1, components: int = 3, droprate = 0.0, map_droprate = 0.0) -> "SingleVENL":
        self = cls(droprate = droprate, map_droprate = map_droprate)
        self.num_waypoints = num_waypoints
        self.components = components
        self.log.INFO("Using waypoint mode")

        self.gmm_head = nn.Sequential(
            nn.Linear(200, 128),
            nn.GELU(),
            nn.Linear(128, components * (1 + num_waypoints * 4))
        ) 
        
        # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation
        self.determ_head = nn.Sequential(
            nn.Linear(100, 64),
            nn.GELU(),
            nn.Linear(64, num_waypoints * 2)
        )
        self.input_metadata = {
            "MU": (1, 1, *map_shape),
            "MR": (1, 3, *map_shape),
        }
        self.output_names = ["waypoint", "weights", "muy", "sigma"]

        return self

    def initialize_module(self, z: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor):
        if self.initialized == False:
            self.initialized = True

            self.forward(z, MU, MR)
            self._init_weights()
            self.log.INFO("Layer initialized")
        else:
            self.log.WARNING("Layer already initialized")
        
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

    def forward(self, z: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor) -> torch.Tensor:
        argcount = self.forward.__code__.co_argcount
        argnames = self.forward.__code__.co_varnames[: argcount]

        if self.initialized == False:
            self.log.ERROR(f"Modules not initialized", exit_code = -1)
        
        if not torch.onnx.is_in_onnx_export() and not self._override_shape_check:
            self._shape_security(argnames, locals())

                    
        # -- features of unrouted map
        fmu = self.unrouted_backbone(MU)
        # -- Pooling Embedding of Encoder
        emb = self.emb_pooling(z)
        
        # -- Concatenation of left, front, right and map features on a single vector
        out = torch.cat([emb, fmu], dim=1) # TENSORRT DOES NOT SUPPORT HSTACK OR VSTACK

        # -- Dowsize and regularize map input
        out = self.feature_downsize(out)
        if self.training:
            dropmask = torch.rand(MR.shape[0], device = MR.device) < self.map_droprate
            MR = MR.clone()
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
            sigma   = (F.softplus(sigma_weights) + 1e-6).view(-1, self.components, self.num_waypoints, 2)  # (batch, modes, waypoints, dim)
            return weights, muy, sigma


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
