import torch
import torch.nn as nn
import torch.nn.init as init

from utils.messages.logger import Logger

class VENL(nn.Module):
    def __init__(self, droprate: float = 0.1):
        self.log = Logger()
        super().__init__()

        # Might change to WRN to improve performance
        self.multicam_backbone: nn.ModuleList[nn.Sequential] = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(in_channels = 3, out_channels = 24, kernel_size = 5, stride = 2),
                nn.LeakyReLU(),
                nn.Dropout(droprate),
                
                nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 3, stride = 2),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 1),
                nn.LeakyReLU(),
                nn.Dropout(droprate),

                nn.Flatten()
            ]) for _ in range(3)
        ])
        
        # Shallow network to prevent translational + rotational invariance
        self.unrouted_backbone: nn.Sequential = nn.Sequential(*[
            nn.Conv2d(in_channels = 1, out_channels = 24, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 3, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Flatten()
        ])

        self.routed_backbone: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 24, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),

            nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 5, stride = 2),
            nn.LeakyReLU(),
            nn.Dropout(droprate),
            
            nn.Flatten()
        )
    
        self.feature_downsize = nn.Sequential(
            nn.Linear(24960, 1000),
            nn.ReLU(),
            nn.Dropout1d(droprate),
            
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout1d(droprate),
        )
        
        self.fusion_projector = nn.Sequential(
            nn.Linear(3700, 100),
            nn.LeakyReLU(),
            nn.Dropout(droprate)
        )

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

    @classmethod
    def steer(cls, components: int = 3, droprate = 0.1):
        self = cls(droprate = droprate)
        self.components = components
        self.log.INFO("Using steer mode")

        self.gmm_head = nn.Linear(100, 3 * components) # 3 gaussian parameters * number of modes
        self.determ_head = nn.Linear(100, 1)
        self._init_weights()
        return self
        
    @classmethod
    def waypoint(cls, num_waypoints = 1, components: int = 3, droprate = 0.1):
        self = cls(droprate = droprate)
        self.num_waypoints = num_waypoints
        self.components = components
        self.log.INFO("Using waypoint mode")

        self.gmm_head = nn.Linear(100, components * (1 + num_waypoints * 4)) # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation
        self.determ_head = nn.Linear(100, num_waypoints * 2)
        self._init_weights()
        return self

    def forward(self, x: list[torch.Tensor], mapU: torch.Tensor, mapR: torch.Tensor) -> torch.Tensor:
        B, I, C, H, W = x.shape
        if I != 3: 
            self.log.ERROR(f"Incorrect amount of RGB image. Received: {I}")
            exit(-1)

        features_cat = [self.multicam_backbone[i](x[:, i]) for i in range(3)] # features of multicam setup
        features_cat += [self.unrouted_backbone(mapU)] # features of unrouted map
        
        # Concatenation of left, front, right and map features on a single vector
        features_cat = torch.hstack(features_cat)
        
        out = self.feature_downsize(features_cat)
        routed_features = self.routed_backbone(mapR)
        
        gmm_out    = self.gmm_head(out)
        determ_in  = torch.hstack([out, routed_features])
        determ_out = self.determ_head(self.fusion_projector(determ_in))
        
        return self._extract_gparams(gmm_out), determ_out
        
    def _extract_gparams(self, gmm_params: torch.Tensor):
        if not hasattr(self, "num_waypoints"):
            weights, muy_weights, sigma_weights = torch.chunk(gmm_params, 3, 1) # predetermined 3 parameters correspond to 3 chunks 

            weights = torch.softmax(weights, dim=1) 
            muy = muy_weights                       
            sigma = torch.exp(sigma_weights)

            return weights, muy, sigma
        
        else:
            weights, muy_weights, sigma_weights = torch.split(
                gmm_params, 
                [
                    self.components, 
                    self.components * self.num_waypoints * 2, 
                    self.components * self.num_waypoints * 2
                ], # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation per components
                dim = 1
            )
            weights = torch.softmax(weights, dim = 1).unsqueeze(-1)
            muy = muy_weights.view(-1, self.components, self.num_waypoints, 2)
            sigma = torch.exp(sigma_weights).view(-1, self.components, self.num_waypoints, 2) # (batch, modes, waypoints, dim)
            
            return weights, muy, sigma
    
    def gaussian_function(self, sample, parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        weights, muy, sigma = parameters
        if not hasattr(self, "num_waypoints"):
            try:
                probs_per_components = self._univariate(sample, muy, sigma)
                return weights * probs_per_components # Return GMM probability per mode with weights
            except Exception as e:
                self.log.ERROR(
                    f"Please check the parameters if it is in the format of univariate or if the sample format is correct. Sample: [bold]{sample.shape}[/], Paramters: [bold]{muy.shape}[/]",
                    full_traceback = e
                )
                exit(-1)
        else:
            if muy.shape[-2] != sample.shape[1]:
                self.log.ERROR(f"Mismatch number of waypoints. Sample: [bold]{sample.shape}[/], Paramters: [bold]{muy.shape}[/]")
                exit(-1)
            
            try:
                probs_per_components = self._multivariate(sample, muy, sigma)
                return weights * probs_per_components # returns joint probability of x, y per mode per waypoint
            except Exception as e:
                self.log.ERROR(
                    f"Please check the parameters if it is in the format of multivariate or if the sample format is correct. Sample: [bold]{sample.shape}[/], Parameters: [bold]{muy.shape}[/]",
                    full_traceback = e
                )
                exit(-1)
            
    @staticmethod
    def _univariate(sample, muy, sigma):
        return (1 / (2 * torch.pi * sigma ** 2) ** .5) * torch.exp( -(sample - muy) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def _multivariate(sample, muy, sigma):
        """Format for sample must be (B, wp, 2)"""
        _, N, *_ = muy.shape

        sample = sample.unsqueeze(1).expand(-1, N, -1, -1)

        # joint probability distribution between x and y => norm const is prod while exp term is sum
        norm_const = (1.0 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * sigma)).prod(dim = 3)
        exp_term = torch.exp(-0.5 * (((sample - muy) / sigma) ** 2).sum(dim = 3))

        return norm_const * exp_term