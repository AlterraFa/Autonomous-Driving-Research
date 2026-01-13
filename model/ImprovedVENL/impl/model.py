import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import warnings
import numpy as np
import math

from ..data_utils.image_transform import Normalization
from utils.messages.logger import Logger
from .cross import CrossAttention, CrossRoPEAttention
from .atten import Attention, RoPEAttention
from .modules import (
    Block, 
    PosEmbed2d, 
    Patchify, 
    GatedAttentionPooling, 
    SpatialFeatureExtractor, 
    trunc_normal_, 
    GRU_WP, 
    GRU_Gaussian
)
from torch.nn import functional as F

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

def get_activation_by_name(name):
    class_name = name.split(".")[-1]
    
    try:
        Logger().INFO(f"Using activation {class_name}")
        return getattr(nn, class_name)
    except AttributeError:
        Logger().ERROR(f"{class_name} is not a valid attribute of torch.nn", exit_code = -1)
        
class ImprovedVENL(nn.Module):
    def __init__(
        self, 
        input_metadata: dict,
        patch_sizes: list[int],
        output_names: list,
        components: int,
        num_waypoints: int,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio = 4,
        qkv_bias = False,
        qk_scale = None,
        hidden_dim = 256,
        drop = 0.0,
        attn_drop = 0.0,
        drop_path = 0.0,
        drop_route = 0.0, 
        drop_all = 0.0,
        act_layer = "nn.GELU",
        wide_silu = True,
        norm_layer = "nn.LayerNorm",
        use_sdpa = True,
        use_rope = False,
        use_gru  = False, 
        use_cls  = False,
        use_gradient_ckpt = True,
        init_std = 0.01
    ):
        self.log = Logger()
        super().__init__()

        # Extract mode from output_names
        self.mode = output_names[0]
        
        # Store configuration
        self.components = components
        self.num_waypoints = num_waypoints
        self.input_metadata = input_metadata
        self.output_names = output_names
        self.use_gru = use_gru
        self.droprate = drop
        self.drop_route = drop_route
        self.drop_all = drop_all
        self.initialized = False
        self.use_rope = use_rope
        self.use_cls = use_cls
        self.use_gradient_ckpt = use_gradient_ckpt
        
        if self.use_rope: self.log.INFO("Using RoPE embedding")
        else: self.log.INFO("Using standard positional embedding")

        self.patchifies = nn.ModuleDict()
        patches_data = {}
        for (name, shapes), patch_size in zip(input_metadata.items(), patch_sizes):
            patches_data[name] = [shape // patch_size for shape in shapes[2:]]
            self.patchifies[name] = Patchify(patch_size, shapes[1], out_channels = embed_dim)

        if use_cls:
            self.log.WARNING("Using class token")
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad = True)
        else:
            self.log.WARNING("Using Spatial and Gated attention pooling")
            self.pooling_uncertain = SpatialFeatureExtractor(num_queries = components, feature_dim=embed_dim, hidden_dim=embed_dim)
            self.pooling_certain   = GatedAttentionPooling(feature_dim=embed_dim, hidden_dim=hidden_dim)

        if not self.use_rope:
            self.pos_embed_I0 = PosEmbed2d(patches_data["I0"], embed_dim)
            self.pos_embed_MU = PosEmbed2d(patches_data["MU"], embed_dim)
            self.pos_embed_MR = PosEmbed2d(patches_data["MR"], embed_dim)
        
        act_layer = get_activation_by_name(act_layer)
        norm_layer = get_activation_by_name(norm_layer)

        general_args = {
            "dim": embed_dim, 
            "num_heads": num_heads, 
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "qk_scale": qk_scale,
            "drop": drop,
            "attn_drop": attn_drop,
            "drop_path": drop_path,
            "act_layer": act_layer,
            "norm_layer": norm_layer,
            "wide_silu": wide_silu,
            "use_sdpa" : use_sdpa,
            "grid_size": patches_data["I0"][0]
        }
        
        cross_type = CrossRoPEAttention if use_rope else CrossAttention
        attn_type  = RoPEAttention if use_rope else Attention
            
            
        self.cross_uncertain = Block(
            attn_type = cross_type, 
            **general_args
        )
        self.uncertain_blocks = nn.ModuleList([
            Block(
                attn_type = attn_type,
                **general_args
            ) for _ in range(depth - 1)
        ])
        self.cross_certain = Block(
            attn_type = cross_type,
            **general_args
        )
        self.certain_blocks = nn.ModuleList([
            Block(
                attn_type = attn_type,
                **general_args
            ) for _ in range(depth - 1)
        ])


        if not use_gru:
            self.log.INFO("Using Linear as decoder")
            self.gmm_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),

                nn.Linear(hidden_dim, components * (1 + num_waypoints * 4))  # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation
            )
            self.determ_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                
                nn.Linear(hidden_dim, num_waypoints * 2)
            )

        else:
            self.log.INFO("Using GRU as decoder")
            self.determ_head = GRU_WP(num_waypoints, embed_dim, hidden_dim)
            self.gmm_head = GRU_Gaussian(num_waypoints, components, embed_dim, hidden_dim)
            
        self.init_std = init_std

        # Apply weight initialization explicitly to avoid bound-method signature issues
        for module in self.modules():
            self._init_weights(module)

        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        all_blocks = [self.cross_uncertain, *self.uncertain_blocks, self.cross_certain, *self.certain_blocks]
        for layer_id, layer in enumerate(all_blocks):
            if hasattr(layer, "attn") and hasattr(layer.attn, "proj"):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc2"):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["log"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.log = Logger()




    def forward(self, I0: torch.Tensor, MU: torch.Tensor, MR: torch.Tensor) -> torch.Tensor:
        argcount = self.forward.__code__.co_argcount
        argnames = self.forward.__code__.co_varnames[: argcount]

        if not torch.onnx.is_in_onnx_export():
            for name in argnames[1: ]: # skip self
                tensor = locals()[name]
                expected_shape = self.input_metadata.get(name)
                if tuple(expected_shape[1:]) != tuple(tensor.shape)[1:]:
                    self.log.ERROR(f"Input tensor {name} has shape {tensor.shape[1:]}, expected {expected_shape[1:]}", exit_code = 12)

        B, *_ = I0.shape

        if self.training:
            mask_route = torch.rand(MR.shape[0], device=MR.device) < self.drop_route
            mask_all = (torch.rand(MR.shape[0], device=MR.device) < self.drop_all) & mask_route
            
            m1 = mask_route.view(-1, 1, 1, 1)
            m2 = mask_all.view(-1, 1, 1, 1)

            MR = torch.where(m1, MU.repeat(1, 3, 1, 1), MR)
            MR = torch.where(m2, torch.zeros_like(MR), MR)
            MU = torch.where(m2, torch.zeros_like(MU), MU)

        patch_I0 = self.patchifies["I0"](I0)
        patch_MU = self.patchifies["MU"](MU)
        patch_MR = self.patchifies["MR"](MR)
        
        
        if not self.use_rope:
            patch_I0 = patch_I0 + self.pos_embed_I0()
            patch_MU = patch_MU + self.pos_embed_MU()
            patch_MR = patch_MR + self.pos_embed_MR()

        if self.use_cls:
            injected_embed = self.cls_token.repeat(B, 1, 1)
            patch_I0 = torch.cat([patch_I0, injected_embed], dim = 1)
        
        if self.use_gradient_ckpt:
            out = torch.utils.checkpoint.checkpoint(
                self.cross_uncertain, patch_I0, patch_MU, use_reentrant = False
            )
        else: 
            out = self.cross_uncertain(patch_I0, patch_MU)
        

        for blk in self.uncertain_blocks:
            if self.use_gradient_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    blk, out, use_reentrant = False
                )
            else: 
                out = blk(out)
        if self.use_cls:
            uncertain_embed = out[:, -1, :]
        else:
            uncertain_embed = self.pooling_uncertain(out)

        if self.use_gradient_ckpt:
            out = torch.utils.checkpoint.checkpoint(
                self.cross_certain, out, patch_MR, use_reentrant = False
            )
        else: 
            out = self.cross_certain(out, patch_MR)

        for blk in self.certain_blocks:
            if self.use_gradient_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    blk, out, use_reentrant = False
                )
            else: 
                out = blk(out)
        if self.use_cls:
            certain_embed = out[:, -1, :]
        else:
            certain_embed = self.pooling_certain(out)
            
        
        gmm_out = self.gmm_head(uncertain_embed)
        determ_out = self.determ_head(certain_embed)
        
        
        if not self.use_gru:
            return determ_out.view(-1, self.num_waypoints, 2), *self.extract_gparams(gmm_out)
        else:
            return determ_out, *gmm_out
        
    def extract_gparams(self, gmm_params: torch.Tensor):
        if not hasattr(self, "num_waypoints"):
            # predetermined 3 parameters correspond to 3 chunks 
            weights, muy_weights, sigma_weights = torch.chunk(gmm_params, 3, 1)
            weights = torch.softmax(weights, dim=1) 
            muy     = muy_weights                       
            sigma   = torch.exp(sigma_weights)
            return weights, muy, sigma
        else:
            try:
                weights, muy_weights, sigma_weights = torch.split(
                    gmm_params, 
                    [
                        self.components, 
                        self.components * self.num_waypoints * 2, 
                        self.components * self.num_waypoints * 2
                    ],  # 1 weights, num_waypoints * 2 mean, num_waypoints * 2 standard deviation per components
                    dim=1
                )
                weights = torch.log_softmax(weights, dim=1).unsqueeze(-1)
                muy     = muy_weights.view(-1, self.components, self.num_waypoints, 2)
                sigma   = (F.softplus(sigma_weights) + 1e-2).view(-1, self.components, self.num_waypoints, 2)  # (batch, modes, waypoints, dim)
                return weights, muy, sigma
            except:
                self.log.ERROR("Cannot split the tensor. Maybe you disabled GRU with CLS which is incompatible at the moment", exit_code = -1)

            
    @staticmethod
    def postprocessor(raw_out: dict, data):
        return tuple([output[0] for output in raw_out.values()])

    def preprocessor(self, **kwargs):
        # Sanity check
        missing_keys = [key for key in self.input_metadata.keys() if key not in kwargs]
        if missing_keys:
            self.log.ERROR(f"Missing keys: {missing_keys}", exit_code = 2)

        if not hasattr(self, "main_norm"):
            import yaml
            with open(self.config_path, "r") as f:
                args = yaml.safe_load(f)
            norm   = args.get('data_aug', {}).get('normalization', [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
            crop   = args.get('data_aug', {}).get("crop", [0, 1, 0, 1])
            input_metadata = args.get('model', {}).get('input_metadata', {
                'I0': [1, 3, 224, 224], 
                'MU': [1, 1, 128, 128], 
                'MR': [1, 3, 128, 128]}
            )

            self.main_norm = Normalization(
                size = input_metadata['I0'][2:],
                crop = crop,
                normalization = norm,   
            )
            self.aux_norm = Normalization(
                size = input_metadata['MU'][2:],
                crop = crop,
            )

        I0 = torch.flip(self.main_norm(kwargs['I0'][..., :3])[None, ...], dims = [1])
        MU = self.aux_norm(kwargs['MU'][..., None])[None, ...]
        MR = torch.flip(self.aux_norm(kwargs['MR'])[None, ...], dims = [1])

        return I0, MU, MR


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


if __name__ == "__main__":
    import os, yaml
    
    FOLDER_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(FOLDER_DIR, "../configs/model_cfg.yaml")
    
    
    with open(yaml_path, "r") as f:
        args = yaml.safe_load(f)
    
    model_cfg = args['model']
    input_metadata = model_cfg.get('input_metadata', {})
    patch_sizes = model_cfg.get('patch_sizes', [16, 16, 16])
    output_names = model_cfg.get('output_names', ["waypoint", "weights", "muy", "sigma"])
    components = model_cfg.get('components', 6)
    num_waypoints = model_cfg.get('num_waypoints', 6)
    depth = model_cfg.get('depth', 5)
    embed_dim = model_cfg.get('embed_dim', 512)
    num_heads = model_cfg.get('num_heads', 8)
    mlp_ratio = model_cfg.get('mlp_ratio', 4)
    qkv_bias = model_cfg.get('qkv_bias', True)
    qk_scale = model_cfg.get('qk_scale', None)
    drop = model_cfg.get('drop', 0.0)
    attn_drop = model_cfg.get('attn_drop', 0.0)
    drop_path = model_cfg.get('drop_path', 0.0)
    drop_route = model_cfg.get('drop_route', 0.0)
    drop_all = model_cfg.get('drop_all', 0.0)
    use_rope = model_cfg.get('use_rope', True)
    use_gradient_ckpt = model_cfg.get('use_gradient_ckpt', True)
    
    model = ImprovedVENL(
        input_metadata=input_metadata,
        patch_sizes=patch_sizes,
        output_names=output_names,
        components=components,
        num_waypoints=num_waypoints,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path,
        drop_route=drop_route,
        drop_all=drop_all,
        use_sdpa=True,
        use_rope=False,
        use_gradient_ckpt=use_gradient_ckpt,
        use_gru = True
    )
    

    device = torch.device("cuda")

    

    dummy_I0 = torch.zeros(input_metadata['I0']).to(device)
    dummy_MU = torch.zeros(input_metadata['MU']).to(device)
    dummy_MR = torch.zeros(input_metadata['MR']).to(device)
    
    model.to(device)
    model(dummy_I0, dummy_MU, dummy_MR)
    # torch.save(model.state_dict(), "./ImprovedVENL/Experiment/run5/weights/best_ImprovedVENL.pt")