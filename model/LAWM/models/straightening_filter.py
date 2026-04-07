import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except:
    print("No Mamba found")
from torch.nn.modules.utils import _pair
from .utils.modules import Block
from .utils.pos_embs import get_3d_sincos_pos_embed

FILTER_REGISTRY = {}
def register(name):
    def decorator(fn):
        FILTER_REGISTRY[name] = fn
        return fn
    return decorator

@register('MambaStraightener')
class MambaStraightener(nn.Module):
    def __init__(
        self, 
        img_size: tuple[int, int] = (224, 224), 
        patch_size: int = 16, 
        embed_dim: int = 1024, 
        filter_dim: int = 512, 
        d_state: int = 16, 
        d_conv: int = 4, 
        expansion: int = 2,
        layer_norm = nn.LayerNorm,
        drop_rate: float = 0.0,
        init_std: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        self.num_patches = [size // patch_size for size in img_size]
        
        self.forward_mamba = Mamba(
            d_model = filter_dim,
            d_state = d_state,
            d_conv  = d_conv,
            expand  = expansion,
            use_fast_path = False,
            bias    = True
        )

        self.backward_mamba = Mamba(
            d_model = filter_dim,
            d_state = d_state,
            d_conv  = d_conv,
            expand  = expansion,
            use_fast_path = False,
            bias    = True
        )
        
        self.norm = layer_norm(filter_dim)
        self.straighten_embed = nn.Linear(embed_dim, filter_dim, bias = True)
        self.reproj = nn.Linear(filter_dim, embed_dim, bias = True)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None

        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor, H_patches: int = None, W_patches: int = None):

        if H_patches is not None:
            H = H_patches
        else: H = self.num_patches[0]
        if W_patches is not None:
            W = W_patches
        else: W = self.num_patches[1]
        
        B, N, D = x.shape

        dtype = self.forward_mamba.dt_proj.weight.dtype
        x = x.to(dtype)
        x = x.view(B, -1, H, W, D) 
        _, T, *_ = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, -1, D).contiguous() # (B*H*W, T, D)
        
        x = self.straighten_embed(x)
        
        identity = x
        if T > 1:
            x_norm = self.norm(x)
            # -- Bidirectional mamba
            forward = self.forward_mamba(x_norm)
            bwd_in = x_norm.flip(dims=[1]).contiguous()
            backward = self.backward_mamba(bwd_in).flip(dims=[1]).contiguous()
            filtered = forward + backward
                
            x = identity + self.res_scale * filtered
        else:
            x = identity + self.res_scale * self.norm(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.reproj(x)
        
        x = x.view(B, H, W, -1, D).permute(0, 3, 1, 2, 4).reshape(B, N, D).contiguous()
        
        return x

@register("TransformerStraightener")
class TransformerStraightener(nn.Module):
    def __init__(
        self, 
        img_size=(224, 224),
        patch_size=16,
        tubelet_size=2,
        embed_dim = 1024, 
        filter_dim = 1024,
        depth=2,
        max_frames=2,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=False,
        use_sdpa=True,
        **kwargs
    ):
        super().__init__()
        
        
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_activation_checkpointing = use_activation_checkpointing

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.max_frames = max_frames
        self.tubelet_size = tubelet_size
        self.num_patches = [size // patch_size for size in img_size]

        self.straighten_embed = nn.Linear(embed_dim, filter_dim, bias = True)
        self.reproj = nn.Linear(filter_dim, embed_dim, bias = True)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.uniform_power = uniform_power
        self.use_rope = use_rope
        self.pos_embed = None if self.use_rope else nn.Parameter(torch.zeros(1, int(torch.prod(torch.Tensor(self.num_patches))) * (max_frames // tubelet_size), embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=max_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        grid_depth = self.max_frames // self.tubelet_size
        sincos = get_3d_sincos_pos_embed(
            embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}
    
    def forward(self, x: torch.Tensor, H_patches: int = None, W_patches: int = None):

        B, N, D = x.shape
        
        if H_patches is not None:
            H = H_patches
        else: H = self.num_patches[0]
        if W_patches is not None:
            W = W_patches
        else: W = self.num_patches[1]
        T = N // (H * W)


        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
            x += pos_embed
            
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, None, None, T = T, H_patches = H, W_patches = W, use_reentrant = False
                )
            else:
                x = blk(x, mask = None, attn_mask = None, T = T, H_patches = H, W_patches = W)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        # If pos_embed already correct size, just return
        _, _, T, H, W = x.shape
        if H == self.img_height and W == self.img_width and T == self.max_frames:
            return pos_embed

        # Just chop off last N tokens of positional embedding
        elif H == self.img_height and W == self.img_width and T < self.max_frames:
            new_N = int((T // self.tubelet_size) * (H // self.patch_size) * (W // self.patch_size))
            return pos_embed[:, :new_N, :]

        # Convert depth, height, width of input to be measured in patches
        # instead of pixels/frames
        T = T // self.tubelet_size
        H = H // self.patch_size
        W = W // self.patch_size

        # Compute the initialized shape of the positional embedding measured
        # in patches
        N_t = self.max_frames // self.tubelet_size
        N_h = self.img_height // self.patch_size
        N_w = self.img_width // self.patch_size
        assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

        # Compute scale factor for spatio-temporal interpolation
        scale_factor = (T / N_t, H / N_h, W / N_w)

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
            scale_factor=scale_factor,
            mode="trilinear",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed

class Aggregation(nn.Module):
    def __init__(self, embed_dim = 768, bottleneck_dim = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, bottleneck_dim)
        )
    def forward(self, x: torch.Tensor):
        return self.mlp(x.mean(1))
        
        
        
if  __name__ == "__main__":
    import yaml
    
    device = torch.device('cuda')
    
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", pretrained=False, trust_repo=True, skip_validation = True)
    encoder = encoder.to(device)
    encoder.use_activation_checkpointing = True
    
    from augmenter.transforms_builder import VideoTransform
    from torch.utils.data import DataLoader
    from datasets.dataset import StraighteningDataset

    with open("./cfgs/probe/probe-256px-1024.24e.yaml", "r") as f:
        test_cfg = yaml.safe_load(f)
    
    
    transform = VideoTransform(
        random_horizontal_flip = False,
        reprob = 0.1,
        random_resize_aspect_ratio = (0.75, 4/3),
        random_resize_scale = (0.7, 1.2),
        auto_augment = True,
        motion_shift = True,
        crop_size = 384,
        normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
    straight_filter = MambaStraightener(
        img_size = (384, 384),
        embed_dim = 768
    ).to(device)

    # straight_filter = TransformerStraightener(
    #     img_size = 384,
    #     embed_dim = 768, filter_dim = 512,
    #     use_activation_checkpointing = True, use_rope = True,
    #     max_frames = 16
    # ).to(device)

    dataset = StraighteningDataset(
        data_paths = [
            "./../Autonomous_Dataset/carla/LAWM/recording_20251025_142727_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260204_010805_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260308_212005_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260317_214033_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260317_233603_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260318_083409_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260323_200940_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260329_233141_best_spatial/",
        ],
        shared_transform = transform,
    )
    
    
    dataloader = DataLoader(dataset, batch_size = 6, shuffle = True)
    
    sample = next(iter(dataloader)).to(device)
    sample = sample[:, :, :1]
    
    with torch.autocast('cuda', torch.bfloat16):
        with torch.no_grad():
            output = encoder(sample)
            output = straight_filter(output)
            
            print(output.shape)