import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from timm.layers import drop_path
from einops.layers.torch import Rearrange

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class Patchify(nn.Module):
    def __init__(self, num_patches, in_channels, out_channels):
        super().__init__()
        # Note: Dropout removed from patchify - it corrupts initial embeddings
        # and causes train/eval mismatch between encoder and EMA target
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = num_patches, stride = num_patches),
        )
        
        self.rearrange = Rearrange("b d h w -> b (h w) d")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.rearrange(x)
        return x
    
class PosEmbed2d(nn.Module):
    def __init__(self, num_patches: list, dim: int):
        super().__init__()
        
        x_coor = torch.arange(num_patches[0], dtype=torch.float32)
        y_coor = torch.arange(num_patches[1], dtype=torch.float32)
        
        grid_x, grid_y = torch.meshgrid(x_coor, y_coor, indexing='ij')
        
        half_dim = dim // 2
        emb_x = self._get_1d_embeddings(grid_x.flatten(), half_dim)
        emb_y = self._get_1d_embeddings(grid_y.flatten(), half_dim)
        
        full_embed = torch.cat([emb_x, emb_y], dim=1).unsqueeze(0)
        self.register_buffer('embed', full_embed)
        
    def _get_1d_embeddings(self, pos, dim):
        indices = torch.arange(dim, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** ( (indices // 2) * 2 / dim))
        
        sin_cos_args = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        
        emb = torch.empty((pos.shape[0], dim), dtype=torch.float32)
        emb[:, 0::2] = torch.sin(sin_cos_args[:, 0::2])
        emb[:, 1::2] = torch.cos(sin_cos_args[:, 1::2])
        return emb
        
    def forward(self) -> torch.Tensor:
        return self.embed
        
class PosEmbed1d(nn.Module):
    def __init__(self, dim):
        super(PosEmbed1d, self).__init__()
        self.dim = dim
        
    def _initialize(self, pos):
        cos = lambda pos, index: torch.cos(pos / 10000 ** (index / self.dim))
        sin = lambda pos, index: torch.sin(pos / 10000 ** (index / self.dim))

        # -- Assumes that pos is already 1D
        index      = torch.arange(self.dim)
        pair_index = (index // 2) * 2
        
        target_size = (pos.reshape(-1).shape[0], pair_index.shape[0])
        pair_index  = pair_index.unsqueeze(0).expand(target_size).float()
        index       = index.unsqueeze(0).expand(target_size)
        flat_pos    = pos.reshape(-1).unsqueeze(1).float() 
        
        emb = torch.where(
            index % 2 == 0, 
            sin(flat_pos, pair_index),
            cos(flat_pos, pair_index)
        )
        self.register_buffer('embed', emb)
        return emb
        
    def forward(self):
        return self.embed
    
class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_type,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        grid_size=16,
        **kwargs
    ):
        super().__init__()

        self.norm11 = norm_layer(dim)
        if "cross" in attn_type.__qualname__.lower():
            self.norm12 = norm_layer(dim)
        if "rope" in attn_type.__qualname__.lower():
            self.attn = attn_type(
                embed_dim = dim,
                num_heads = num_heads,
                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
                attn_drop = attn_drop,
                use_sdpa = use_sdpa,
                grid_size = grid_size,
                proj_drop = drop,
            )
        else:
            self.attn = attn_type(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x1, x2 = None, use_cls = False):
        if "cross" in self.attn._get_name().lower():
            if x2 is None: 
                raise ValueError("Another tensor must be specifed when using cross attention")
            y = self.attn(self.norm11(x1), self.norm12(x2), use_cls)
        else:
            y = self.attn(self.norm11(x1), use_cls)
        x1 = x1 + self.drop_path(y)
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        return x1

def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()

    # use float positions for trig while keeping shapes static
    pos = pos.to(dtype=x.dtype)

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    # freq: (..., N, D/2) outer product of positions and base frequencies
    freq = torch.einsum("...n, f -> ...nf", pos, omega)

    # precompute sin/cos expanded to last dim = D
    sin = freq.sin().repeat_interleave(2, dim=-1).view(1, 1, N, D)
    cos = freq.cos().repeat_interleave(2, dim=-1).view(1, 1, N, D)

    # even/odd split to avoid unflatten/flatten reshape chains
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).reshape(B, num_heads, N, D)

    return (x * cos) + (rotated * sin)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate to
        # [2*lower-1, 2*upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

    


class GatedAttentionPooling(nn.Module):
    def __init__(self, feature_dim = 512, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Output 1 score per token
        )

    def forward(self, x):
        attn_weights = self.attention_net(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_features = torch.sum(x * attn_weights, dim=1)
        return weighted_features
    
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, num_queries=6, feature_dim = 512, hidden_dim=256, use_sdpa = False):
        from .cross import CrossAttention
        super().__init__()
        self.waypoint_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.cross_attn = CrossAttention(
            embed_dim = hidden_dim,
            num_heads = 8,
            use_sdpa = use_sdpa
        )
        
    def forward(self, z):
        B = z.shape[0]
        z_proj = self.proj(z)
        queries = self.waypoint_queries.expand(B, -1, -1)
        
        out = self.cross_attn(queries, z_proj)
        return out


class GRU_WP(nn.Module):
    def __init__(self, num_waypoints, feature_dim, hidden_size):
        super().__init__()
        
        self.encoder = nn.Linear(feature_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.gru_cell = nn.GRUCell(input_size = 2, hidden_size = hidden_size)
        self.decoder  = nn.Linear(hidden_size, 2)
        self.num_waypoints = num_waypoints
        
    def forward(self, x):
        B, *_ = x.shape
        out = self.encoder(x)
        out = self.layer_norm(out)

        waypoints = torch.empty((B, self.num_waypoints, 2), device = x.device)
        
        # Predict relative, supervise absolute
        current_wp = torch.zeros((B, 2), device = x.device)
        for i in range(self.num_waypoints):
            
            
            out = self.gru_cell(current_wp, out)
            delta_wp = self.decoder(out)
            current_wp += delta_wp
            waypoints[:, i, :] = current_wp
            
        return waypoints

class GRU_Gaussian(nn.Module):
    def __init__(self, num_waypoints, num_components, feature_dim, hidden_size):
        super().__init__()
        
        self.num_components = num_components
        self.num_waypoints = num_waypoints
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        
        # For cls_token input: (B, feature_dim) -> process single vector
        self.encoder = nn.Linear(feature_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # For SpatialFeatureExtractor input: (B, num_components, feature_dim) -> process per-component
        self.component_encoder = nn.Linear(feature_dim, hidden_size)
        self.component_layer_norm = nn.LayerNorm(hidden_size)
        
        self.gru_cell = nn.GRUCell(self.num_components * 2, hidden_size)
        self.decoder = nn.Linear(hidden_size, 4 * self.num_components) # mean, mean, std, std
        
        self.decode_weight = nn.Linear(hidden_size, num_components)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Handle both cls_token input (B, feature_dim) and SpatialFeatureExtractor input (B, num_components, feature_dim)
        if x.dim() == 2:
            # cls_token case: (B, feature_dim)
            ctx = self.encoder(x)
            out = self.layer_norm(ctx)
        elif x.dim() == 3:
            # SpatialFeatureExtractor case: (B, num_components, feature_dim)
            # Use mean pooling across components to get a single vector per batch
            ctx = x.mean(dim=1)  # (B, feature_dim)
            out = self.component_encoder(ctx)  # (B, feature_dim) -> (B, hidden_size)
            out = self.component_layer_norm(out)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D input")
        
        weight = F.log_softmax(self.decode_weight(out), dim=1).unsqueeze(2)
        
        mean_stds = torch.empty((B, self.num_components, self.num_waypoints, 4), device=x.device)
        current_mean = torch.zeros((B, self.num_components * 2), device=x.device)
        for i in range(self.num_waypoints):
            out = self.gru_cell(current_mean, out)

            params = self.decoder(out).view(B, self.num_components, 4)
            current_mean += params[..., :2].reshape(B, self.num_components * 2)
            mean_stds[:, :, i, 2:] = params[..., 2:]
            mean_stds[:, :, i, :2] = current_mean.view(B, self.num_components, 2)
        
        muy, sigma = torch.chunk(mean_stds, 2, 3)
        sigma = F.softplus(sigma) + 1e-2
        
        return weight, muy, sigma