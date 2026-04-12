import torch
import torch.nn as nn
import math
from timm.layers.drop import DropPath
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn import functional as F

_USABLE_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]

class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim = 512,
        num_heads = 8,
        qkv_bias  = True,
        qk_scale  = None,
        attn_drop = 0.0,
        proj_drop = 0.0,
        use_sdpa = False,
        **kwargs
    ):
        super().__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -.5
        self.kv = nn.Linear(self.dim, self.dim * 2, bias = qkv_bias)
        self.q  = nn.Linear(self.dim, self.dim, bias = qkv_bias)

        self.proj = nn.Linear(self.dim, self.dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_prob = proj_drop
        self.use_sdpa = use_sdpa

        
    def forward(self, x1, x2):
        B2, N2, C2 = x2.shape
        # -- (2, B, num_heads, tokens, dim_per_head)
        kv = self.kv(x2).reshape(B2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        B1, N1, C1 = x1.shape
        q = self.q(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        
        if self.use_sdpa:
            with sdpa_kernel(_USABLE_BACKENDS):
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p = self.proj_drop_prob
                )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim = -1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B1, N1, C1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class CrossRoPEAttention(nn.Module):
    def __init__(
        self, 
        embed_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        **kwargs
    ):
        super().__init__()

        
        self.num_heads = num_heads
        self.head_dim = head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q  = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

        self.h_dim = int(head_dim // 2)
        self.w_dim = int(head_dim // 2)
        self.rope_dim = self.h_dim + self.w_dim
        self.has_residual_dim = self.rope_dim < head_dim
        
        self.grid_size = grid_size
        
    
    def get_row_pos(self, ids, H_patches = None, W_patches = None):
        if H_patches is None or W_patches is None:
            tokens_per_row = int(self.grid_size)
        else: 
            tokens_per_row = int(W_patches)
        return ids // tokens_per_row
    
    def separate_position(self, ids, H_patches = None, W_patches = None):
        if H_patches is None or W_patches is None:
            tokens_per_row = int(self.grid_size)
        else: 
            tokens_per_row = int(W_patches)
        
        row_ids = self.get_row_pos(ids, H_patches, W_patches)
        
        col_ids =  ids - row_ids * tokens_per_row
        
        return row_ids, col_ids
        
    def forward(self, x1, x2, use_cls):
        B2, N2, C2 = x2.shape
        # -- (2, B, num_heads, tokens, dim_per_head)
        kv = self.kv(x2).reshape(B2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        B1, N1, C1 = x1.shape
        q = self.q(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        grid_k_tensor = torch.sqrt(torch.tensor(N2, dtype=x2.dtype, device=x2.device))
        grid_k = int(torch.round(grid_k_tensor).item())
        if use_cls:
            num_patches = min(self.grid_size * self.grid_size, N1)
            q_cls = q[:, :, num_patches:, :]

        # integer positions for patch tokens
        mask_1 = torch.arange(N1, device=x1.device, dtype=torch.long)
        mask_2 = torch.arange(N2, device=x2.device, dtype=torch.long)
        
        row_ids1, col_ids1 = self.separate_position(mask_1)
        row_ids2, col_ids2 = self.separate_position(mask_2, grid_k, grid_k)
        
        # Apply RoPE to height and width position dimensions
        qh = rotate_queries_or_keys(q[..., :self.h_dim], pos = row_ids1)
        kh = rotate_queries_or_keys(k[..., :self.h_dim], pos = row_ids2)
        
        qw = rotate_queries_or_keys(q[..., self.h_dim: self.h_dim + self.w_dim], pos = col_ids1)
        kw = rotate_queries_or_keys(k[..., self.h_dim: self.h_dim + self.w_dim], pos = col_ids2)
        
        # Remaining dimensions (if any) are left unrotated
        # Use precomputed flag to ensure static branching for TensorRT
        if self.has_residual_dim:
            qr = q[..., self.rope_dim:]
            kr = k[..., self.rope_dim:]
            q_final = torch.cat([qh, qw, qr], dim = 3)
            k_final = torch.cat([kh, kw, kr], dim = 3)
        else:
            q_final = torch.cat([qh, qw], dim = 3)
            k_final = torch.cat([kh, kw], dim = 3)
        
        if use_cls:
            q[:, :, num_patches:, :] = q_cls

        if self.use_sdpa:
            with sdpa_kernel(_USABLE_BACKENDS):
                out = F.scaled_dot_product_attention(q_final, k_final, v)
        else:
            attn = (q_final @ k_final.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim = -1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B1, N1, C1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
            

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