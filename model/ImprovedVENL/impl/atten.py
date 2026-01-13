import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn import functional as F
from .modules import rotate_queries_or_keys

_USABLE_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim = 512,
        num_heads = 8,
        qkv_bias  = True,
        qk_scale  = None,
        attn_drop = 0.0,
        proj_drop = 0.0,
        use_sdpa = False,
    ):
        super().__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias = qkv_bias)

        self.proj = nn.Linear(self.dim, self.dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_prob = proj_drop
        self.use_sdpa = use_sdpa

        
    def forward(self, x):
        B, N, C = x.shape
        # -- (2, B, num_heads, tokens, dim_per_head)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
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

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class RoPEAttention(nn.Module):
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
    ):
        super().__init__()

        
        self.num_heads = num_heads
        self.head_dim = head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

        self.h_dim = int(head_dim // 2)
        self.w_dim = int(head_dim // 2)
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
        
    def forward(self, x):
        B, N, C = x.shape
        # -- (2, B, num_heads, tokens, dim_per_head)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        num_patches = self.grid_size * self.grid_size
        
        q_cls = q[:, :, num_patches:, :]
        k_cls = k[:, :, num_patches:, :]

        
        mask = torch.arange(N, device = x.device, dtype = torch.long)
        row_ids, col_ids = self.separate_position(mask)
        
        s = 0
        qh = rotate_queries_or_keys(q[..., s: s + self.h_dim], pos = row_ids)
        kh = rotate_queries_or_keys(k[..., s: s + self.h_dim], pos = row_ids)
        s += self.h_dim
        
        qw = rotate_queries_or_keys(q[..., s: s + self.w_dim], pos = col_ids)
        kw = rotate_queries_or_keys(k[..., s: s + self.w_dim], pos = col_ids)
        s += self.w_dim

        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qh, qw, qr], dim = 3)
            k = torch.cat([kh, kw, kr], dim = 3)
        else:
            # -- Pls specify the correct dimenstion instead of lazying it and use dim = -1
            q = torch.cat([qh, qw], dim = 3)
            k = torch.cat([kh, kw], dim = 3)

        q[:, :, num_patches:, :] = q_cls
        k[:, :, num_patches:, :] = k_cls
            

        # -- When using tensorrt please fucking use sdpa
        if self.use_sdpa or torch.onnx.is_in_onnx_export():
            with sdpa_kernel(_USABLE_BACKENDS):
                out = F.scaled_dot_product_attention(
                    q, k, v
                )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim = -1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
            