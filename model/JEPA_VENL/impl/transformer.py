import torch
import torch.nn as nn
import math
import numpy as np

from einops.layers.torch import Rearrange
from timm.layers.drop import DropPath
from typing import Optional


class Patchify(nn.Module):
    def __init__(self, num_patches, in_channels, out_channels, droprate=0.0):
        super().__init__()
        # Note: Dropout removed from patchify - it corrupts initial embeddings
        # and causes train/eval mismatch between encoder and EMA target
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = num_patches, stride = num_patches),
            Rearrange("b d h w -> b (h w) d")
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)

class PosEmbed2d(nn.Module):
    def __init__(self, num_patches: list, dim):
        super(PosEmbed2d, self).__init__()
        
        x_coor = torch.arange(num_patches[0])
        y_coor = torch.arange(num_patches[1])
        grid   = torch.meshgrid(x_coor, y_coor, indexing = 'ij')
        
        embed1d = PosEmbed1d(dim // 2)
        
        embed1 = embed1d._initialize(grid[0].flatten())
        embed2 = embed1d._initialize(grid[1].flatten())
        
        self.embed = nn.Parameter(torch.concatenate([embed1, embed2], 1), requires_grad = False)
        
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
        
    def forward(self, x):
        return self.embed
        
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViTEncode(nn.Module):
    def __init__(self,         
        img_size   = [224, 224],
        patch_size = 16,
        in_chans   = 3,
        embed_dim  = 768,
        depth      = 12,
        num_heads  = 12,
        mlp_ratio  = 4.0,
        qkv_bias   = True,
        qk_scale   = None,
        drop_rate  = 0.0,
        attn_drop_rate = 0.0,
        drop_path_rate = 0.0,
        init_std   = 1.0
    ):
        super(ViTEncode, self).__init__()
        """Includes:
            - Implemention sinusoidal positional encoding
            - Implemention interpolation
            - Implemention attention
            - Implemention mlp
            - Implemention droppath
        """
        self.base_size   = img_size
        self.base_patch  = patch_size
        self.num_patches = [self.base_size[0] // self.base_patch, self.base_size[1] // self.base_patch]
        self.dim         = embed_dim
        
        self.patchify = Patchify(
            num_patches  = patch_size,
            in_channels  = in_chans,
            out_channels = embed_dim,
            droprate     = drop_rate
        )
        
        self.pos_embed = PosEmbed2d(self.num_patches, embed_dim)
        
        blocks = []
        drop_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for idx in range(depth):
            blocks += [Block(
                dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias,
                qk_scale = qk_scale, drop = drop_list[idx], attn_drop = attn_drop_rate, drop_path = drop_rate,
                act_layer = nn.GELU, norm_layer = nn.LayerNorm
            )]
        self.blocks = nn.ModuleList(blocks)
        self.norm_layer = nn.LayerNorm(embed_dim)
        
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

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

    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:

        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]
        _, _, H, W = x.shape
        x = self.patchify(x)
        x += self.interpolate(x, self.pos_embed(), (H // self.base_patch, W // self.base_patch))
        
        if masks is not None:
            # -- enc mask idx -> image idx hierachy
            x = apply_masks(x, masks)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm_layer(x)
        return x

    def interpolate(self, x, embed, size):
        _, img_sz, _ = x.shape
        emb_sz, dim  = embed.shape
        target_h, target_w = size
        base_h, base_w     = self.num_patches
        
        if img_sz == emb_sz:
            return embed
        
        embed_aligned = nn.functional.interpolate(
            embed.reshape(-1, base_h, base_w, dim).permute(0, 3, 1, 2),
            size = size,
            mode = 'bicubic'
        )
        
        return embed_aligned.view(-1, dim, target_h * target_w).permute(0, 2, 1)


class ViTPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches, 
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
    ):
        super(ViTPredictor, self).__init__()
        """Includes:
            - Dimension bottleneck to reducce influence of prediction token
            - Positional tokens application on both prediction token and embeddings from `ViTEncode`
            - Token concatentation from of prediction and encoder embeddings
            - Implementation of Attention
        """
        
        self.pred_dim  = predictor_embed_dim
        self.embed_dim = embed_dim

        self.dim_reduction = nn.Linear(embed_dim, predictor_embed_dim, bias = True)
        
        self.pred_tokens = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        self.pos_embed = PosEmbed2d(num_patches, predictor_embed_dim)
        
        blk_drop = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blk = []
        for idx in range(depth):
            blk += [Block(
                dim = predictor_embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias,
                qk_scale = qk_scale, drop = drop_rate, attn_drop = attn_drop_rate, drop_path = blk_drop[idx],
                act_layer = nn.GELU, norm_layer = nn.LayerNorm   
            )]
        self.blocks = nn.ModuleList(blk)
        self.norm   = nn.LayerNorm(predictor_embed_dim)
        
        self.dim_increase = nn.Linear(predictor_embed_dim, embed_dim, bias = True)
        

        self.init_std = init_std
        
        nn.init.trunc_normal_(self.pred_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

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

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'
        
        # -- Dimension reduction
        x = self.dim_reduction(x)
        B = len(x) // len(masks_x)
        
        # -- Add positional encoding to representation
        pos_embedding = self.pos_embed().repeat(B, 1, 1)
        x += apply_masks(pos_embedding, masks_x)

        _, nctx, _ = x.shape

        # -- Add position encoding to the tokens
        # -- Each embeddings correspond to `npred` tokens
        pred_pos_emb = self.pos_embed().repeat(B, 1, 1)
        # -- pred mask idx -> pos embed idx hierachy
        pred_pos_emb = apply_masks(pred_pos_emb, masks)
        pred_pos_emb = repeat_interleave_batch(
            x = pred_pos_emb, 
            B = B,
            repeat = len(masks_x)
        )
            
        pred_tokens  = self.pred_tokens.repeat(pred_pos_emb.size(0), pred_pos_emb.size(1), 1)
        pred_tokens += pred_pos_emb

        # # -- Concatenate tokens
        x = x.repeat(len(masks), 1, 1)
        x = torch.concatenate([x, pred_tokens], dim = 1)
        
        # -- Attention inference
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # -- extract the predicted token and increase its dimension
        x = x[:, nctx:, :]
        x = self.dim_increase(x)
        
        return x


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12)
    
    max_enc_patches  = torch.randint(0, 70, (1, ))
    max_pred_patches = torch.randint(0, 70, (1, ))
    nenc    = 1
    npred   = 4
    batchsz = 3
    
    dummy = torch.randn((batchsz, 3, 304, 224)).to(device)
    dummy_masks = [torch.randint(0, 196, (batchsz, max_enc_patches)).to(device) for i in range(nenc)]
    dummy_pred  = [torch.randint(0, 196, (batchsz, max_pred_patches)).to(device) for i in range(npred)]

    enc_model = ViTEncode(
        img_size = [288, 224],
        patch_size = 16,
        embed_dim = 512, 
        depth = 12,
        num_heads = 8   
    ).to(device)

    pred_model = ViTPredictor(
        num_patches = [18, 14],
        embed_dim = 512 
    ).to(device)
    
    # -- masks shape must be (B, npred, ?) where this ? depends on the batch
    # -- Batch ordering hierarchy: pred->img->enc
    z = enc_model(dummy, dummy_masks)
    print(z.shape)
    z = pred_model(z, dummy_masks, dummy_pred)
    print(z.shape)