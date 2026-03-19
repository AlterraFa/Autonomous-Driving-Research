import math

import torch
import torch.nn as nn

import torch.nn.functional as F
from ..utils.modules import Block, EfficientAttention
from ..utils.tensors import trunc_normal_
from ..utils.pos_embs import UnitEncoding
from .base import Prober

class EfficientPooler(nn.Module):
    def __init__(
        self,
        num_patches = 16,
        max_frames = 16,
        tubelet_size = 2,
        embed_dim = 768,
        num_heads = 12,
        num_queries = 32,
        mlp_ratio = 4.0,
        depth = 1,
        norm_layer = nn.LayerNorm,
        init_std = 0.02,
        qkv_bias = True,
        use_activation_checkpointing = False,
    ):
        super().__init__()

        self.use_activation_checkpointing = use_activation_checkpointing
        self.num_patches = num_patches
        self.timestep = max_frames // tubelet_size
        self.embed_dim = embed_dim

        self.query_tokens = nn.Parameter(torch.randn(1, self.timestep, num_queries, embed_dim) * init_std)
        self.pos_encode = UnitEncoding(
            embed_dim = embed_dim, num_patches = self.num_patches
        )
        self.efficient_probe = EfficientAttention(
            dim=embed_dim,
            qkv_bias=qkv_bias,
            num_queries=num_queries,
            num_subspace=1,
            use_sdpa=False
        )

        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer,
                ) for _ in range(depth - 1)
            ])
        else:
            self.blocks = None

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        rescale(self.efficient_probe.v.weight.data, layer_id + 1)

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

    def forward(self, x: torch.Tensor):
        if self.blocks is not None:
            for block in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, False, None, use_reentrant=False
                    )
                else:
                    x = block(x)

        batch_size, num_tokens, embed_dim = x.shape
        patches_per_frame = self.num_patches ** 2

        if num_tokens % patches_per_frame != 0:
            raise ValueError(
                f"Input token count ({num_tokens}) must be divisible by patches per frame ({patches_per_frame})"
            )

        target_timestep = num_tokens // patches_per_frame

        x = self.pos_encode(x, target_timestep)
        query_tokens = self.interp_q(self.query_tokens, target_timestep)
        query_tokens = query_tokens.repeat(batch_size, 1, 1, 1)

        out = self.efficient_probe(x, cls_token=query_tokens)
        return out

    def interp_q(self, queries: torch.Tensor, tgt_size: int) -> torch.Tensor:
        if self.timestep != tgt_size:
            B, T, Q, D = queries.shape
            queries = queries.permute(0, 3, 2, 1).reshape(B, D * Q, T)
            queries = F.interpolate(queries, size=tgt_size, mode="linear", align_corners=False)
            queries = queries.reshape(B, Q, D, tgt_size).permute(0, 3, 2, 1)
        return queries


class EfficientProbe(Prober):
    def __init__(
        self,
        output_dim = 1,
        num_patches = 16,
        max_frames = 16,
        tubelet_size = 2,
        embed_dim = 768,
        num_heads = 12,
        num_queries = 16,
        mlp_ratio = 4.0,
        depth = 1,
        norm_layer = nn.LayerNorm,
        init_std = 0.02,
        qkv_bias = True,
        use_activation_checkpointing=False,
        init_scales = None,
        init_shifts = None,
        *args, **kwargs
    ):
        super().__init__(output_dim=output_dim, init_scales=init_scales, init_shifts=init_shifts)
        self.pooler = EfficientPooler(
            num_patches=num_patches,
            max_frames=max_frames,
            tubelet_size=tubelet_size,
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            use_activation_checkpointing=use_activation_checkpointing,
        )

        self.linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2, bias = True),
                nn.LeakyReLU(),
                nn.Linear(embed_dim // 2, 1)
            ) for _ in range(output_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(x)
        head_outputs = [head(pooled) for head in self.linear]
        x = torch.cat(head_outputs, dim=-1)
        return self.apply_output_affine(x)


if __name__ == '__main__':
    num_patches = 16
    max_frames = 18
    tubelet_size = 2
    img_size = 256

    device = torch.device('cuda')
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
    encoder = encoder.to(device)
    probe = EfficientProbe(
        output_dim = 2,
        num_patches = num_patches,
        max_frames = max_frames,
        tubelet_size = tubelet_size,
        depth = 4,
        num_heads = 8,
        embed_dim = encoder.embed_dim,
        use_activation_checkpointing = True
    ).to(device)

    with torch.no_grad():
        with torch.autocast("cuda", dtype = torch.bfloat16):
            input = torch.rand((4, 3, max_frames + 2, img_size, img_size)).to(device)
            output = probe(encoder(input))
            print(output.shape)