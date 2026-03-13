import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .utils.modules import ACBlock, GCBlock, build_gc_causal_attn_mask
from .utils.pos_embs import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed
from .utils.tensors import trunc_normal_


class TransformerActionPredictor(nn.Module):
    def  __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        ctx_nframes=1,
        goal_nframes=1,
        tubelet_size=2,
        action_per_step = 1,
        embed_dim=768,
        action_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=True,
        use_sdpa = False,
        **kwargs):
        super().__init__()
        
        img_size = _pair(img_size)
        
        self.grid_H, self.grid_W = (int(size / patch_size) for size in img_size)
        self.grid_ctx  = ctx_nframes // tubelet_size
        self.grid_goal = goal_nframes // tubelet_size
        self.token_pframes = self.grid_H * self.grid_W
        self.action_pframe = action_per_step
        
        self.ctx_nframes  = ctx_nframes
        self.goal_nframes = goal_nframes

        self.use_ckpt = use_activation_checkpointing
        self.use_rope = use_rope
            
        self.total_timestep = self.grid_ctx + self.grid_goal
        self.action_embed = nn.Parameter(torch.empty(1, self.total_timestep * self.action_pframe, action_embed_dim))
        self.action_embed_dim = action_embed_dim
        
        self.to_action = nn.Linear(embed_dim, action_embed_dim, bias = True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.norm = nn.LayerNorm(action_embed_dim)
        self.action_blocks = nn.ModuleList(
            [
                ACBlock(
                    dim = action_embed_dim,
                    num_heads = num_heads,
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    qk_scale = qk_scale,
                    drop = drop_rate,
                    attn_drop = attn_drop_rate,
                    drop_path = dpr[i],
                    act_layer = nn.SiLU if use_silu else nn.GELU,
                    wide_silu = wide_silu,
                    norm_layer = norm_layer,
                    use_sdpa = use_sdpa,
                    is_causal = False,
                    grid_size = self.grid_H,
                    use_rope = use_rope
                ) for i in range(depth)
            ]
        )

        self.init_std = init_std
        self.apply(self._init_weights)
        self._init_action()
        self._rescale_blocks()
        
    def _init_action(self):
        nn.init.trunc_normal_(self.action_embed, std=self.init_std)

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

        for layer_id, layer in enumerate(self.action_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.action_blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, ctxt_z: torch.Tensor, goal_z: torch.Tensor = None):

        if goal_z is None:
            z = ctxt_z
        else:
            z = torch.concat([ctxt_z, goal_z], dim=1)
        z = self.to_action(z)

        B, tokens, _ = z.shape
        
        # -- Handle case where there are less timestep
        total_timestep = tokens // self.token_pframes
        frame_tokens = z.view(B, total_timestep, self.token_pframes, self.action_embed_dim) # -- B, T, H*W, D
        action_tokens = self.action_embed.expand(B, -1, -1)[:, :total_timestep * self.action_pframe, :]
        action_tokens = action_tokens.view(
            B, total_timestep, self.action_pframe, self.action_embed_dim
        ) # -- B, T, A, D

        # -- Interleave frame tokens with action tokens per frame
        # -- Reason: The model must create action from goal frames
        # -- We will use cross attention between image and action latent for prediction
        x = torch.cat([frame_tokens, action_tokens], dim=2).reshape(
            B, total_timestep * (self.token_pframes + self.action_pframe), self.action_embed_dim
        ) # -- B, T, H*W + A, D -> B, T(H*W + A), D 

        for i, blk in enumerate(self.action_blocks):
            if self.use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x, 
                    mask = None,
                    attn_mask = None,
                    T = total_timestep,
                    H = self.grid_H,
                    W = self.grid_W,
                    action_tokens = self.action_pframe,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x, 
                    mask=None,
                    attn_mask=None,
                    T=total_timestep,
                    H=self.grid_H,
                    W=self.grid_W,
                    action_tokens = self.action_pframe
                )
                
        a = x.reshape(B, total_timestep, -1, self.action_embed_dim)[:, :, -self.action_pframe:, :]
        a = a.reshape(B, -1, self.action_embed_dim)

                
        return a

class ActionTransformerPredictorGC(nn.Module):
    def  __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        max_frames=16,
        tubelet_size=2,
        action_per_step = 1,
        embed_dim=768,
        action_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=True,
        use_sdpa = False,
        **kwargs):
        super().__init__()
        
        img_size = _pair(img_size)
        
        self.grid_H, self.grid_W = (int(size / patch_size) for size in img_size)
        self.token_pframes = self.grid_H * self.grid_W
        self.action_pstep = action_per_step
        
        self.use_ckpt = use_activation_checkpointing
        self.use_rope = use_rope
            
        self.max_tubelets = max_frames // tubelet_size
        self.action_embed = nn.Parameter(torch.empty(1, (self.max_tubelets - 1) * self.action_pstep, action_embed_dim))
        self.action_embed_dim = action_embed_dim
        
        self.to_action = nn.Linear(embed_dim, action_embed_dim, bias = True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.norm = nn.LayerNorm(action_embed_dim)
        self.action_blocks = nn.ModuleList(
            [
                GCBlock(
                    dim = action_embed_dim,
                    num_heads = num_heads,
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    qk_scale = qk_scale,
                    drop = drop_rate,
                    attn_drop = attn_drop_rate,
                    drop_path = dpr[i],
                    act_layer = nn.SiLU if use_silu else nn.GELU,
                    wide_silu = wide_silu,
                    norm_layer = norm_layer,
                    use_sdpa = use_sdpa,
                    is_causal = False,
                    grid_size = self.grid_H,
                    use_rope = use_rope
                ) for i in range(depth)
            ]
        )
        
        self.ctx_mask, self.goal_pad = build_gc_causal_attn_mask(
            self.max_tubelets, 
            self.grid_H, 
            self.grid_W, 
            add_tokens = self.action_pstep
        )        
        self.mask_idx = torch.arange(self.max_tubelets * self.token_pframes, dtype = torch.long)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._init_action()
        self._rescale_blocks()
        
    def _init_action(self):
        nn.init.trunc_normal_(self.action_embed, std=self.init_std)

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

        for layer_id, layer in enumerate(self.action_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.action_blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, context: torch.Tensor, goal: torch.Tensor, goal_pos = -1):

        B, N_ctx, _ = context.shape
        B, N_goal, _ = goal.shape
        
        # -- Handle case where there are less timestep
        ctx_timestep = N_ctx // self.token_pframes
        total_timestep = ctx_timestep + N_goal // self.token_pframes
        
        context = self.to_action(context)
        goal    = self.to_action(goal)

        context = context.view(B, ctx_timestep, self.token_pframes, self.action_embed_dim) # -- B, T, H*W, D

        action_tokens = self.action_embed.expand(B, -1, -1)[:, :ctx_timestep * self.action_pstep, :]
        action_tokens = action_tokens.view(
            B, ctx_timestep, self.action_pstep, self.action_embed_dim
        ) # -- B, T, A, D

        # -- Interleave frame tokens with action tokens per frame
        ctx_a = torch.cat([context, action_tokens], dim=2).reshape(
            B, ctx_timestep * (self.token_pframes + self.action_pstep), self.action_embed_dim
        ) # -- (B, T, H*W + A, D) -> (B, T*(H*W + A), D) 

        # -- Action attends to past context and goal image
        x = torch.cat([ctx_a, goal], dim = 1)
        ctx_mask = self.ctx_mask[: ctx_a.size(1), :ctx_a.size(1)].to(ctx_a.device)
        attn_mask = F.pad(ctx_mask, (0, self.goal_pad, 0, self.goal_pad), value = True)
        
        g_start = goal_pos * N_goal
        g_end   = (goal_pos + 1) * N_goal
        mask_idx = torch.cat([self.mask_idx[:N_ctx], self.mask_idx[g_start: (g_end if g_end != 0 else None)]]).to(ctx_a.device)
        
        for i, blk in enumerate(self.action_blocks):
            if self.use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x, 
                    mask = mask_idx,
                    attn_mask = attn_mask,
                    T = total_timestep,
                    H = self.grid_H,
                    W = self.grid_W,
                    apstep = self.action_pstep,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x, 
                    mask=mask_idx,
                    attn_mask=attn_mask,
                    T=total_timestep,
                    H=self.grid_H,
                    W=self.grid_W,
                    apstep = self.action_pstep
                )
        
        ctx_a = x[:, :-self.token_pframes]    
        a = ctx_a.reshape(B, ctx_timestep, -1, self.action_embed_dim)[:, :, -self.action_pstep:, :]
        a = a.reshape(B, -1, self.action_embed_dim)

        a = self.norm(a)
                
        return a
            
if __name__ == "__main__":
    device = torch.device('cuda')

    img_size = 224
    patch_size = 16
    max_frames = 16
    tubelet_size = 2
    action_per_step = 3
    
    tokens_pframe = (img_size // patch_size) ** 2
    
    
    model = ActionTransformerPredictorGC(
        img_size = (img_size, ) * 2,
        patch_size = patch_size,
        max_frames = max_frames,
        tubelet_size = tubelet_size, 
        embed_dim = 512,
        action_per_step = action_per_step,
        action_embed_dim = 512,
        depth = 4,
        num_heads = 8, 
        use_activation_checkpointing = True
    ).to(device)
    
    context = torch.rand((6, tokens_pframe * (max_frames // tubelet_size - 1), 512)).to(device)
    goal = torch.rand((6, tokens_pframe, 512)).to(device)

    with torch.no_grad():
        with torch.autocast('cuda', torch.bfloat16):
            output = model(context, goal)
            print(output.shape)