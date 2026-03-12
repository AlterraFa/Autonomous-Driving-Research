import torch
import torch.nn as nn
import copy
from ....models.vision_transformer import VisionTransformer as Enc
from ....models.latent_dreamer import VisionTransformerPredictorAC as LPred
from ....models.action_predictor import TransformerActionPredictor as APred

from ....utils.logger import Logger

from torch import distributed as dist

logger = Logger(__name__)

def compile_model(
    enc_cfg: dict = None,
    lpred_cfg: dict = None,
    apred_cfg: dict = None,
    max_frames: int = 16,
    depth: int = 12,
    device = torch.device('cpu')
):
    
    name = enc_cfg.get('name', "Not found")
    repo = enc_cfg.get('load_from', 'Not found')
    
    # -- Encoder has sdpa and grad checkpoint enabled
    model = torch.hub.load(repo, name, trust_repo=True)
    encoder: Enc = model[0]
    encoder.use_activation_checkpointing = enc_cfg['use_activation_checkpointing']
    
    # -- Configure encoder with image/video parameters
    if hasattr(encoder, 'img_size'):
        encoder.img_size = enc_cfg.get('crop_size', 224)
    if hasattr(encoder, 'patch_size'):
        encoder.patch_size = enc_cfg.get('patch_size', 16)
    if hasattr(encoder, 'tubelet_size'):
        encoder.tubelet_size = enc_cfg.get('tubelet_size', 2)
    
    encoder.to(device)

    # -- Initialize Latent Predictor (VisionTransformerPredictorAC)
    latent_predictor = LPred(
        img_size=lpred_cfg.get('crop_size', 224),
        patch_size=lpred_cfg.get('patch_size', 16),
        num_frames=lpred_cfg.get('ctx_fpcs', max_frames) + lpred_cfg.get('pred_fpcs', max_frames),
        tubelet_size=lpred_cfg.get('tubelet_size', 2),
        action_pframe=lpred_cfg.get('action_pframe', 1),
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=lpred_cfg.get('pred_embed_dim', 1024),
        depth=lpred_cfg.get('depth', depth),
        num_heads=lpred_cfg.get('num_heads', 16),
        mlp_ratio=lpred_cfg.get('mlp_ratio', 4.0),
        qkv_bias=lpred_cfg.get('qkv_bias', True),
        qk_scale=lpred_cfg.get('qk_scale', None),
        drop_rate=lpred_cfg.get('drop_rate', 0.0),
        attn_drop_rate=lpred_cfg.get('attn_drop_rate', 0.0),
        drop_path_rate=lpred_cfg.get('drop_path_rate', 0.0),
        norm_layer=nn.LayerNorm,
        init_std=lpred_cfg.get('init_std', 0.1),
        uniform_power=lpred_cfg.get('uniform_power', True),
        use_silu=lpred_cfg.get('use_silu', False),
        wide_silu=lpred_cfg.get('wide_silu', True),
        is_frame_causal=lpred_cfg.get('is_frame_causal', True),
        use_activation_checkpointing=lpred_cfg.get('use_activation_checkpointing', False),
        use_rope=lpred_cfg.get('use_rope', True),
        action_embed_dim=lpred_cfg.get('action_embed_dim', 256),
        use_sdpa=lpred_cfg.get('use_sdpa', False),

    )
    latent_predictor.to(device)

    # -- Initialize Action Predictor (TransformerActionPredictor)
    action_predictor = APred(
        img_size=apred_cfg.get('crop_size', 224),
        patch_size=apred_cfg.get('patch_size', 16),
        ctx_nframes=apred_cfg.get('ctx_fpcs', 1),
        goal_nframes=apred_cfg.get('pred_fpcs', 1),
        tubelet_size=apred_cfg.get('tubelet_size', 2),
        action_per_step=apred_cfg.get('action_pframe', 1),
        embed_dim=encoder.embed_dim,
        action_embed_dim=apred_cfg.get('action_embed_dim', 1024),
        depth=apred_cfg.get('depth', depth // 2),  # Usually smaller than latent predictor
        num_heads=apred_cfg.get('num_heads', 16),
        mlp_ratio=apred_cfg.get('mlp_ratio', 4.0),
        qkv_bias=apred_cfg.get('qkv_bias', True),
        qk_scale=apred_cfg.get('qk_scale', None),
        drop_rate=apred_cfg.get('drop_rate', 0.0),
        attn_drop_rate=apred_cfg.get('attn_drop_rate', 0.0),
        drop_path_rate=apred_cfg.get('drop_path_rate', 0.0),
        norm_layer=apred_cfg.get('norm_layer', nn.LayerNorm),
        init_std=apred_cfg.get('init_std', 0.1),
        uniform_power=apred_cfg.get('uniform_power', True),
        use_silu=apred_cfg.get('use_silu', False),
        wide_silu=apred_cfg.get('wide_silu', True),
        use_activation_checkpointing=apred_cfg.get('use_activation_checkpointing', False),
        use_rope=apred_cfg.get('use_rope', True),
        use_sdpa=apred_cfg.get('use_sdpa', False),
    )
    action_predictor.to(device)

    
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.INFO(f"Encoder number of parameters: {count_params(encoder)}")
    logger.INFO(f"Action Predictor number of parameters: {count_params(action_predictor)}")
    logger.INFO(f"Latent Predictor number of parameters: {count_params(latent_predictor)}")
    
    return encoder, latent_predictor, action_predictor

def load_hub_model(repo, model_name, **kwargs):
    # 1. Determine rank safely
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # 2. Rank 0 downloads the model
    if rank == 0:
        logger.INFO(f"Rank 0: Initializing download/cache check for {repo}...")
        _ = torch.hub.load(repo, model_name, trust_repo=True, **kwargs)
        logger.INFO("Rank 0: Model cached successfully.")

    if dist.is_initialized():
        dist.barrier() 

    return torch.hub.load(repo, model_name, trust_repo=True, **kwargs)