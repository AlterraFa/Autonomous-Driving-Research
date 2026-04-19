import torch
import torch.nn as nn
import copy
from models.vision_transformer import VisionTransformer as Enc
from models.latent_dreamer import VisionTransformerPredictorAC as LPred
from models.action_predictor import ActionTransformerPredictorGC as APred
from models.straightening_filter import FILTER_REGISTRY, Aggregation

from utils.logger import Logger

logger = Logger(__name__)


def _log_module_params(module_name: str, params: dict):
    logger.DEBUG(f"{module_name} config:")
    logger.DEBUG(params)

def compile_model(
    enc_cfg: dict = None,
    lpred_cfg: dict = None,
    apred_cfg: dict = None,
    filter_cfg: dict = None,
    max_frames: int = 16,
    depth: int = 12,
    device = torch.device('cpu')
):
    enc_cfg = enc_cfg or {}
    lpred_cfg = lpred_cfg or {}
    apred_cfg = apred_cfg or {}
    filter_cfg = filter_cfg or {}

    encoder_params = {
        'device': str(device),
        'enc_cfg': enc_cfg,
        'filter_cfg': filter_cfg,
    }
    _log_module_params("Encoder", encoder_params)
    
    name = enc_cfg.get('name', "Not found")
    repo = enc_cfg.get('load_from', 'Not found')
    source = enc_cfg.get('source', 'github')
    
    # -- Encoder has sdpa and grad checkpoint enabled
    model = torch.hub.load(repo, name, trust_repo=True, source=source, pretrained=False, skip_validation=True)
    encoder = model[0]
    encoder.use_activation_checkpointing = enc_cfg['use_activation_checkpointing']
    
    # -- Configure encoder with image/video parameters
    if hasattr(encoder, 'img_size'):
        encoder.img_size = enc_cfg.get('crop_size', 224)
    if hasattr(encoder, 'patch_size'):
        encoder.patch_size = enc_cfg.get('patch_size', 16)
    if hasattr(encoder, 'tubelet_size'):
        encoder.tubelet_size = enc_cfg.get('tubelet_size', 2)
        
    encoder: nn.Module = encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.to(device)

    
    # -- Initialize Latent Predictor (VisionTransformerPredictorAC)
    latent_params = {
        'img_size': lpred_cfg.get('crop_size', 224),
        'patch_size': lpred_cfg.get('patch_size', 16),
        'num_frames': lpred_cfg.get('fpcs', max_frames),
        'tubelet_size': lpred_cfg.get('tubelet_size', 2),
        'action_pframe': lpred_cfg.get('action_pframe', 1),
        'embed_dim': encoder.embed_dim,
        'predictor_embed_dim': lpred_cfg.get('pred_embed_dim', 1024),
        'depth': lpred_cfg.get('depth', depth),
        'num_heads': lpred_cfg.get('num_heads', 16),
        'mlp_ratio': lpred_cfg.get('mlp_ratio', 4.0),
        'qkv_bias': lpred_cfg.get('qkv_bias', True),
        'qk_scale': lpred_cfg.get('qk_scale', None),
        'drop_rate': lpred_cfg.get('drop_rate', 0.0),
        'attn_drop_rate': lpred_cfg.get('attn_drop_rate', 0.0),
        'drop_path_rate': lpred_cfg.get('drop_path_rate', 0.0),
        'norm_layer': lpred_cfg.get('norm_layer', "LayerNorm"),
        'init_std': lpred_cfg.get('init_std', 0.1),
        'out_norm': lpred_cfg.get('out_norm', 'LayerNorm'),
        'uniform_power': lpred_cfg.get('uniform_power', True),
        'use_silu': lpred_cfg.get('use_silu', False),
        'wide_silu': lpred_cfg.get('wide_silu', True),
        'is_frame_causal': lpred_cfg.get('is_frame_causal', True),
        'use_activation_checkpointing': lpred_cfg.get('use_activation_checkpointing', False),
        'use_rope': lpred_cfg.get('use_rope', True),
        'action_embed_dim': apred_cfg.get('action_embed_dim', 256),
        'use_sdpa': lpred_cfg.get('use_sdpa', False),
    }
    _log_module_params("Latent Predictor", latent_params)

    latent_predictor = LPred(
        img_size=latent_params['img_size'],
        patch_size=latent_params['patch_size'],
        num_frames=latent_params['num_frames'],
        tubelet_size=latent_params['tubelet_size'],
        action_pframe=latent_params['action_pframe'],
        embed_dim=latent_params['embed_dim'],
        predictor_embed_dim=latent_params['predictor_embed_dim'],
        depth=latent_params['depth'],
        num_heads=latent_params['num_heads'],
        mlp_ratio=latent_params['mlp_ratio'],
        qkv_bias=latent_params['qkv_bias'],
        qk_scale=latent_params['qk_scale'],
        drop_rate=latent_params['drop_rate'],
        attn_drop_rate=latent_params['attn_drop_rate'],
        drop_path_rate=latent_params['drop_path_rate'],
        norm_layer=latent_params['norm_layer'],
        init_std=latent_params['init_std'],
        uniform_power=latent_params['uniform_power'],
        use_silu=latent_params['use_silu'],
        wide_silu=latent_params['wide_silu'],
        is_frame_causal=latent_params['is_frame_causal'],
        use_activation_checkpointing=latent_params['use_activation_checkpointing'],
        use_rope=latent_params['use_rope'],
        action_embed_dim=latent_params['action_embed_dim'],
        use_sdpa=latent_params['use_sdpa'],
        out_norm=latent_params['out_norm']

    )
    latent_predictor.to(device)

    # -- Initialize Action Predictor (TransformerActionPredictor)
    action_params = {
        'img_size': apred_cfg.get('crop_size', 224),
        'patch_size': apred_cfg.get('patch_size', 16),
        'max_frames': apred_cfg.get('fpcs', 1),
        'tubelet_size': apred_cfg.get('tubelet_size', 2),
        'action_per_step': apred_cfg.get('action_pframe', 1),
        'embed_dim': encoder.embed_dim,
        'action_embed_dim': apred_cfg.get('action_embed_dim', 1024),
        'depth': apred_cfg.get('depth', depth // 2),
        'num_heads': apred_cfg.get('num_heads', 16),
        'mlp_ratio': apred_cfg.get('mlp_ratio', 4.0),
        'qkv_bias': apred_cfg.get('qkv_bias', True),
        'qk_scale': apred_cfg.get('qk_scale', None),
        'drop_rate': apred_cfg.get('drop_rate', 0.0),
        'attn_drop_rate': apred_cfg.get('attn_drop_rate', 0.0),
        'drop_path_rate': apred_cfg.get('drop_path_rate', 0.0),
        'norm_layer': apred_cfg.get('norm_layer', "LayerNorm"),
        'out_norm': apred_cfg.get('out_norm', 'LayerNorm'),
        'init_std': apred_cfg.get('init_std', 0.1),
        'uniform_power': apred_cfg.get('uniform_power', True),
        'use_silu': apred_cfg.get('use_silu', False),
        'wide_silu': apred_cfg.get('wide_silu', True),
        'use_activation_checkpointing': apred_cfg.get('use_activation_checkpointing', False),
        'use_rope': apred_cfg.get('use_rope', True),
        'use_sdpa': apred_cfg.get('use_sdpa', False),
    }
    _log_module_params("Action Predictor", action_params)

    action_predictor = APred(
        img_size=action_params['img_size'],
        patch_size=action_params['patch_size'],
        max_frames=action_params['max_frames'],
        tubelet_size=action_params['tubelet_size'],
        action_per_step=action_params['action_per_step'],
        embed_dim=action_params['embed_dim'],
        action_embed_dim=action_params['action_embed_dim'],
        depth=action_params['depth'],  # Usually smaller than latent predictor
        num_heads=action_params['num_heads'],
        mlp_ratio=action_params['mlp_ratio'],
        qkv_bias=action_params['qkv_bias'],
        qk_scale=action_params['qk_scale'],
        drop_rate=action_params['drop_rate'],
        attn_drop_rate=action_params['attn_drop_rate'],
        drop_path_rate=action_params['drop_path_rate'],
        norm_layer=action_params['norm_layer'],
        init_std=action_params['init_std'],
        uniform_power=action_params['uniform_power'],
        use_silu=action_params['use_silu'],
        wide_silu=action_params['wide_silu'],
        use_activation_checkpointing=action_params['use_activation_checkpointing'],
        use_rope=action_params['use_rope'],
        use_sdpa=action_params['use_sdpa'],
        out_norm=action_params['out_norm']
    )
    action_predictor.to(device)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    logger.DEBUG(f"Encoder number of parameters: {count_params(encoder)}")
    logger.DEBUG(f"Action Predictor number of parameters: {count_params(action_predictor)}")
    logger.DEBUG(f"Latent Predictor number of parameters: {count_params(latent_predictor)}")

    
    return encoder, latent_predictor, action_predictor