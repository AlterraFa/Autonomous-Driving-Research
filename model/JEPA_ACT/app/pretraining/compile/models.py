import torch.nn as nn
from model.JEPA_ACT.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from model.JEPA_ACT.models.vision_transformer import VisionTransformer
from model.JEPA_ACT.models.predictor import VisionTransformerPredictor


def compile_model(
    img_size: tuple,
    patch_size: int,
    fpc: int,
    tubelet_size: int,
    enc_embed_dim: int,
    enc_depth: int,
    enc_num_heads: int,
    pred_embed_dim: int,
    pred_depth: int,
    pred_num_heads: int,
    enc_drop_rate: float = 0.0,
    enc_attn_drop_rate: float = 0.0, 
    enc_drop_path_rate: float = 0.0, 
    pred_drop_rate: float = 0.0,
    pred_attn_drop_rate: float = 0.0, 
    pred_drop_path_rate: float = 0.0, 
    use_silu: bool = True,
    use_rope: bool = True,
    use_activation_checkpointing: bool = False,
    num_unique_fpcs = 1,
):
    encoder = VisionTransformer(
        img_size = img_size,
        patch_size = patch_size,
        num_frames = fpc,
        tubelet_size = tubelet_size,
        embed_dim = enc_embed_dim,
        depth = enc_depth,
        num_heads = enc_num_heads,
        drop_rate = enc_drop_rate,
        attn_drop_rate = enc_attn_drop_rate,
        drop_path_rate = enc_drop_path_rate,
        use_silu = use_silu,
        wide_silu = True,
        use_sdpa = True,
        use_rope = use_rope,
        use_activation_checkpointing = use_activation_checkpointing
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = VisionTransformerPredictor(
        img_size = img_size,
        patch_size = patch_size,
        num_frames = fpc,
        tubelet_size = tubelet_size,
        embed_dim = enc_embed_dim,
        predictor_embed_dim = pred_embed_dim,
        depth = pred_depth,
        num_heads = pred_num_heads,
        drop_rate = pred_drop_rate,
        attn_drop_rate = pred_attn_drop_rate,
        drop_path_rate = pred_drop_path_rate,
        use_silu = use_silu,
        wide_silu = True,
        use_rope = use_rope,
        use_activation_checkpointing = use_activation_checkpointing ,
        use_mask_tokens = True,
        num_mask_tokens = num_unique_fpcs
    )
    predictor = PredictorMultiSeqWrapper(predictor)
    
    
    return encoder, predictor