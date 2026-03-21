import torch
import torch.nn as nn
import copy
import inspect
from models.vision_transformer import VisionTransformer as Enc
from models.probes import available_probe_classes, build_probe, get_probe_class

from utils.logger import Logger

from torch import distributed as dist

logger = Logger(__name__)

def compile_model(
    enc_cfg: dict = None,
    probe_cfg: dict = None,
    device = torch.device('cpu')
):
    
    name   = enc_cfg.get('name', "Not found")
    repo   = enc_cfg.get('load_from', 'Not found')
    source = enc_cfg.get('source', 'github')
    
    # -- Encoder has sdpa and grad checkpoint enabled
    logger.INFO(f"Loading the model from {source}")
    model = torch.hub.load(repo, name, source=source, pretrained=False, trust_repo=True, skip_validation = True)
    encoder: Enc = model[0]
    encoder.use_activation_checkpointing = enc_cfg['use_activation_checkpointing']
    
    # -- Configure encoder with image/video parameters
    if hasattr(encoder, 'img_size'):
        encoder.img_size = enc_cfg.get('crop_size', 224)
    if hasattr(encoder, 'patch_size'):
        encoder.patch_size = enc_cfg.get('patch_size', 16)
    if hasattr(encoder, 'tubelet_size'):
        encoder.tubelet_size = enc_cfg.get('tubelet_size', 2)
    

    # -- Initialize Probe
    probe_cfg = dict(probe_cfg or {})
    crop_size = enc_cfg.get('crop_size', 224)
    patch_size = probe_cfg.get('patch_size', enc_cfg.get('patch_size', 16))
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    num_patches = crop_size // patch_size
    computed_probe_args_all = {
        "embed_dim": encoder.embed_dim,
        "patch_size": patch_size,
        "num_patches": num_patches,
    }
    probe_name = probe_cfg.get("name")
    if not probe_name:
        raise ValueError("model.probe.name must be provided in config")

    probe_cls = get_probe_class(probe_name)
    probe_sig = inspect.signature(probe_cls.__init__)
    accepted_args = {
        param.name
        for param in probe_sig.parameters.values()
        if param.name not in {"self", "args", "kwargs"}
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    computed_probe_args = {
        key: value for key, value in computed_probe_args_all.items() if key in accepted_args
    }
    probe_args = probe_cfg | computed_probe_args

    logger.INFO("Computed probe args:", computed_probe_args)
    probe = build_probe(**probe_args)

    encoder.to(device)
    probe.to(device)
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.INFO(f"Encoder number of parameters: {count_params(encoder)}")
    logger.INFO(f"{probe_args.get('name', probe.__class__.__name__)} number of parameters: {count_params(probe)}")
    
    return encoder, probe
