import os, sys
import resource
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .compile import (
    compile_model,
    compile_transform,
)
from datasets.dataset import StraighteningProbeDataset
from utils.distributed import init_distributed
from utils.logger import Logger

logger = Logger(__name__)
def main(args: dict, path: str):
    train_cfg: dict = args.get("train", {})
    crop_size = train_cfg.get('crop_size', 256)
    patch_size = train_cfg.get('patch_size', 16)
    tubelet_size = train_cfg.get('tubelet_size', 2)
    fpcs = train_cfg.get('fpcs', 12)

    loader_cfg: dict = args.get('loader_setup', {})
    num_workers = loader_cfg.get('num_workers', 4)
    persistent_workers = loader_cfg.get('persistent_workers', True)
    pin_mem = loader_cfg.get('pin_mem', True)

    model_cfg: dict = args.get("model", {})
    enc_cfg = model_cfg.get('enc', {})
    probe_cfg = model_cfg.get('probe', {})
    world_model_cfg = model_cfg.get('world_model', {})

    augment_cfg: dict = args.get('data_aug', {})
    auto_augment = augment_cfg.get('auto_augment', False)
    horizontal_flip = augment_cfg.get('horizontal_flip', False)
    motion_shift = augment_cfg.get('motion_shift', False)
    random_aspect_ratio = augment_cfg.get('random_resize_aspect_ratio', (1.0, 1.0))
    random_resize_scale = augment_cfg.get('random_resize_scale', (1.0, 1.0))
    reprob = augment_cfg.get('reprob', 0.0)

    meta_cfg: dict = args.get('meta', {})
    dtype = meta_cfg.get('dtype', 'bfloat16')
    seed = meta_cfg.get('seed', 239)

    # ======================== Distributed setup ========================
    world_size, rank = init_distributed()
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        logger.CUSTOM("SUCCESS", f"DDP enabled (world_size={world_size}, rank={rank})")
    else:
        logger.INFO("DDP disabled (single-GPU/single-process mode)")

    if dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    torch.manual_seed(seed)
    torch.cuda.set_device(rank)
    device_type = f'cuda:{rank}'
    device = torch.device(device_type)

    transform = compile_transform(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=random_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    eval_paths = [
        '../Autonomous_Dataset/carla/Test/recording_20260416_145243_test_spatial',
    ]

    # Transformed dataset — gives model-ready tensors
    model_dataset = StraighteningProbeDataset(
        data_paths=eval_paths,
        shared_transform=transform,
        waypoint_key=train_cfg.get('waypoint_key', 'midlane_wp'),
        n_waypoints=train_cfg.get('n_waypoints', 12),
        wp_clip=train_cfg.get('wp_clip', None),
        wp_normalize=train_cfg.get('wp_normalize', False),
        wp_center=train_cfg.get('wp_center', None),
    )
    
    non_transform = StraighteningProbeDataset(
        data_paths=eval_paths,
        shared_transform=None,
        waypoint_key=train_cfg.get('waypoint_key', 'midlane_wp'),
        n_waypoints=train_cfg.get('n_waypoints', 12),
        wp_clip=train_cfg.get('wp_clip', None),
        wp_normalize=train_cfg.get('wp_normalize', False),
        wp_center=train_cfg.get('wp_center', None),
    )

    logger.WARNING(f"Eval dataset size: {len(model_dataset)}")

    # ======================== Load models ========================
    world_model, _ = compile_model(
        enc_cfg=enc_cfg,
        probe_cfg=probe_cfg,
        world_model_cfg=world_model_cfg,
        device=device,
        detailed_out=True
    )

    # Extract individual components from FrozenWorldModel
    encoder = world_model.encoder
    filterer = world_model.filterer
    target_filterer = world_model.target_filterer
    apred = world_model.apred
    lpred = world_model.lpred
    normalize_reps = world_model.normalize_reps
    normalize_actions = world_model.normalize_actions
    auto_steps = world_model.auto_steps
    tokens_pframe = world_model.tokens_pframe

    logger.INFO(f"normalize_reps={normalize_reps}, normalize_actions={normalize_actions}, "
                f"auto_steps={auto_steps}, tokens_pframe={tokens_pframe}")

    # ======================== Forward helpers (matching training exactly) ========================
    def to_latent(c: torch.Tensor):
        """encoder + layer_norm — matches training's to_latent()"""
        latent = encoder(c)
        if normalize_reps:
            latent = F.layer_norm(latent, (latent.size(-1),))
        return latent

    def forward_context(latent: torch.Tensor, H: int = None, W: int = None):
        """online filterer + layer_norm — matches training's forward_context()"""
        h = filterer(latent, H, W)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def forward_target(latent: torch.Tensor, H: int = None, W: int = None):
        """target_filterer + layer_norm — matches training's forward_target()"""
        h = target_filterer(latent, H, W)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h
    def _step_action(h, g):
        _a: torch.Tensor = apred(h, g)
        if normalize_actions:
            _a = F.layer_norm(_a, (_a.size(-1), ))
        return _a
    
    def _step_prediction(h, a):
        _z: torch.Tensor = lpred(h, a)
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1), ))
        return _z

    save_dir = "./latent"
    # ======================== Eval loop ========================
    for idx in range(len(model_dataset)):
        video_data, _ = non_transform[idx]
        model_buffer, _ = model_dataset[idx]

        clips = model_buffer.unsqueeze(0).to(device)  # (1, C, T, H, W)

        with torch.no_grad(), torch.amp.autocast(device_type, dtype = dtype, enabled = mixed_precision):
                # ---- Exact replica of straightening/train.py forward ----

                # 1. Encode (encoder + layer_norm, same as training's to_latent)
                latent_ctx = to_latent(clips[:, :, :-1])
                latent_goal = to_latent(clips[:, :, -1:])
                H_patches = clips.shape[3] // patch_size
                W_patches = clips.shape[4] // patch_size
                T = (latent_ctx.shape[1] + latent_goal.shape[1]) // tokens_pframe

                h_ctx  = forward_context(latent_ctx,)
                h_goal = forward_target(latent_goal,)
                h = torch.cat([h_ctx, h_goal], dim=1)

               
                action, z_full = world_model(clips[:, :, :1], clips[:, :, -1:], perturb_coeff = 0.0) 
                z_ar = z_full[:, world_model.tokens_pframe:]

                # # 4. Latent loss (all timesteps — matches training default)
                loss_ar = latent_loss(h, z_ar, tokens_pframe, [-1, -2, -3])  
        
        z_full = z_full.float().cpu().numpy()
        z_full = z_full.reshape(6, -1, encoder.embed_dim)
        h = h.reshape(6, -1, encoder.embed_dim)
        
        seq_dir = os.path.join(save_dir, f"seq_{idx:05d}")
        os.makedirs(seq_dir, exist_ok = True)
        paths = _write_images(video_data, seq_dir)
        metadata = {"image_dir": paths, "z": torch.cat([latent_ctx, latent_goal], dim = 1).reshape(6, -1, encoder.embed_dim), "z_straight": h}
        _write_metadata(metadata, seq_dir)
        
        logger.INFO(f"[{idx}] loss_ar={loss_ar.item():.4f}")

def _write_metadata(metadata: np.ndarray, directory: str):
    np.save(os.path.join(directory, "data.npy"), metadata)

def _write_images(buffer: np.ndarray, directory: str):
    image_dir = os.path.join(directory, "images/")
    
    os.makedirs(image_dir, exist_ok=True)
    
    paths = []
    for idx, image in enumerate(buffer):
        img_path = os.path.join(image_dir, f"{idx:03d}.jpg")
        paths += ["/".join(img_path.split("/")[3:])]
        
        success = cv2.imwrite(img_path, image[..., ::-1].astype(np.uint8))
        if not success:
            print(f"Failed to write image to {img_path}")
    return paths


def _latent_loss(h, z, tokens_pframe, loss_exp=1.0):
    """Matches straightening/train.py latent_loss with no time_indicies (all timesteps)."""
    sub_h = h[:, tokens_pframe: z.size(1) + tokens_pframe]
    return torch.mean(torch.abs(z - sub_h) ** loss_exp) / loss_exp

def latent_loss(h: torch.Tensor, z: torch.Tensor, tokens_pframe, time_indicies: list[int] = None):
    sub_h = h[:, tokens_pframe: z.size(1) + tokens_pframe]
    T = sub_h.size(1) // tokens_pframe
    B, L, D = sub_h.shape

    sub_h_view = sub_h.view(B, T, tokens_pframe, D)
    z_view = z.view(B, T, tokens_pframe, D)

    device = h.device
    if time_indicies is None or len(time_indicies) == 0:
        idx_tensor = torch.arange(T, device=device)
    else:
        valid_idx = [i for i in time_indicies if i < T]
        if not valid_idx: # If provided list was out of bounds, default to all
            idx_tensor = torch.arange(T, device=device)
        else:
            idx_tensor = torch.tensor(valid_idx, device=device)
    
    selected_h = sub_h_view[:, idx_tensor].flatten(1, 2)
    selected_z = z_view[:, idx_tensor].flatten(1, 2)
    
    return torch.mean(torch.abs(selected_z - selected_h) ** 1.0) / 1.0