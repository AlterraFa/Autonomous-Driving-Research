import os, sys
import resource
import time
import gc
from functools import partial
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist

from .compile import (
    compile_model,
    compile_transform,
    compile_dataloader,
    compile_loss,
    format_targets,
    load_checkpoint,
)
from ..probe_notebook import recurse_weight_fd
from utils.training_logger import (
    get_next_run,
    create_supervised_logger,
    NoOpLogger
)

from utils.distributed import init_distributed
from utils.logger import Logger

logger = Logger(__name__)
GLOBAL_SEED = 12


ANALYSIS_STORAGE = {}
def analysis_name(name: str):
    def decorator(func):
        ANALYSIS_STORAGE[name] = func
        return func
    return decorator

def main(args: dict, path: str):
    # Optionally allow user to specify which modality's affine params to display via YAML (meta section)
    meta_cfg: dict = args.get("meta", {})
    affine_modality = meta_cfg.get("affine_modality", None)  # e.g., "velocity", "steer", "lat err"
    # Keep argument unpacking aligned with train.py so notebook PARAMS can be
    # forwarded directly without reshaping.
    train_cfg: dict = args.get("train", {})
    crop_size = train_cfg.get("crop_size", 224)
    nclips = train_cfg.get("nclips", 1)
    train_cfg['batch_size'] = 8

    loader_cfg: dict = args.get("loader_setup", {})
    num_workers = loader_cfg.get("num_workers", 1)
    persistent_workers = loader_cfg.get("persistent_workers", False)
    pin_mem = loader_cfg.get("pin_mem", False)

    model_cfg: dict = args.get("model", {})
    enc_cfg = model_cfg.get("enc", {})
    probe_cfg = model_cfg.get("probe", {})

    augment_cfg: dict = args.get("data_aug", {})
    auto_augment = augment_cfg.get("auto_augment", False)
    horizontal_flip = augment_cfg.get("horizontal_flip", False)
    motion_shift = augment_cfg.get("motion_shift", False)
    random_aspect_ratio = augment_cfg.get("random_resize_aspect_ratio", (1.0, 1.0))
    random_resize_scale = augment_cfg.get("random_resize_scale", (1.0, 1.0))
    reprob = augment_cfg.get("reprob", 0.0)
    normalize_targets = augment_cfg.get("normalize_targets", False)

    optim_cfg: dict = args.get("optimization", {})
    anneal = optim_cfg.get("annel", optim_cfg.get("anneal", 1))
    epochs = optim_cfg.get("epochs", 100)
    final_lr = optim_cfg.get("final_lr", 0.0)
    final_wd = optim_cfg.get("final_weight_decay", 0.0)
    ipe = optim_cfg.get("ipe", 100)
    lr = optim_cfg.get("lr", 1e-3)
    start_lr = optim_cfg.get("start_lr", 1e-3)
    warmup = optim_cfg.get("warmup", 10)
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    betas = optim_cfg.get("betas", (0.9, 0.999))
    eps = optim_cfg.get("eps", 1.0e-8)

    grad_optim_cfg: dict = optim_cfg.get("gradient_optimizer", {})
    grad_optim_name = grad_optim_cfg.get("type", "normal")
    grad_optim_params = grad_optim_cfg.get("params", {})

    loss_cfg: dict = args.get("loss", {})
    normalize_rep = loss_cfg.get("normalize_rep", False)

    meta_cfg: dict = args.get("meta", {})
    dtype = meta_cfg.get("dtype", "float32")
    seed = int(meta_cfg.get("seed", GLOBAL_SEED))
    save_freq = meta_cfg.get("save_every_freq", 2)
    save_root_dir_cfg = meta_cfg.get("save_root_dir", "./Experiment")
    sync_gc = meta_cfg.get("sync_gc", False)
    continue_from_path = meta_cfg.get("continue_from_path", None)
    continue_train = bool(continue_from_path)
    resume_prefer_best = bool(meta_cfg.get("resume_prefer_best", True))

    checkpoint_cfg: dict = args.get("checkpoint", {})
    patience = checkpoint_cfg.get("patience", epochs)
    min_delta = checkpoint_cfg.get("min_delta", 0.0)

    logging_cfg: dict = args.get("logging", {})
    progress_type = logging_cfg.get("progress_type", "table")
    save_csv = logging_cfg.get("save_csv", True)
    save_batch_csv = logging_cfg.get("save_batch_csv", False)
    save_epoch_csv = logging_cfg.get("save_epoch_csv", True)

    # Locate the directory that contains all available weight files for eval.
    weights_fd = recurse_weight_fd(path)
    if not weights_fd:
        raise FileNotFoundError(
            f"Could not resolve a unique weights folder from path: {path}"
        )
    weights_dir = os.path.abspath(weights_fd)
    logger.INFO(f"Resolved weights directory: {weights_dir}")

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
    
    
    torch.cuda.set_device(rank)
    device_type = f'cuda:{rank}'
    device = torch.device(device_type)

    # Keep model initialization parity with train.py.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    encoder, probe = compile_model(enc_cfg=enc_cfg, probe_cfg=probe_cfg, device=device)
    try:
        probe, _, _, resume_score, resume_meta = load_checkpoint(
            model=probe,
            optimizer=None,
            checkpoint_dir=weights_dir,
            checkpoint_name="probe.pt",
            prefer_best=False,
            map_location=device,
        )
        logger.CUSTOM("SUCCESS", f"Successfully loaded probe from: {weights_dir}")
        if resume_score is not None:
            logger.INFO(f"Checkpoint score: {float(resume_score):.6f}")
    except Exception as e:
        logger.ERROR("Unable to load weights", exit_code = -1, full_traceback = e)

    # Match validation/inference behavior: disable dropout and train-time layers.
    encoder.eval()
    probe.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    criterion = compile_loss(loss_cfg=loss_cfg, device=device)
    criterion_payload = resume_meta.get("criterion") if isinstance(resume_meta, dict) else None
    if isinstance(criterion_payload, dict):
        criterion.load_state_dict(criterion_payload, strict=True)
        logger.CUSTOM("SUCCESS", "Loaded criterion state from checkpoint metadata")
    else:
        logger.WARNING("Criterion state not found in checkpoint metadata. Using criterion from config only.")
    criterion.eval()

    # Get enabled tasks for mapping
    criterion_core = criterion
    if hasattr(criterion, "module"):
        criterion_core = criterion.module
    enabled_tasks = list(getattr(criterion_core, "enabled_tasks", []))
    # Print only the selected modality's affine params
    if affine_modality is not None:
        if affine_modality not in enabled_tasks:
            print(f"Requested modality '{affine_modality}' not in enabled tasks: {enabled_tasks}")
        else:
            idx = enabled_tasks.index(affine_modality)
            scale = getattr(probe, "scale", None)
            shift = getattr(probe, "shift", None)
            if scale is not None:
                raw_scale = scale.detach().cpu().tolist()
                eff_scale = F.softplus(scale.detach()).cpu().tolist()
                print(f"Affine Raw Scale {affine_modality}: {raw_scale[idx]}")
                print(f"Affine Scale {affine_modality} (softplus): {eff_scale[idx]}")
            if shift is not None:
                raw_shift = shift.detach().cpu().tolist()
                print(f"Affine Shift {affine_modality}: {raw_shift[idx]}")
    else:
        affine_param_names = [
            n for n, _ in probe.named_parameters()
            if "scale" in n or "shift" in n
        ]
        affine_state_keys = [
            k for k in probe.state_dict().keys()
            if "scale" in k or "shift" in k
        ]
        print(f"Affine parameter names: {affine_param_names}")
        print(f"Affine state_dict keys: {affine_state_keys}")
        for name, param in probe.named_parameters():
            if "scale" in name or "shift" in name:
                raw_values = param.detach().cpu().tolist()
                print(f"{name} raw values: {raw_values}")
                if "scale" in name:
                    effective_values = F.softplus(param.detach()).cpu().tolist()
                    print(f"{name} effective values (softplus): {effective_values}")



    transforms = compile_transform(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=random_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,    
    )
    train_loader, val_loader, train_sampler, val_sampler, _  = compile_dataloader(
        train_cfg=train_cfg,
        nclips=nclips,
        collate_fn=torch.utils.data.default_collate,
        num_workers=4,
        persistance_workers = persistent_workers,
        pin_memory = pin_mem,
        world_sz = world_size,
        rank = rank,
        normalize_targets = normalize_targets,
        transform = transforms,
    )

    eval_loader = val_loader if val_loader is not None and len(val_loader) > 0 else train_loader
    if eval_loader is val_loader:
        if hasattr(val_sampler, 'set_epoch'):
            val_sampler.set_epoch(0)
        logger.INFO("Using validation loader for loss evaluation")
    else:
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(0)
        logger.WARNING("Validation loader unavailable, using training loader for loss evaluation")

    loader = iter(eval_loader)

    ANALYSIS_STORAGE[probe_cfg['name']](
        probe=probe,
        encoder=encoder,
        criterion=criterion,
        loader=loader,
        transforms=transforms,
        device=device,
        dtype=dtype,
        normalize_rep=normalize_rep,
        mixed_precision=mixed_precision,
    )

@analysis_name("EfficientProbe")
def efficient(probe, encoder, criterion, loader, transforms, device, dtype, normalize_rep=False, mixed_precision=False, **kwargs):
    probe.pooler.efficient_probe.debug = False
    running_loss = 0.0
    n_batches = 0
    for sample in loader:
        
        def load_clips():
            clips = sample[0][0].to(device, non_blocking = True)
            actions = [{key: value.to(device, non_blocking=True) for key, value in action_dict.items()} for action_dict in sample[1]]
            return clips, actions

        clips, actions = load_clips()
        
        with torch.no_grad():
            with torch.amp.autocast(device.type, dtype = dtype, enabled = mixed_precision):
                h = encoder(clips)
                if normalize_rep:
                    h = F.layer_norm(h, (h.size(-1), ))
                output = probe(h)
                attn_map = probe.pooler.efficient_probe.attn_map
                
        enabled_tasks = getattr(criterion, "enabled_tasks", ("velocity", "steer"))
        actions = format_targets(output, actions, enabled_tasks)
        loss, detail = criterion(output, actions)
        running_loss += float(loss.item())
        n_batches += 1

        print(actions["velocity"][0])
        print(output[0, :, 0])
        print(f"batch_loss={float(detail['total_loss'].item()):.6f}")
        print()

    if n_batches > 0:
        print(f"mean_loss={running_loss / n_batches:.6f} over {n_batches} batches")


def _decode_probe_waypoints(decoder, a_latent, decoder_type, n_waypoints):
    pred = decoder(a_latent)
    if decoder_type != 'EfficientProbe':
        return pred

    if pred.ndim == 4:
        pred = pred[:, -1, :, :].mean(dim=1)
    elif pred.ndim == 3:
        pred = pred[:, -1, :]
    elif pred.ndim != 2:
        raise ValueError(f"Unexpected EfficientProbe output shape: {tuple(pred.shape)}")

    expected_dim = n_waypoints * 2
    if pred.shape[-1] != expected_dim:
        raise ValueError(
            f"EfficientProbe output dim mismatch: got {pred.shape[-1]}, expected {expected_dim}"
        )

    return pred.view(pred.shape[0], n_waypoints, 2)
                