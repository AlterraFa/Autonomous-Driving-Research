import os
import sys
import yaml
import time
import gc
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import optim

from model.ImprovedVENL.impl.model import ImprovedVENL
from model.ImprovedVENL.utils.schedulers import CosineSchedule, CosineWDSchedule
from model.training_logger import TrainingLogger, get_next_run
from model.early_stop import EarlyStopping
from model.ImprovedVENL.data_utils.data_loader import CarlaDatasetLoader
from model.ImprovedVENL.utils.loss import NavLoss
from model.ImprovedVENL.data_utils.image_transform import (
    Augment, 
    Normalization
)
from utils.messages.logger import Logger

logger = Logger()

FOLDER_DIR = os.path.dirname(os.path.abspath(__file__))


def get_vram_usage() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / (1024 ** 3), 3)
    return 0.0


def load_checkpoint(
    model, 
    optimizer, 
    checkpoint_dir, 
    prefer_best=True, 
    map_location=None
):
    """Load model checkpoint and optimizer state."""
    from glob import glob
    
    basename = "checkpoint.pt"
    meta_path = os.path.join(checkpoint_dir, basename)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location)
    score = meta.get("score")
    start_epoch = meta.get("epoch", 0)

    prefix = "best_" if prefer_best else "last_"
    extname = ".pt"
    model_path = glob(os.path.join(checkpoint_dir, f"{prefix}*{extname}"))[0]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}")

    loaded_model = torch.load(model_path, map_location=map_location)
    model.load_state_dict(loaded_model)
    
    if optimizer is not None and meta.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])

    return model, optimizer, start_epoch + 1, score


def main(args, yaml_path):


    
    # ================================================= #
    #              DATA CONFIGURATION
    # ================================================= #
    train_cfg        = args.get('train', {})
    train_datasets   = train_cfg.get('datasets', [])
    train_batch_size = train_cfg.get('batch_size', 32)
    train_shuffle    = train_cfg.get('shuffle', True)
    
    val_cfg          = args.get('val', {})
    val_datasets     = val_cfg.get('datasets', [])
    val_batch_size   = val_cfg.get('batch_size', 32)
    val_split_ratio  = val_cfg.get('split_ratio', 0.2)
    
    # ================================================= #
    #              LOADER CONFIGURATION
    # ================================================= #
    loader_cfg         = args.get('loader_setup', {})
    num_workers        = loader_cfg.get('num_workers', 4)
    persistent_workers = loader_cfg.get('persistent_workers', True)
    pin_mem            = loader_cfg.get('pin_mem', True)
    ram_caching        = loader_cfg.get("ram_caching", False)
    
    # ================================================= #
    #              AUGMENTATION CONFIGURATION
    # ================================================= #
    aug_cfg = args.get('data_aug', {})
    color_jitter     = aug_cfg.get('color_jitter', 1.0)
    color_distortion = aug_cfg.get('color_distortion', False)
    gaussian_blur    = aug_cfg.get('gaussian_blur', False)
    normalization    = aug_cfg.get('normalization', ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    crop_size        = aug_cfg.get("crop", [0.0, 1.0, 0.0, 1.0])
    
    # ================================================= #
    #              META CONFIGURATION
    # ================================================= #
    meta_cfg   = args.get('meta', {})
    dtype      = meta_cfg.get('dtype', 'float32')
    seed       = meta_cfg.get('seed', 42)
    use_sdpa   = meta_cfg.get('use_sdpa', True)
    do_compile = meta_cfg.get('do_compile', False)
    
    # ================================================= #
    #              MODEL CONFIGURATION
    # ================================================= #
    model_cfg         = args['model']
    input_metadata    = model_cfg.get('input_metadata', {})
    patch_sizes       = model_cfg.get('patch_sizes', [16, 16, 16])
    output_names      = model_cfg.get('output_names', ["waypoint", "weights", "muy", "sigma"])
    components        = model_cfg.get('components', 6)
    num_waypoints     = model_cfg.get('num_waypoints', 6)
    depth             = model_cfg.get('depth', 5)
    embed_dim         = model_cfg.get('embed_dim', 512)
    num_heads         = model_cfg.get('num_heads', 8)
    mlp_ratio         = model_cfg.get('mlp_ratio', 4)
    qkv_bias          = model_cfg.get('qkv_bias', True)
    qk_scale          = model_cfg.get('qk_scale', None)
    act_layer         = model_cfg.get("act_layer", "nn.GELU")
    use_gru           = model_cfg.get('use_gru', False) 
    drop              = model_cfg.get('drop', 0.0)
    attn_drop         = model_cfg.get('attn_drop', 0.0)
    drop_path         = model_cfg.get('drop_path', 0.0)
    drop_route        = model_cfg.get('drop_route', 0.0)
    drop_all          = model_cfg.get('drop_all', 0.0)
    use_rope          = model_cfg.get('use_rope', True)
    use_cls           = model_cfg.get('use_cls', True)
    use_gradient_ckpt = model_cfg.get('use_gradient_ckpt', True)
    
    # ================================================= #
    #              OPTIMIZATION CONFIGURATION
    # ================================================= #
    opt_cfg   = args.get('optim', {})
    epochs    = opt_cfg.get('epochs', 300)
    init_lr   = opt_cfg.get('initLR', 3.0e-4)
    target_lr = opt_cfg.get('targetLR', 1.0e-7)
    init_wd   = opt_cfg.get('initWD', 0.0)
    target_wd = opt_cfg.get('targetWD', 0.0)
    betas     = opt_cfg.get("betas", [0.9, 0.999])
    
    
    # ================================================= #
    #              LOSS CONFIGURATION
    # ================================================= #
    loss_contrib = args.get('loss_contrib', {})
    loss_args    = args.get('loss_args', {})
    loss_reg     = args.get('loss_reg', {})

    huber_delta  = loss_args.get('huber_delta', 1.0)
    pad_value    = loss_args.get('pad_value', 0.0)
    target_std   = loss_args.get('target_std', [0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    target_sep   = loss_args.get('target_sep', [0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    wp_coeff     = loss_args.get('wp_coeff', [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    
    
    # ================================================= #
    #              CHECKPOINT CONFIGURATION
    # ================================================= #
    checkpoint_cfg    = args.get('checkpoints', {})
    save_best_only    = checkpoint_cfg.get('save_best_only', True)
    mode              = checkpoint_cfg.get('mode', 'min')
    save_weights_only = checkpoint_cfg.get('save_weights_only', True)
    frequency         = checkpoint_cfg.get('frequency', 10)
    patience          = checkpoint_cfg.get('patience', 20)
    min_delta         = checkpoint_cfg.get('min_delta', 1.0e-4)
    load_model        = checkpoint_cfg.get('load_model', False)
    
    # ================================================= #
    #              SETUP
    # ================================================= #
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    if dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
    elif dtype.lower() == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    logger.INFO(f"Using device: {device_str}, dtype: {dtype}")
    
    # ================================================= #
    #              MODEL INITIALIZATION
    # ================================================= #
    model = ImprovedVENL(
        input_metadata=input_metadata,
        patch_sizes=patch_sizes,
        output_names=output_names,
        components=components,
        num_waypoints=num_waypoints,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop,
        act_layer=act_layer,
        attn_drop=attn_drop,
        use_gru=use_gru,
        drop_path=drop_path,
        drop_route=drop_route,
        drop_all=drop_all,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_cls=use_cls,
        use_gradient_ckpt=use_gradient_ckpt,
        init_std=0.05   
    )
    
    
    # ================================================= #
    #              DATASET INIT
    # ================================================= #
    train_path = [meta['path'] for meta in train_cfg['datasets']]
    train = CarlaDatasetLoader(
        dataset_dir = train_path,
        fraction = 1.0,
        ram_caching = ram_caching,
        shuffle = True,
        pad_value = pad_value
    )
    collate_fn = train.collate_fn
    val_path  = [meta['path'] for meta in val_cfg['datasets']]
    if val_path:
        val  = CarlaDatasetLoader(
            dataset_dir = val_path,
            fraction = 1.0,
            ram_caching = ram_caching,
            shuffle = False,
            pad_value = pad_value
        )
    else:
        train, val, _ = train.split(1-val_split_ratio, val_split_ratio)
        
        
    train_loader = DataLoader(
        train, 
        batch_size = train_batch_size,
        shuffle = train_shuffle,
        num_workers = num_workers,
        pin_memory = pin_mem,
        persistent_workers = persistent_workers,
        collate_fn = collate_fn
    )
    val_loader = DataLoader(
        val, 
        batch_size = val_batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_mem,
        persistent_workers = persistent_workers,
        collate_fn = collate_fn
    )
    
    loss_compute = NavLoss(
        model,
        delta = huber_delta,
        wp_coeff = wp_coeff,
        loss_coeffs = loss_contrib,
        target_sep = target_sep,
        target_std = target_std,
        device = device,
        pad_value = pad_value,
        enabled_losses = ["soft_dtw", "nll", "std_reg", "l1_model", "l2_model"]
    )

    # ================================================= #
    #                AUGMENTATION INIT
    # ================================================= #
    augment_transform = Augment(
        dimension = input_metadata['I0'][2:],
        crop = crop_size,
        color_jitter = color_jitter,
        color_distortion = color_distortion,
        gaussian_blur = gaussian_blur,
        normalization = normalization
    )
    image_normal_transform  = Normalization(
        size = input_metadata['I0'][2:],
        crop = crop_size,
        normalization = normalization
    )
    map_normal_transform  = Normalization(
        size = input_metadata['MU'][2:]
    )

    optimizer = optim.AdamW(model.parameters(), lr = init_lr, betas = betas, weight_decay = init_wd)
    lr_scheduler = CosineSchedule(
        optimizer, 
        ref_lr = init_lr, 
        T_max = int(epochs * len(train_loader)), 
        final_lr = target_lr
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, 
        ref_wd = init_wd, 
        T_max = int(epochs * len(train_loader)), 
        final_wd = target_wd
    )

    earlystop = EarlyStopping(
        patience, 
        min_delta, 
        freq = frequency, 
        path = f"{log_dir}/run{run}/weights/{model._get_name()}.pt", 
        mode = mode, 
        verbose = False,
        weights_only = True
    )
    
    if argument.cont_train:
        logger.INFO(f"Continuing training from run {run-1}")
        model, optimizer, current_epochs, score = load_checkpoint(model, optimizer, os.path.join(os.path.dirname(yaml_path), "weights"))
        for _ in range(current_epochs):
            for _ in range(len(train_loader)):
                lr_scheduler.step()
                wd_scheduler.step()
        earlystop.best_loss = score

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to("cuda")
    
    
    model = model.to(device)
    logger.INFO(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    if do_compile:
        logger.INFO("Compiling model")
        model.compile()
        logger.CUSTOM("SUCCESS", "Model compiling was a success")
    log_stats = TrainingLogger(
        log_dir = log_dir,
        epochs  = epochs,
        run_name = f"run{run}",
        progress_type = "table"
    ) 
    log_stats.log_model_graph(
        model, 
        input_sample = [torch.zeros(shape, device = device) for shape in input_metadata.values()]
    )
    os.system(f"cp {yaml_path} {os.path.join(log_dir, f'run{run}')}")
    
    def to_device(img_batch, controls_batch):
        img_batch      = {name: img_batch[name].to(device) for name in img_batch.keys()}
        controls_batch = {name: controls_batch[name].to(device) for name in controls_batch.keys()}
        return img_batch, controls_batch

    def apply_transform(img_batch, augment=True):
        new_batch = {}
        for key, value in img_batch.items():
            if key == "I0": 
                new_batch[key] = augment_transform(value) if augment else image_normal_transform(value)
            else: 
                new_batch[key] = map_normal_transform(value)
        return new_batch
        
    # ================================================= #
    #                TRAINING PHASE
    # ================================================= #
    with log_stats:
        log_stats.start_training("Finetune JEPANav")
        for epoch_idx in range(epochs):
            # =================== Training ==================== #

            log_stats.start_epoch(epoch_idx, len(train_loader), desc = "Training")
            model.train()
            for img_batch, controls_batch in log_stats.batch_iterator(train_loader):
                optimizer.zero_grad()
                
                img_batch, controls_batch = to_device(img_batch, controls_batch)
                img_batch                 = apply_transform(img_batch, augment=True)
                
                with torch.amp.autocast("cuda", dtype = dtype):
                    pred   = model(**img_batch)                    
                    gt_wp  = controls_batch['midlane_wp']
                    aux_wp = controls_batch['aux_wp']
                    loss, metrics = loss_compute(pred, gt_wp, aux_wp)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                wd_scheduler.step()
                loss_val = loss.item()
                
                log_stats.log_batch({
                    "Loss": loss_val,
                    **metrics
                }, phase = "train")
                
            log_stats.start_phase(len(val_loader), desc = "Validating")
            model.eval()
            # =================== Validating ==================== #
            for img_batch, controls_batch in log_stats.batch_iterator(val_loader):
                
                img_batch, controls_batch = to_device(img_batch, controls_batch)
                img_batch                 = apply_transform(img_batch, augment=False)
                
                with torch.amp.autocast("cuda", dtype = dtype):
                    with torch.no_grad():
                        pred   = model(**img_batch)                    
                        gt_wp  = controls_batch['midlane_wp']
                        aux_wp = controls_batch['aux_wp']
                        loss, metrics = loss_compute(pred, gt_wp, aux_wp)
                    

                loss_val = loss.item()
                log_stats.log_batch({
                    "Loss": loss_val,
                    **metrics
                }, phase = "val")

            
            earlystop(log_stats.get_metric("Loss", "val"), model, epoch_idx, optimizer)
            if earlystop.early_stop:
                break

            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']
            log_stats.log_epoch(
                extra_metrics = {
                    "LR": current_lr,
                    "WD": current_wd
                }
            )

if __name__ == "__main__":
    log_dir = f"{FOLDER_DIR}/Experiment/"
    run = get_next_run(log_dir)

    parser = ArgumentParser()
    parser.add_argument("--cont_train", action = "store_true")
    argument = parser.parse_args()
    
    if argument.cont_train:
        yaml_path = os.path.join(log_dir, f'run{run-1}/model_cfg.yaml')
    else:
        yaml_path = os.path.join(FOLDER_DIR, "configs/model_cfg.yaml")

    
    if not os.path.exists(yaml_path):
        logger.ERROR(f"Config file not found: {yaml_path}")
        sys.exit(1)
    
    with open(yaml_path, "r") as f:
        args = yaml.safe_load(f)
    
    main(args, yaml_path)
