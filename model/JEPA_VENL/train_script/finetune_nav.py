import os, sys
import yaml
import torch
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# -- Append the root folder
script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from model.JEPA_VENL.impl.transformer import ViTEncode
from model.JEPA_VENL.impl.venl import SingleVENL
from model.JEPA_VENL.impl.jepa_nav import UnifiedJEPANav
from model.JEPA_VENL.data_utils.jepa_nav.compose import (
    optim_schedulers_composer,
    dataloader_composer,
)
from model.JEPA_VENL.data_utils.jepa_nav.image_transform import Augment, Normalization
from model.training_logger import (
    TrainingLogger, 
    get_next_run
)
from model.JEPA_VENL.utils.loss import NavLoss
from utils.others.helper import EarlyStopping

def get_vram_usage() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / (1024 ** 3), 3)
    return 0.0

torch.manual_seed(45)

def main(args, yaml_path):
    
    # -- Encoder args (excluding architecture)
    model_cfg = args['model']
    map_shape       = model_cfg['map_shape']
    num_waypoints   = model_cfg["num_waypoints"]
    components      = model_cfg["components"]
    droprate        = model_cfg["drop_rate"]
    map_droprate    = model_cfg["map_droprate"]
    enc_weight_path = model_cfg["enc_weight_path"]
    
    
    # -- Data args
    data_cfg = args['data']
    dataset_path  = data_cfg['dataset_path']
    ram_caching   = data_cfg['ram_caching']
    batch_size    = data_cfg['batch_size']
    num_workers   = data_cfg['num_workers']
    train_split   = data_cfg['train_split']
    val_split     = data_cfg['val_split']
    shuffle       = data_cfg['shuffle']
    pad_value     = data_cfg['pad_value']
    augmentations = data_cfg['augmentations']
    
    # -- Training args
    train_cfg = args['train_param']
    full_finetune = train_cfg['full_finetune']
    init_lr       = train_cfg['init_lr']
    final_lr      = train_cfg['final_lr']
    weight_decay  = train_cfg['weight_decay']
    epochs        = train_cfg['epochs']
    betas         = train_cfg['betas']
    plateau_point = train_cfg['plateau_point']
    patience      = train_cfg['early_stopping']['patience']
    min_delta     = train_cfg['early_stopping']['delta']
    loss_contrib  = train_cfg['loss_contrib']
    loss_args     = train_cfg['loss_args']

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ================================================= #
    #               INITIALIZE MODELS
    # ================================================= #
    encoder: ViTEncode          = torch.load(
        os.path.join(script_dir, enc_weight_path), 
        weights_only = False, 
        map_location = device
    )
    readout: SingleVENL         = SingleVENL.waypoint(
        map_shape     = map_shape,
        num_waypoints = num_waypoints, 
        components    = components,
        droprate      = droprate,
        map_droprate  = map_droprate
    )
    unified_nav: UnifiedJEPANav = UnifiedJEPANav(encoder, readout, full_finetune).to(device)
    
    # -- Initialize all lazy layers 
    dummy_inp = torch.randn((1, 3, *augmentations['image_size'])).to(device)
    dummy_MU  = torch.randn((1, 1, *map_shape)).to(device)
    dummy_MR  = torch.randn((1, 3, *map_shape)).to(device)
    unified_nav.initialize_module(dummy_inp, dummy_MU, dummy_MR)
    
    # ================================================= #
    #               DATASET LOADING
    # ================================================= #
    augment_transform = Augment(
        dimension     = augmentations['image_size'],
        crop          = augmentations['crop'],
        color_jitter  = augmentations['color_jitter'],
        gaussian_blur = augmentations["gaussian_blur"],
        normalization = augmentations["normalization"]
    )
    normalization_transform = Normalization(size = map_shape)
    normal_transform  = Augment(
        dimension     = augmentations['image_size'],
        crop          = augmentations['crop'],
        normalization = augmentations["normalization"]
    )
    
    train_loader, val_loader, test_loader = dataloader_composer(
        root        = os.path.join(FOLDER_DIR, dataset_path),
        ram_caching = ram_caching,
        split       = [train_split, val_split],
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pad_value   = pad_value
    )
    
    # ================================================= #
    #       TRAINING OPTIMIZER, SCHEDULER AND LOSS
    # ================================================= #
    optimizer, scheduler = optim_schedulers_composer(
        nav_model     = unified_nav,
        full_finetune = full_finetune,
        epochs        = epochs,
        plateau_point = float(plateau_point),
        init_lr       = init_lr,
        final_lr      = final_lr,
        weight_decay  = weight_decay,
        betas         = betas
    )
    loss_compute = NavLoss(
        nav_model     = unified_nav,
        delta         = loss_args['huber_delta'],
        target_sep    = loss_args['target_sep'],
        target_std    = loss_args['target_std'],
        loss_coeffs   = loss_contrib,
        full_finetune = full_finetune,
        pad_value     = pad_value,
        device = device
    )
    
    # ================================================= #
    #           OTHER THINGS FOR MONITORING
    # ================================================= #
    log_dir = os.path.join(FOLDER_DIR, "../Experiment/finetune/")
    run_idx = get_next_run(log_dir)
    log_stats = TrainingLogger(
        log_dir = log_dir,
        epochs  = epochs,
        run_name = f"run{run_idx}",
        progress_type = 'table'
    ) 
    log_stats.log_model_graph(unified_nav, {"I0": dummy_inp, "MU": dummy_MU, "MR": dummy_MR})
    finetune_stop = EarlyStopping(patience = patience, min_delta = min_delta, freq = 10, path = os.path.join(log_dir, f"run{run_idx}/weights/{unified_nav._get_name()}.pt"))
    os.system(f"cp {yaml_path} {os.path.join(log_dir, f'run{run_idx}')}")
    
    # -- Helper functions 
    def to_device(img_batch, controls_batch):
        img_batch      = {name: img_batch[name].to(device) for name in img_batch.keys()}
        controls_batch = {name: controls_batch[name].to(device) for name in controls_batch.keys()}
        return img_batch, controls_batch
    
    def apply_transform(img_batch, augment=True):
        new_batch = {}
        for key, value in img_batch.items():
            if key == "I0": 
                new_batch[key] = augment_transform(value) if augment else normal_transform(value)
            else: 
                new_batch[key] = normalization_transform(value)
        return new_batch
    
    def grad_mean_abs(module):
        grads = [p.grad.abs().mean().item() for p in module.parameters() if p.grad is not None]
        return float(np.mean(grads)) if grads else 0.0
    
    # ================================================= #
    #                TRAINING PHASE
    # ================================================= #
    with log_stats:
        log_stats.start_training("Finetune JEPANav")
        for epoch_idx in range(epochs):
            # =================== Training ==================== #

            log_stats.start_epoch(epoch_idx, len(train_loader), desc = "Training")
            unified_nav.train()
            for img_batch, controls_batch in log_stats.batch_iterator(train_loader):
                optimizer.zero_grad()
                
                img_batch, controls_batch = to_device(img_batch, controls_batch)
                img_batch                 = apply_transform(img_batch, augment=True)
                
                with torch.amp.autocast("cuda", dtype = torch.bfloat16):
                    pred   = unified_nav(**img_batch)                    
                    gt_wp  = controls_batch['midlane_wp']
                    aux_wp = controls_batch['aux_wp']
                    loss, metrics = loss_compute(pred, gt_wp, aux_wp)

                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                
                log_stats.log_batch({
                    "Loss": loss_val,
                    **metrics
                }, phase = "train")
                
            log_stats.start_phase(len(val_loader), desc = "Validating")
            unified_nav.eval()
            # =================== Validating ==================== #
            for img_batch, controls_batch in log_stats.batch_iterator(val_loader):
                
                img_batch, controls_batch = to_device(img_batch, controls_batch)
                img_batch                 = apply_transform(img_batch, augment=False)
                
                with torch.amp.autocast("cuda", dtype = torch.bfloat16):
                    with torch.no_grad():
                        pred   = unified_nav(**img_batch)                    
                        gt_wp  = controls_batch['midlane_wp']
                        aux_wp = controls_batch['aux_wp']
                        loss, metrics = loss_compute(pred, gt_wp, aux_wp)
                    

                loss_val = loss.item()
                log_stats.log_batch({
                    "Loss": loss_val,
                    **metrics
                }, phase = "val")

            # -- Monitor the gradients of these as well
            grad_emb      = grad_mean_abs(unified_nav.readout.emb_pooling)
            grad_unrouted = grad_mean_abs(unified_nav.readout.unrouted_backbone)
            grad_routed   = grad_mean_abs(unified_nav.readout.routed_backbone)
            
            finetune_stop(log_stats.get_metric("Loss", "val"), unified_nav, epoch_idx, optimizer)
            if finetune_stop.early_stop:
                break

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            log_stats.log_epoch(
                extra_metrics = {
                    "LR": current_lr,
                    "Emb Grad": grad_emb,
                    "Unrouted Grad": grad_unrouted,
                    "Routed Grad": grad_routed, 
                }
            )
    
FOLDER_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    YAML_PATH   = os.path.join(FOLDER_DIR, "../configs/jepa_nav/config.yaml")
    with open(YAML_PATH, "r") as f:
        args = yaml.safe_load(f)
    main(args, YAML_PATH)