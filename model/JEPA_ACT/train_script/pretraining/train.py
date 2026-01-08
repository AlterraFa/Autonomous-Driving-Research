import os, sys
import resource
import yaml
import copy
import time
import gc

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import torch
import torch.nn.functional as F

from model.JEPA_ACT.masks.multiseq_multiblock3d import MaskCollator
from model.JEPA_ACT.train_script.pretraining.compile.models import compile_model
from model.JEPA_ACT.train_script.pretraining.compile.dataloader import compile_dataloader
from model.JEPA_ACT.train_script.pretraining.compile.optim import compile_optim
from model.JEPA_ACT.augmenter.transforms_builder import VideoTransform
from utils.messages.logger import Logger

from model.JEPA_ACT.masks.utils import apply_masks
from model.training_logger import (
    TrainingLogger, 
    get_next_run
)
from model.early_stop import EarlyStopping
from torch.nn import functional as F

def get_vram_usage() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / (1024 ** 3), 3)
    return 0.0

logger = Logger()

FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))


def load_checkpoint(model, optimizer, checkpoint_dir, prefer_best=True, map_location=None):
    basename = "checkpoint.pt"
    meta_path = os.path.join(checkpoint_dir, basename)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location)
    score = meta.get("score")
    start_epoch = meta.get("epoch", 0)

    prefix = "best_" if prefer_best else "last_"
    model_path = os.path.join(checkpoint_dir, f"{prefix}{basename}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}")

    loaded_model = torch.load(model_path, map_location=map_location)
    model.load_state_dict(loaded_model.state_dict())
    if optimizer is not None and meta.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])

    return model, optimizer, start_epoch + 1, score

def load_multi_checkpoint(
    models: dict,            # {"encoder": enc_model, "target": tgt_model, "predictor": pred_model}
    checkpoint_dir: str,
    prefer_best: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    map_location=None,
):
    meta_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location)
    score = meta.get("score")
    start_epoch = meta.get("epoch", 0)
    prefix = "best_" if prefer_best else "last_"

    def _load_model_file(model_key: str, filename_suffix: str):
        model_path = os.path.join(checkpoint_dir, f"{prefix}checkpoint{filename_suffix}.pt")
        if not os.path.exists(model_path):
            return  # allow missing optional models
        loaded = torch.load(model_path, map_location=map_location)
        models[model_key].load_state_dict(loaded.state_dict())

    _load_model_file("encoder", "")

    if "target" in models:
        _load_model_file("target", "_target")
    if "predictor" in models:
        _load_model_file("predictor", "_predictor")

    if optimizer is not None and meta.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])

    return models, optimizer, start_epoch + 1, score

def main(args, yaml_path):
    
    # -- Datasets arguments
    train_cfg  = args.get('train', {}) 
    val_cfg    = args.get('val', {})
    train_fpcs = [ds['fpcs'] for ds in args['train']['datasets']]
    val_fpcs = [ds['fpcs'] for ds in args['val']['datasets']]
    dataset_fpcs = set(train_fpcs + val_fpcs)
    if len(dataset_fpcs) == 1:
        ...
    
    # -- Common arguments
    common_cfg = args.get('common', {})
    patch_size = common_cfg.get("patch_size", 16)
    tubelet_size = common_cfg.get("tubelet_size", 2)
    crop_size    = common_cfg.get("crop_size", 224)
    nclips       = common_cfg.get("nclips", 1)

    # -- Loader hardware arguments
    loader_cfg = args.get("loader_setup", {})
    num_workers = loader_cfg.get("num_workers", 1)
    persistent_workers = loader_cfg.get("persistent_workers", False)
    pin_mem            = loader_cfg.get("pin_mem", False)

    

    # -- Data augmentation arguments    
    aug_cfg = args.get('data_aug', {})
    auto_aug                   = aug_cfg.get('auto_augment', None)
    motion_shift               = aug_cfg.get('motion_shift', False)
    random_resize_aspect_ratio = aug_cfg.get('random_resize_aspect_ratio', (0.75, 1.33))
    random_resize_scale        = aug_cfg.get('random_resize_scale', (0.08, 1.0))
    reprob                     = aug_cfg.get('reprob', 0.0)

    # -- Masks arguments
    masks_cfg = args.get('mask', {})
    
    # -- Meta arguments
    meta_cfg = args.get('meta', {})
    dtype           = meta_cfg.get('dtype', 'float32')
    save_every_freq = meta_cfg.get('save_every_freq', 1)
    seed            = meta_cfg.get('seed', 42)
    use_sdpa        = meta_cfg.get('use_sdpa', True)

    # -- Model arguments
    model_cfg = args['model']
    # --- Encoder Parameters ---
    enc_dim          = model_cfg.get('enc_dim', 512)
    enc_head         = model_cfg.get('enc_head', 8)
    enc_depth        = model_cfg.get('enc_depth', 12)
    enc_dropout      = model_cfg.get('enc_dropout', 0.1)
    enc_attn_dropout = model_cfg.get('enc_attn_dropout', 0.1)
    enc_droppath     = model_cfg.get('enc_droppath', 0.0)
    # --- Predictor Parameters ---
    pred_depth        = model_cfg.get('pred_depth', 12)
    pred_dim          = model_cfg.get('pred_embed_dim', 384)
    pred_head         = model_cfg.get('pred_num_heads', 12)
    pred_dropout      = model_cfg.get('pred_dropout', 0.1)
    pred_attn_dropout = model_cfg.get('pred_attn_dropout', 0.1)
    pred_droppath     = model_cfg.get('pred_droppath', 0.0)
    # --- Feature Toggles ---
    uniform_power     = model_cfg.get('uniform_power', True)
    use_act_ckpt      = model_cfg.get('use_activation_checkpointing', True)
    use_mask_tokens   = model_cfg.get('use_mask_tokens', True)
    use_rope          = model_cfg.get('use_rope', False)
    use_silu          = model_cfg.get('use_silu', False)
    zero_init_mask    = model_cfg.get('zero_init_mask_tokens', True)
    do_compile        = model_cfg.get('compile_model', False)

    # -- Optimizer configuration 
    opt_cfg = args.get('optimization', {})
    ema                = opt_cfg.get('ema', [0.999, 1.0])
    epochs             = opt_cfg.get('epochs', 10)
    start_lr           = opt_cfg.get('start_lr', 0.0001)
    final_lr           = opt_cfg.get('final_lr', 1.0e-6)
    betas              = opt_cfg.get('betas', [0.9, 0.999])
    init_wd            = opt_cfg.get('weight_decay', 0.0)
    final_wd           = opt_cfg.get('final_weight_decay', 0.0)
    ipe                = opt_cfg.get('ipe', 100)
    
    # -- Loss configuration
    loss_exp = args.get("loss", {}).get("loss_exp", 1.0)

    # -- Checkpoint configuration
    checkpoint_cfg = args.get('checkpoint', {})
    patience = checkpoint_cfg.get('patience', 10)
    min_delta = checkpoint_cfg.get('min_delta', 0.001)
    load_model = checkpoint_cfg.get('load_model', False)
    

    

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

    encoder, predictor = compile_model(
        img_size = (crop_size, crop_size),
        patch_size = patch_size,
        fpc = max(dataset_fpcs),
        tubelet_size = tubelet_size,
        enc_embed_dim = enc_dim,
        enc_depth = enc_depth,
        enc_num_heads = enc_head,
        enc_drop_rate = enc_dropout,
        enc_attn_drop_rate = enc_attn_dropout,
        enc_drop_path_rate = enc_droppath,
        pred_embed_dim = pred_dim,
        pred_depth = pred_depth,
        pred_num_heads = pred_head,
        pred_drop_rate = pred_dropout,
        pred_attn_drop_rate = pred_attn_dropout,
        pred_drop_path_rate = pred_droppath,
        use_silu = use_silu,
        use_rope = use_rope,
        use_activation_checkpointing = use_act_ckpt,
        num_unique_fpcs = 1
    )
    target = copy.deepcopy(encoder)
    
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    target = target.to(device)

    if do_compile:
        encoder   = torch.compile(encoder, mode="reduce-overhead", fullgraph=False)
        predictor = torch.compile(predictor, mode="reduce-overhead", fullgraph=False)
        target    = torch.compile(target, mode="reduce-overhead", fullgraph=False)
        
    collator = MaskCollator(
        cfgs_mask = masks_cfg,
        dataset_fpcs = dataset_fpcs,
        crop_size = (crop_size, crop_size),
        patch_size = (patch_size, patch_size),
        tubelet_size = tubelet_size
    )

    transform = VideoTransform(
        random_resize_scale = random_resize_scale,
        random_resize_aspect_ratio = random_resize_aspect_ratio,
        motion_shift = motion_shift,
        auto_augment = auto_aug,
        reprob = reprob
    )
    
    train_loader, val_loader = compile_dataloader(
        train_cfg = train_cfg,
        val_cfg = val_cfg,
        nclips = nclips,
        transform = transform,
        collate_fn = collator,
        num_workers = num_workers,
        persistance_workers = persistent_workers,
        pin_memory = pin_mem
    )
    dataset_ratio = len(train_loader) // len(val_loader)
    
    optim, scheduler, wd_scheduler = compile_optim(
        encoder = encoder,
        predictor = predictor,
        betas = betas,
        init_lr = start_lr,
        final_lr = final_lr,
        init_wd = init_wd,
        final_wd = final_wd,
        epochs = epochs,
        ipe = ipe,
    )

    # ================================================= #
    #           OTHER THINGS FOR MONITORING
    # ================================================= #
    log_dir = os.path.join(FOLDER_DIR, "../Experiment/pretraining/")
    run_idx = get_next_run(log_dir)
    log_stats = TrainingLogger(
        log_dir = log_dir,
        epochs  = epochs,
        run_name = f"run{run_idx}",
        progress_type = "table"
    ) 
    
    if load_model:
        logger.WARNING(f"Continuing training from run {run_idx-1}. Disable load_model if you intent on retraining")
        models = {
            "encoder": encoder,
            "target": target,          # include only if you have it
            "predictor": predictor,
        }
        models, optim, start_epoch, last_score = load_multi_checkpoint(
            {
                "encoder": encoder,
                "target": target,
                "predictor": predictor
            },
            checkpoint_dir = log_dir + f"run{run_idx-1}/weights",
            prefer_best = True,
            optimizer = optim,
            map_location = 'cuda'
        )    
        encoder = models['encoder']
        predictor = models['predictor']
        target = models['target']
        
        for _ in range(start_epoch * ipe):
            wd_scheduler.step()
            scheduler.step()
    
    
    enc_stop    = EarlyStopping(patience = patience, freq = save_every_freq, min_delta = min_delta, path = os.path.join(log_dir, f"run{run_idx}/weights/encoder.pt"))
    pred_stop   = EarlyStopping(patience = patience, freq = save_every_freq, min_delta = min_delta, path = os.path.join(log_dir, f"run{run_idx}/weights/predictor.pt"))
    target_stop = EarlyStopping(patience = patience, freq = save_every_freq, min_delta = min_delta, path = os.path.join(log_dir, f"run{run_idx}/weights/target.pt"))
    os.system(f"cp {yaml_path} {os.path.join(log_dir, f'run{run_idx}')}")

    
    
    # ================================================= #
    #                TRAINING FUNCTIONS
    # ================================================= #
    def load_clips(sample):
        all_clips, masks_encs, masks_preds = [], [], []
        
        for clip_meta, masks_enc, masks_pred in sample:
            all_clips   += [clip_meta[0][0].to(device)]
            masks_encs  += [[m.to(device, non_blocking = True) for m in masks_enc]]
            masks_preds += [[m.to(device, non_blocking = True) for m in masks_pred]]
        return all_clips, masks_encs, masks_preds

    def forward_pred(clips, enc_masks, pred_masks):
        z = encoder(clips, enc_masks)
        z = predictor(z, enc_masks, pred_masks)
        return z
    
    def forward_target(clips):
        """Returns a list of clips' embeddings"""
        with torch.no_grad():
            h = target(clips)
            h = [F.layer_norm(embed, (embed.size(-1), )) for embed in h]
            return h
    
    def loss_fn(z, h, preds_masks):
        h = [apply_masks(hi, mi, concat = False) for hi, mi in zip(h, preds_masks)]

        loss, n = 0, 0
        all_h = []
        for hi, zi in zip(h, z):
            for hi_cfg, zi_cfg in zip(hi, zi): 
                # -- Generalized power loss
                loss += torch.mean(torch.abs(hi_cfg - zi_cfg) ** loss_exp) / loss_exp
                all_h += [hi_cfg.reshape(-1, hi_cfg.size(-1))]
                n += 1
        loss /= n
        with torch.no_grad():
            all_h = torch.cat(all_h, dim=0)
            std_per_feature = torch.sqrt(all_h.var(dim=0) + 1e-04)
            embed_std = std_per_feature.mean()
        return loss, embed_std
    
    def ema_update(encoder, target, current_ema):
        with torch.no_grad():
            params_k = []
            params_q = []
            for param_q, param_k in zip(encoder.parameters(), target.parameters()):
                params_k.append(param_k)
                params_q.append(param_q)
            torch._foreach_mul_(params_k, current_ema)
            torch._foreach_add_(params_k, params_q, alpha=1 - current_ema)
        return target
    
    
    
    with log_stats:
        log_stats.start_training("Pretraining Video Jepa")
        iter_retries = 0
        for epoch in range(epochs):
            log_stats.start_epoch(epoch, len(train_loader), desc = "Training")
            encoder.train()
            predictor.train()
            target.eval()
            # -- Linearly increase ema
            temp_train = iter(train_loader)
            current_ema = ema[0] + (ema[1] - ema[0]) * ((epoch + 1) / epochs)
            for iteration in log_stats.batch_iterator([i for i in range(ipe)]):
                optim.zero_grad()
                
                try:
                    sample = next(temp_train)
                except StopIteration:
                    temp_train = iter(train_loader)
                except Exception as e:
                    max_iter = 5
                    if iter_retries < max_iter:
                        logger.WARNING(f"Something went wrong with Dataloader at {iteration=}. Skipping")
                        iter_retries += 1
                        time.sleep(1)
                    else:
                        logger.ERROR(f"Something went wrong with Dataloader at {iteration=}. Exceeded max retries. Ending this epoch")
                        raise e

                all_clips, masks_encs, masks_preds = load_clips(sample)

                    
                
                with torch.amp.autocast(device_str, dtype):
                    h = forward_target(all_clips)
                    z = forward_pred(all_clips, masks_encs, masks_preds)

                    loss, emb_std = loss_fn(z, h, masks_preds)
                    
                    
                loss.backward() 
                optim.step()
                loss_val = loss.item()
                target = ema_update(encoder, target, current_ema)

                scheduler.step()                
                wd_scheduler.step()
                
                log_stats.log_batch({
                    "P Loss": loss_val,
                    "Vram": get_vram_usage(),
                    "Embeddings Std": emb_std.item()
                }, phase = "train")

            
            log_stats.start_phase(ipe // dataset_ratio, desc = "Validating")
            for iteration in log_stats.batch_iterator([i for i in range(ipe // dataset_ratio)]):
                
                log_stats.log_batch({
                    "P Loss": 0,
                    "Vram": get_vram_usage(),
                    "Embeddings Std": 0
                }, phase = "val")
                    
            current_lr = optim.param_groups[0]['lr']
            current_wd = optim.param_groups[0]['weight_decay']
            log_stats.log_epoch(
                extra_metrics = {
                    "Current LR": current_lr,
                    "Current WD": current_wd,
                    "Current Ema": current_ema
                }
            )
                    
            gc.collect()
            torch.cuda.empty_cache()

            enc_stop(log_stats.get_metric("P Loss", "train"), encoder, epoch, optim)
            pred_stop(log_stats.get_metric("P Loss", "train"), predictor, epoch, optim)
            target_stop(log_stats.get_metric("P Loss", "train"), target, epoch, optim)
                
            
        
    
if __name__ == "__main__":
    yaml_path = "./JEPA_ACT/cfgs/pretrain-224px-512.12e-384.12p.yaml"
    with open(yaml_path, "r") as f:
        args = yaml.safe_load(f)
        
    main(args, yaml_path)