import os, sys
import copy
import yaml
import torch

script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from model.JEPA_VENL.impl.transformer import (
    ViTEncode,
    ViTPredictor,
    apply_masks,
    repeat_interleave_batch
)
from model.JEPA_VENL.data_utils.jepa.compose import (
    optim_schedulers_composer,
    dataloader_composer,
    transform_composer
)
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

torch.manual_seed(12)
def main(args, yaml_path):
    
    # -- Encoder args (excluding architecture)
    enc_cfg = args['model']['encoder']
    image_size     = tuple(enc_cfg['image_size'])
    patch_size     = enc_cfg['patch_size']
    in_channels    = enc_cfg['in_channels']
    embed_dim      = enc_cfg['embed_dim']
    depth          = enc_cfg['depth']
    num_heads      = enc_cfg['num_heads']
    mlp_ratio      = enc_cfg['mlp_ratio']
    qkv_bias       = enc_cfg['qkv_bias']
    qk_scale       = enc_cfg['qk_scale']
    drop_rate      = enc_cfg['drop_rate']
    attn_drop_rate = enc_cfg['attn_drop_rate']
    drop_path_rate = enc_cfg['drop_path_rate']
    enc_init_std   = enc_cfg['init_std']
    
    # -- Predictor args (excluding architecture)
    pred_cfg = args['model']['predictor']
    pred_embed_dim      = pred_cfg['embed_dim']
    pred_depth          = pred_cfg['depth']
    pred_num_heads      = pred_cfg['num_heads']
    pred_mlp_ratio      = pred_cfg['mlp_ratio']
    pred_qkv_bias       = pred_cfg['qkv_bias']
    pred_qk_scale       = pred_cfg['qk_scale']
    pred_drop_rate      = pred_cfg['drop_rate']
    pred_attn_drop_rate = pred_cfg['attn_drop_rate']
    pred_drop_path_rate = pred_cfg['drop_path_rate']
    pred_init_std       = pred_cfg['init_std']
    
    # -- Data args
    data_cfg = args['data']
    dataset_path  = data_cfg['dataset_path']
    batch_size    = data_cfg['batch_size']
    num_workers   = data_cfg['num_workers']
    train_split   = data_cfg['train_split']
    val_split     = data_cfg['val_split']
    shuffle       = data_cfg['shuffle']
    augmentations = data_cfg['augmentations']
    
    # -- Collator args
    collator_cfg = data_cfg['collator']
    enc_mask_scale  = collator_cfg['enc_mask_scale']
    pred_mask_scale = collator_cfg['pred_mask_scale']
    aspect_ratio    = collator_cfg['aspect_ratio']
    nenc            = collator_cfg['nenc']
    npred           = collator_cfg['npred']
    allowed_overlap = collator_cfg['allowed_overlap']
    min_patches     = collator_cfg['min_patches']
    
    # -- Training args
    train_cfg = args['train_param']
    ema           = train_cfg['ema']
    init_lr       = train_cfg['init_lr']
    final_lr      = train_cfg['final_lr']
    weight_decay  = train_cfg['weight_decay']
    epochs        = train_cfg['epochs']
    betas         = train_cfg['betas']
    patience      = train_cfg['early_stopping']['patience']
    min_delta     = train_cfg['early_stopping']['delta']
    plateau_point = train_cfg['plateau_point']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ================================================= #
    #               INITIALIZE MODELS
    # ================================================= #
    encoder   = ViTEncode(
        img_size       = image_size,
        patch_size     = patch_size,
        in_chans       = in_channels,
        embed_dim      = embed_dim,
        depth          = depth,
        num_heads      = num_heads,
        mlp_ratio      = mlp_ratio,
        qkv_bias       = qkv_bias,
        qk_scale       = qk_scale,
        drop_rate      = drop_rate,
        attn_drop_rate = attn_drop_rate,
        drop_path_rate = drop_path_rate,
        init_std       = enc_init_std
    ).to(device)
    target    = copy.deepcopy(encoder) # EMA model
    for p in target.parameters():
        p.requires_grad = False
    predictor = ViTPredictor(
        num_patches          = (image_size[0] // patch_size, image_size[1] // patch_size),
        embed_dim            = embed_dim,
        predictor_embed_dim  = pred_embed_dim,
        depth                = pred_depth,
        num_heads            = pred_num_heads,
        mlp_ratio            = pred_mlp_ratio,
        qkv_bias             = pred_qkv_bias,
        qk_scale             = pred_qk_scale,
        drop_rate            = pred_drop_rate,
        attn_drop_rate       = pred_attn_drop_rate,
        drop_path_rate       = pred_drop_path_rate,
        init_std             = pred_init_std
    ).to(device)
    
    
    # ================================================= #
    #               DATASET LOADING
    # ================================================= #
    transform = transform_composer(
        dimension       = image_size,
        crop            = augmentations['crop'],
        color_jitter    = augmentations['color_jitter'],
        horizontal_flip = augmentations['horizontal_flip'],
        gaussian_blur   = augmentations["gaussian_blur"],
        random_resize   = augmentations["random_resize"],
        scale           = augmentations['scale'],
        ratio           = augmentations["ratio"],
        normalization   = augmentations["normalization"]
    )
    
    train_loader, val_loader, test_loader = dataloader_composer(
        root            = os.path.join(FOLDER_DIR, dataset_path),
        transform       = transform,
        split           = [train_split, val_split],
        image_size      = image_size,
        patch_size      = patch_size,
        enc_mask_scale  = enc_mask_scale,
        pred_mask_scale = pred_mask_scale,
        aspect_ratio    = aspect_ratio,
        nenc            = nenc,
        npred           = npred,
        allowed_overlap = allowed_overlap,
        min_patches     = min_patches,
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = num_workers
    )
    
    # ================================================= #
    #           TRAINING OPTIMIZER AND SCHEDULER
    # ================================================= #
    # -- seperate into bias w/ layernorm (bias in n and len(p.shape) == 1) params and large array params
    optimizer, scheduler = optim_schedulers_composer(
        encoder = encoder,
        predictor = predictor,
        epochs = epochs,
        plateau_point = float(plateau_point),
        init_lr = init_lr,
        final_lr = final_lr,
        weight_decay = weight_decay,
        betas = betas
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
    enc_stop    = EarlyStopping(patience = patience, min_delta = min_delta, path = os.path.join(log_dir, f"run{run_idx}/weights/encoder_run{run_idx}.pt"))
    pred_stop   = EarlyStopping(patience = patience, min_delta = min_delta, path = os.path.join(log_dir, f"run{run_idx}/weights/predictor_run{run_idx}.pt"))
    target_stop = EarlyStopping(patience = patience, min_delta = min_delta, path = os.path.join(log_dir, f"run{run_idx}/weights/target_run{run_idx}.pt"))
    os.system(f"cp {yaml_path} {os.path.join(log_dir, f'run{run_idx}')}")
    
    with log_stats:
        log_stats.start_training("Pretrain JEPA")
        for epoch_idx in range(epochs):
            # =================== Training ==================== #

            log_stats.start_epoch(epoch_idx, len(train_loader), desc = "Training")
            encoder.train()
            predictor.train()
            target.eval()
            # -- Linearly increase ema
            current_ema = ema[0] + (ema[1] - ema[0]) * ((epoch_idx + 1) / epochs)
            for img_batch, enc_masks, pred_masks in log_stats.batch_iterator(train_loader):
                optimizer.zero_grad()
                
                def to_device(img_batch, enc_masks, pred_masks):
                    img_batch  = img_batch[0].to(device)
                    enc_masks  = [mask.to(device) for mask in enc_masks]
                    pred_masks = [mask.to(device) for mask in pred_masks]
                    return img_batch, enc_masks, pred_masks
                img_batch, enc_masks, pred_masks = to_device(img_batch, enc_masks, pred_masks)

                def forward_pred(img, enc_masks, pred_masks):
                    z = encoder(img, enc_masks)
                    z = predictor(z, enc_masks, pred_masks)
                    return z
                
                def forward_target(img, enc_masks, pred_masks):
                    with torch.no_grad():
                        h = target(img)
                        B = h.shape[0]
                        # h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        h = apply_masks(h, pred_masks)
                        h = repeat_interleave_batch(h, B, repeat = len(enc_masks))
                    return h

                def ema_update(encoder, target):
                    with torch.no_grad():
                        for enc_param, tar_param in zip(encoder.parameters(), target.parameters()):
                            tar_param.data.mul_(current_ema)
                            tar_param.data.add_(enc_param.data, alpha = (1 - current_ema))
                        return target
                
                with torch.amp.autocast("cuda", dtype = torch.bfloat16):
                    z = forward_pred(img_batch, enc_masks, pred_masks)
                    h = forward_target(img_batch, enc_masks, pred_masks)
                    loss = F.smooth_l1_loss(z, h)
                    emb_std = torch.std(h, dim = 0).mean()
                    

                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                target   = ema_update(encoder, target)        
                
                log_stats.log_batch({
                    "L1 Loss": loss_val,
                    "Vram": get_vram_usage(),
                    "Enc Patch": enc_masks[0].size(-1),
                    "Pred Patch": pred_masks[0].size(-1),
                    "Embeddings Std": emb_std.item()
                }, phase = "train")
            
                
            log_stats.start_phase(len(val_loader), desc = "Validating")
            encoder.eval()
            predictor.eval()
            target.eval()
            # =================== Validating ==================== #
            for img_batch, enc_masks, pred_masks in log_stats.batch_iterator(val_loader):
                
                def to_device(img_batch, enc_masks, pred_masks):
                    img_batch  = img_batch[0].to(device)
                    enc_masks  = [mask.to(device) for mask in enc_masks]
                    pred_masks = [mask.to(device) for mask in pred_masks]
                    return img_batch, enc_masks, pred_masks
                img_batch, enc_masks, pred_masks = to_device(img_batch, enc_masks, pred_masks)

                def forward_pred(img, enc_masks, pred_masks):
                    with torch.no_grad():
                        z = encoder(img, enc_masks)
                        z = predictor(z, enc_masks, pred_masks)
                        return z
                
                def forward_target(img, enc_masks, pred_masks):
                    with torch.no_grad():
                        h = target(img)
                        B = h.shape[0]
                        # h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        h = apply_masks(h, pred_masks)
                        h = repeat_interleave_batch(h, B, repeat = len(enc_masks))
                    return h
                
                with torch.amp.autocast("cuda", dtype = torch.bfloat16):
                    z = forward_pred(img_batch, enc_masks, pred_masks)
                    h = forward_target(img_batch, enc_masks, pred_masks)
                    loss = F.smooth_l1_loss(z, h)
                    emb_std = torch.std(h, dim = 0).mean()
                    

                loss_val = loss.item()
                
                log_stats.log_batch({
                    "L1 Loss": loss_val,
                    "Vram": get_vram_usage(),
                    "Enc Patch": enc_masks[0].size(-1),
                    "Pred Patch": pred_masks[0].size(-1),
                    "Embeddings Std": emb_std
                }, phase = "val") 
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            log_stats.log_epoch(
                extra_metrics = {
                    "Current LR": current_lr,
                    "Current Ema": current_ema
                }
            )

            torch.cuda.empty_cache()
            enc_stop(log_stats.get_metric("L1 Loss", "val"), encoder)
            pred_stop(log_stats.get_metric("L1 Loss", "val"), predictor)
            target_stop(log_stats.get_metric("L1 Loss", "val"), target)
            if enc_stop.early_stop:
                break

    
FOLDER_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    YAML_PATH   = os.path.join(FOLDER_DIR, "../configs/jepa/config.yaml")
    with open(YAML_PATH, "r") as f:
        args = yaml.safe_load(f)
    main(args, YAML_PATH)