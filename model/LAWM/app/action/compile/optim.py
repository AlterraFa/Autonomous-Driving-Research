import torch
from utils.schedulers import CosineWSDSchedule, CosineWDSchedule
from utils.logger import Logger

logger = Logger(__name__)

def compile_opt(
    apred,
    lpred,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in lpred.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in apred.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in lpred.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in apred.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = CosineWSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler() if mixed_precision else None

    logger.DEBUG("Optimizer, weight decay and learning rate scheduler initialized with:")
    logger.DEBUG({
        "optimizer": {
            "type": "AdamW",
            "betas": betas,
            "eps": eps,
        },
        "lr_scheduler": {
            "type": "CosineWSDScheduler",
            "warmup_steps": int(warmup * iterations_per_epoch),
            "anneal_steps": int(anneal * iterations_per_epoch),
            "start_lr": start_lr,
            "ref_lr": ref_lr,
            "final_lr": final_lr,
            "T_max": int(num_epochs * iterations_per_epoch),
        },
        "wd_scheduler": {
            "type": "CosineWDSchedule",
            "ref_wd": wd,
            "final_wd": final_wd,
            "T_max": int(num_epochs * iterations_per_epoch),
        },
    })
    return optimizer, scaler, scheduler, wd_scheduler
