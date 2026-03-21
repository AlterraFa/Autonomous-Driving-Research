import torch
from utils.schedulers import WSDSchedule, CosineWDSchedule
from utils.logger import Logger
from utils.grad_optim import create_gradient_optimizer

logger = Logger(__name__)

def compile_opt(
    encoder,
    probe,
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
    lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in probe.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": lr_scale,
        },
        {
            "params": (p for n, p in probe.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
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

    logger.INFO("Optimizer, weight decay and learning rate scheduler initialized with:")
    logger.INFO({
        "optimizer": {
            "type": "AdamW",
            "betas": betas,
            "eps": eps,
        },
        "lr_scheduler": {
            "type": "WSDSchedule",
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


def compile_grad_optimizer(
    base_optimizer,
    optimizer_name="normal",
    n_tasks=None,
    device="cuda",
    **kwargs
):
    """
    Create a gradient optimizer for multi-task learning.
    
    Args:
        base_optimizer: PyTorch optimizer instance
        optimizer_name: Type of gradient optimizer ('normal', 'pcgrad', 'gradnorm', 'famo')
        n_tasks: Number of tasks (required for GradNorm and FAMO)
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Additional optimizer-specific parameters
                 - For GradNorm: alpha, w_lr
                 - For FAMO: gamma, w_lr, max_norm
    
    Returns:
        Initialized GradientOptim instance
    """
    grad_optim = create_gradient_optimizer(
        optimizer_name=optimizer_name,
        base_optimizer=base_optimizer,
        n_tasks=n_tasks,
        device=device,
        **kwargs
    )
    
    logger.INFO(f"Gradient optimizer initialized: {optimizer_name.upper()}")
    return grad_optim
