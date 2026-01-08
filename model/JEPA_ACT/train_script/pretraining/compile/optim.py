import torch
from model.JEPA_ACT.utils.schedulers import CosineWDSchedule, CosineSchedule

def compile_optim(
    encoder,
    predictor,
    betas,
    init_lr,
    final_lr,
    init_wd,
    final_wd,
    epochs, 
    ipe
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr = init_lr,
        betas = betas
    )
    
    scheduler = CosineSchedule(
        optimizer = optimizer,
        ref_lr = init_lr,
        final_lr = final_lr,
        T_max = int(epochs * ipe),
    )
    
    wd_scheduler = CosineWDSchedule(
        optimizer, 
        ref_wd = init_wd,
        final_wd = final_wd,
        T_max  = int(epochs * ipe),
    )
    
    return optimizer, scheduler, wd_scheduler