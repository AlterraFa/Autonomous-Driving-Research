import os, sys

script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader

def optim_schedulers_composer(nav_model, full_finetune, epochs, plateau_point, init_lr, final_lr, weight_decay, betas):
    
    param_groups = [
        {
            'params': (p for n, p in nav_model.readout.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in nav_model.readout.named_parameters()
                    if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, 
    ]
    if full_finetune:
        param_groups += [
            {
                'params': (p for n, p in nav_model.backbone.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in nav_model.backbone.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
    
    milestone = int(epochs * plateau_point)
    optimizer = optim.AdamW(
        params = param_groups, 
        lr = init_lr, 
        weight_decay = weight_decay, 
        betas = betas
    )
    cosine = lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer, 
        T_max = milestone, 
        eta_min = final_lr
    )
    plateau = lr_scheduler.ConstantLR(
        optimizer = optimizer, 
        factor = final_lr / init_lr,
        total_iters = epochs - milestone + 10
    ) 
    scheduler = lr_scheduler.SequentialLR(
        optimizer = optimizer,
        schedulers = [cosine, plateau],
        milestones = [milestone]
    )
    return optimizer, scheduler

def transform_composer(
    dimension: tuple = (244, 244),
    crop=[0, 1.0],
    color_jitter=1.0,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    from JEPA_VENL.data_utils.jepa.image_transform import GuidedCrop, GaussianBlur
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        return rnd_color_jitter
        
    transform_list = []
    transform_list += [
        GuidedCrop(*crop),
        transforms.Resize(dimension)
    ]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.25, radius_max = 1.0)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]
    transform = transforms.Compose(transform_list)

    # -- Transform for rgb image, transform for unrouted and routed map
    return transform, transforms.ToTensor(), transforms.ToTensor()


def dataloader_composer(
    root: str,
    ram_caching,
    split,
    batch_size,
    shuffle,
    num_workers,
    pad_value  
):
    from model.JEPA_VENL.data_utils.jepa_nav.dataloader import JEPANavLoader
    from model.JEPA_VENL.data_utils.jepa.multiblock import JEPACollator
    trainset, valset, testset = JEPANavLoader(
        root        = root,
        fraction    = 1.0,
        ram_caching = ram_caching,
        pad_value   = pad_value
    ).split(*split)
    
    collator = trainset.dataset.collate_fn
    
    train_loader = DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        persistent_workers = True,
        collate_fn = collator
    )
    val_loader = DataLoader(
        dataset = valset,
        batch_size = batch_size, # Faster for validation
        shuffle = shuffle,
        num_workers = num_workers,
        persistent_workers = True,
        collate_fn = collator
    )
    try:
        test_loader = DataLoader(
            dataset = testset,
            batch_size = batch_size, # Faster for validation
            shuffle = shuffle,
            num_workers = num_workers,
            persistent_workers = True,
            collate_fn = collator
        )
    except:
        test_loader = None
    return train_loader, val_loader, test_loader