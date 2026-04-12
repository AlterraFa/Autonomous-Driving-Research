import os, sys

script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

from torch.utils.data import DataLoader

def optim_schedulers_composer(encoder, predictor, epochs, plateau_point, init_lr, final_lr, weight_decay, betas):
    
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
        dimension=224,
        crop=[0, 1.0],
        color_jitter=1.0,
        horizontal_flip=False,
        color_distortion=False,
        gaussian_blur=False,
        random_resize=False,
        scale=(0.08, 1.0),
        ratio=(0.75, 4/3),
        normalization=((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ):
    from model.JEPA_VENL.data_utils.jepa.image_transform import GuidedCrop, GaussianBlur
    
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort
    def get_resize_transform(random = False):
        default_resize = transforms.Resize(dimension)
        random_resize  = transforms.RandomResizedCrop(dimension, scale, ratio)
        return (
            [default_resize, random_resize] if random else [default_resize], 
            [0.5, 0.5] if random else [1.0]
        )
        
    transform_list = []
    transform_list += [
        GuidedCrop(*crop),
        transforms.RandomChoice(*get_resize_transform(random_resize))
    ]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform

def dataloader_composer(
    root: str,
    transform,
    split,
    image_size,
    patch_size, 
    enc_mask_scale,
    pred_mask_scale, 
    aspect_ratio,
    nenc, 
    npred,
    allowed_overlap,
    min_patches, 
    batch_size,
    shuffle,
    num_workers  
):
    from model.JEPA_VENL.data_utils.jepa.dataloader import JEPALoader
    from model.JEPA_VENL.data_utils.jepa.multiblock import JEPACollator
    trainset, valset, testset = JEPALoader(
        root = root,
        transform = transform,
        fraction = 1.0
    ).split(*split)
    collator = JEPACollator(
        input_size      = image_size,
        patch_size      = patch_size,
        enc_mask_scale  = enc_mask_scale,
        pred_mask_scale = pred_mask_scale,
        aspect_ratio    = aspect_ratio,
        nenc            = nenc,
        npred           = npred,
        allow_overlap   = allowed_overlap,
        min_patches     = min_patches
    )
    train_loader = DataLoader(
        dataset            = trainset,
        batch_size         = batch_size,
        pin_memory         = True,
        shuffle            = shuffle,
        num_workers        = num_workers,
        persistent_workers = True,
        collate_fn         = collator
    )
    val_loader = DataLoader(
        dataset            = valset,
        batch_size         = batch_size * 5, # Faster for validation
        pin_memory         = True,
        shuffle            = shuffle,
        num_workers        = num_workers,
        persistent_workers = True,
        collate_fn         = collator
    )
    test_loader = None if testset == None else DataLoader(
        dataset            = testset,
        batch_size         = batch_size * 5, # Faster for validation
        pin_memory         = True,
        shuffle            = shuffle,
        num_workers        = num_workers,
        persistent_workers = True,
        collate_fn         = collator
    )
    return train_loader, val_loader, test_loader