import copy
from model.JEPA_ACT.datasets.dataset import VideoDataset
from torch.utils.data import DataLoader

def compile_dataloader(
    train_cfg,
    val_cfg,
    nclips,
    transform,
    collate_fn, 
    num_workers,
    persistance_workers,
    pin_memory
):
    
    
    train = VideoDataset(
        data_paths = [dataset['path'] for dataset in train_cfg['datasets']],
        dataset_fpc = [dataset['fpcs'] for dataset in train_cfg['datasets']],
        frame_step = train_cfg['fps'],
        nclips = nclips,
        individual_transform = transform,
        allow_clip_overlap = train_cfg['allow_clip_overlap'],
        random_jiggle_part = train_cfg['random_jiggle']
    )

    val = VideoDataset(
        data_paths = [dataset['path'] for dataset in val_cfg['datasets']],
        dataset_fpc = [dataset['fpcs'] for dataset in val_cfg['datasets']],
        frame_step = val_cfg['fps'],
        nclips = nclips,
        individual_transform = transform,
        allow_clip_overlap = val_cfg['allow_clip_overlap'],
        random_jiggle_part = val_cfg['random_jiggle']
    )

    train_loader = DataLoader(
        dataset = train,
        batch_size = train_cfg['batch_size'],
        collate_fn = collate_fn,
        pin_memory = pin_memory,
        num_workers = num_workers,
        persistent_workers = persistance_workers,
        shuffle = True
    )

    val_loader = DataLoader(
        dataset = val,
        batch_size = val_cfg['batch_size'],
        collate_fn = collate_fn,
        pin_memory = pin_memory,
        num_workers = num_workers,
        persistent_workers = persistance_workers,
        shuffle = False
    )

    return train_loader, val_loader