import copy
from ....datasets.dataset import ActVideoDataset
from torch.utils.data import DataLoader

def compile_dataloader(
    train_cfg,
    nclips,
    transform,
    collate_fn, 
    num_workers,
    persistance_workers,
    pin_memory
):
    
    
    train = ActVideoDataset(
        data_paths = [dataset['path'] for dataset in train_cfg['datasets']],
        frame_step = train_cfg['fps'],
        ctx_frames_per_clips = train_cfg['ctx_fpcs'],
        pred_frames_per_clips = train_cfg['pred_fpcs'],
        nclips = nclips,
        individual_transform = transform,
        allow_clip_overlap = train_cfg['allow_clip_overlap'],
        random_jiggle_part = train_cfg['random_jiggle']
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


    return train_loader