import torch
from datasets.dataset import ActVideoDataset
from torch.utils.data import DataLoader
from utils.logger import Logger

logger = Logger(__name__)

def compile_dataloader(
    train_cfg,
    nclips,
    transform,
    collate_fn, 
    num_workers,
    persistance_workers,
    pin_memory,
    world_sz,
    rank
):
    
    train = ActVideoDataset(
        data_paths = [dataset['path'] for dataset in train_cfg['datasets']],
        frame_step = [dataset['fps'] for dataset in train_cfg['datasets']],
        ctx_frames_per_clips = train_cfg['ctx_fpcs'],
        pred_frames_per_clips = train_cfg['pred_fpcs'],
        nclips = nclips,
        individual_transform = transform,
        allow_clip_overlap = train_cfg['allow_clip_overlap'],
        random_jiggle_part = train_cfg['random_jiggle']
    )

    dist_sampler = torch.utils.data.DistributedSampler(
        train, num_replicas = world_sz, rank = rank, shuffle = True
    )

    train_loader = DataLoader(
        dataset = train,
        batch_size = train_cfg['batch_size'],
        collate_fn = collate_fn,
        pin_memory = pin_memory,
        num_workers = num_workers,
        persistent_workers = persistance_workers,
        sampler = dist_sampler,
        drop_last = True
    )

    logger.INFO("Data loader and distributed sampler initialized with:")
    logger.INFO({
        "dataset": {
            "data_paths": [dataset['path'] for dataset in train_cfg['datasets']],
            "frame_step": [dataset['fps'] for dataset in train_cfg['datasets']],
            "ctx_frames_per_clips": train_cfg['ctx_fpcs'],
            "pred_frames_per_clips": train_cfg['pred_fpcs'],
            "nclips": nclips,
            "allow_clip_overlap": train_cfg['allow_clip_overlap'],
            "random_jiggle_part": train_cfg['random_jiggle'],
        },
        "dataloader": {
            "batch_size": train_cfg['batch_size'],
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": persistance_workers,
            "drop_last": True,
        },
        "distributed_sampler": {
            "num_replicas": world_sz,
            "rank": rank,
            "shuffle": True,
        },
    })


    return train_loader, dist_sampler