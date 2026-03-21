import torch
from datasets.dataset import ProbeDataset
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
    
    dataset = ProbeDataset(
        data_paths = [dataset['path'] for dataset in train_cfg['datasets']],
        frame_step = [dataset['fps'] for dataset in train_cfg['datasets']],
        frames_per_clips = train_cfg['fpcs'],
        nclips = nclips,
        individual_transform = transform,
        allow_clip_overlap = train_cfg['allow_clip_overlap'],
        random_jiggle_part = train_cfg['random_jiggle']
    )

    train_fraction = float(train_cfg.get('train_fraction', 0.9))
    val_fraction = float(train_cfg.get('val_fraction', 0.1))
    train_set, val_set, test_set = dataset.split(train=train_fraction, val=val_fraction)

    gt_stats = dataset.statistics(indices=train_set.indices)
    if rank == 0:
        logger.INFO("Train ground-truth statistics:")
        logger.INFO(gt_stats)

    train_sampler = torch.utils.data.DistributedSampler(
        train_set, num_replicas = world_sz, rank = rank, shuffle = True
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_set, num_replicas = world_sz, rank = rank, shuffle = False
    )

    train_loader = DataLoader(
        dataset = train_set,
        batch_size = train_cfg['batch_size'],
        collate_fn = collate_fn,
        pin_memory = pin_memory,
        num_workers = num_workers,
        persistent_workers = persistance_workers,
        sampler = train_sampler,
        drop_last = True
    )

    val_loader = DataLoader(
        dataset = val_set,
        batch_size = train_cfg['batch_size'],
        collate_fn = collate_fn,
        pin_memory = pin_memory,
        num_workers = num_workers,
        persistent_workers = persistance_workers,
        sampler = val_sampler,
        drop_last = False
    )

    logger.INFO("Data loader and distributed sampler initialized with:")
    logger.INFO({
        "dataset": {
            "data_paths": [dataset['path'] for dataset in train_cfg['datasets']],
            "frame_step": [dataset['fps'] for dataset in train_cfg['datasets']],
            "frames_per_clips": train_cfg['fpcs'],
            "nclips": nclips,
            "allow_clip_overlap": train_cfg['allow_clip_overlap'],
            "random_jiggle_part": train_cfg['random_jiggle'],
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": len(test_set),
        },
        "train_dataloader": {
            "batch_size": train_cfg['batch_size'],
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": persistance_workers,
            "drop_last": True,
        },
        "val_dataloader": {
            "batch_size": train_cfg['batch_size'],
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": persistance_workers,
            "drop_last": False,
        },
        "train_sampler": {
            "num_replicas": world_sz,
            "rank": rank,
            "shuffle": True,
        },
        "val_sampler": {
            "num_replicas": world_sz,
            "rank": rank,
            "shuffle": False,
        },
    })

    return train_loader, val_loader, train_sampler, val_sampler