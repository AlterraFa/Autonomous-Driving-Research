import torch
from datasets.dataset import StraighteningProbeDataset
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
    rank,
    **kwargs
):
    dataset = StraighteningProbeDataset(
        data_paths=train_cfg['datasets_path'],
        shared_transform=transform,
        waypoint_key=train_cfg.get('waypoint_key', 'midlane_wp'),
        n_waypoints=train_cfg.get('n_waypoints', 12),
        wp_clip=train_cfg.get('wp_clip', None),
        wp_normalize=train_cfg.get('wp_normalize', False),
    )

    train_fraction = float(train_cfg.get('train_fraction', 0.9))
    val_fraction = float(train_cfg.get('val_fraction', 0.1))
    train_set, val_set, test_set = dataset.split(train=train_fraction, val=val_fraction)

    train_sampler = torch.utils.data.DistributedSampler(
        train_set, num_replicas=world_sz, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_set, num_replicas=world_sz, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_cfg['batch_size'],
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistance_workers,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=train_cfg['batch_size'],
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistance_workers,
        sampler=val_sampler,
        drop_last=False,
    )

    logger.INFO("StraighteningProbe dataloader initialized:")
    logger.INFO({
        "datasets_path": train_cfg['datasets_path'],
        "waypoint_key": train_cfg.get('waypoint_key', 'midlane_wp'),
        "n_waypoints": train_cfg.get('n_waypoints', 12),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "batch_size": train_cfg['batch_size'],
    })

    return train_loader, val_loader, train_sampler, val_sampler, {}