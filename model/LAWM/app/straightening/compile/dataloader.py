import torch
from datasets.dataset import StraighteningDataset
from torch.utils.data import DataLoader
from utils.logger import Logger

logger = Logger(__name__)

def compile_dataloader(
    train_cfg,
    transform,
    collate_fn, 
    num_workers,
    persistance_workers,
    pin_memory,
    world_sz,
    rank
):
    
    train = StraighteningDataset(
        data_paths = train_cfg['datasets_path'],
        shared_transform = transform,
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

    logger.DEBUG("Data loader and distributed sampler initialized with:")
    logger.DEBUG({
        "dataset": {
            "data_paths": train_cfg['datasets_path'],
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