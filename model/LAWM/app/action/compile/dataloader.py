import torch
import numpy as np
import warnings
from datasets.dataset import FormattedVideoDataset
from datasets.sampler import DistributedWeightedSampler, WeightedSampler
from torch.utils.data import DataLoader
from utils.logger import Logger

logger = Logger(__name__)


def _safe_read_road_type(dataset, sample_index):
    sample_path = dataset.samples[sample_index]
    try:
        data = np.load(sample_path, allow_pickle=True).item()
        metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
        condition = metadata.get('condition', {}) if isinstance(metadata, dict) else {}
        road_type = condition.get('road_type', 'uni')
        if road_type not in ('uni', 'multi'):
            return 'uni'
        return road_type
    except Exception:
        return 'uni'


def _build_action_train_sampler(dataset, train_cfg, world_sz, rank):
    sampling_cfg = train_cfg.get('sampling', {})
    if not sampling_cfg.get('enabled', False):
        return torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_sz, rank=rank, shuffle=True
        ), {"sampling_mode": "distributed_shuffle"}

    multi_frac = float(sampling_cfg.get('multi_fraction', 0.5))
    uni_frac = float(sampling_cfg.get('uni_fraction', 0.5))
    if multi_frac < 0 or uni_frac < 0:
        raise ValueError("sampling.multi_fraction and sampling.uni_fraction must be >= 0")
    denom = multi_frac + uni_frac
    if denom <= 0:
        raise ValueError("sampling.multi_fraction + sampling.uni_fraction must be > 0")

    p_multi = multi_frac / denom
    p_uni = uni_frac / denom

    class_labels = [_safe_read_road_type(dataset, i) for i in range(len(dataset))]
    n_multi = sum(1 for c in class_labels if c == 'multi')
    n_uni = sum(1 for c in class_labels if c == 'uni')

    if n_multi == 0 or n_uni == 0:
        warnings.warn(
            f"sampling.enabled=True but class counts are uni={n_uni}, multi={n_multi}; "
            "falling back to DistributedSampler"
        )
        return torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_sz, rank=rank, shuffle=True
        ), {
            "sampling_mode": "distributed_shuffle_fallback",
            "uni_count": n_uni,
            "multi_count": n_multi,
        }

    per_sample_target = {
        'uni': p_uni / n_uni,
        'multi': p_multi / n_multi,
    }
    weights = [per_sample_target[c] for c in class_labels]

    num_samples = int(sampling_cfg.get('num_samples', len(dataset)))
    seed = int(sampling_cfg.get('seed', 0))
    if world_sz > 1:
        sampler = DistributedWeightedSampler(
            weights=weights,
            num_samples=num_samples,
            num_replicas=world_sz,
            rank=rank,
            seed=seed,
        )
        sampling_mode = "distributed_weighted"
    else:
        sampler = WeightedSampler(
            weights=weights,
            num_samples=num_samples,
            seed=seed,
        )
        sampling_mode = "weighted"

    return sampler, {
        "sampling_mode": sampling_mode,
        "uni_count": n_uni,
        "multi_count": n_multi,
        "requested_multi_fraction": multi_frac,
        "requested_uni_fraction": uni_frac,
        "effective_multi_probability": p_multi,
        "effective_uni_probability": p_uni,
        "num_samples": num_samples,
    }


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
    
    train = FormattedVideoDataset(
        data_paths = train_cfg['datasets_path'],
        shared_transform = transform,
        fpcs = train_cfg['fpcs'],
        frame_selection = train_cfg['interpolation_mode']
    )

    sampler, sampler_info = _build_action_train_sampler(
        dataset=train,
        train_cfg=train_cfg,
        world_sz=world_sz,
        rank=rank,
    )

    train_loader = DataLoader(
        dataset = train,
        batch_size = train_cfg['batch_size'],
        collate_fn = collate_fn,
        pin_memory = pin_memory,
        num_workers = num_workers,
        persistent_workers = persistance_workers,
        sampler = sampler,
        drop_last = True
    )

    logger.DEBUG("Data loader and sampler initialized with:")
    logger.DEBUG({
        "dataset": {
            "data_paths": train_cfg['datasets_path'],
            'fpcs': train_cfg['fpcs'],
            'frame_selection': train_cfg['interpolation_mode']
        },
        "dataloader": {
            "batch_size": train_cfg['batch_size'],
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": persistance_workers,
            "drop_last": True,
        },
        "sampling": sampler_info,
    })


    return train_loader, sampler