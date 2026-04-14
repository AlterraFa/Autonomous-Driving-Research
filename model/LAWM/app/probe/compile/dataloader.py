import torch
import numpy as np
import warnings
from datasets.dataset import StraighteningProbeDataset
from datasets.sampler import DistributedWeightedSampler, WeightedSampler
from torch.utils.data import DataLoader
from torch.utils.data import Subset
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


def _build_probe_train_sampler(train_set, train_cfg, world_sz, rank):
    sampling_cfg = train_cfg.get('sampling', {})
    if not sampling_cfg.get('enabled', False):
        return torch.utils.data.DistributedSampler(
            train_set, num_replicas=world_sz, rank=rank, shuffle=True
        ), {"sampling_mode": "distributed_shuffle"}

    if not isinstance(train_set, Subset):
        warnings.warn("Expected train_set to be a Subset; falling back to DistributedSampler")
        return torch.utils.data.DistributedSampler(
            train_set, num_replicas=world_sz, rank=rank, shuffle=True
        ), {"sampling_mode": "distributed_shuffle_fallback"}

    multi_frac = float(sampling_cfg.get('multi_fraction', 0.5))
    uni_frac = float(sampling_cfg.get('uni_fraction', 0.5))
    if multi_frac < 0 or uni_frac < 0:
        raise ValueError("sampling.multi_fraction and sampling.uni_fraction must be >= 0")
    denom = multi_frac + uni_frac
    if denom <= 0:
        raise ValueError("sampling.multi_fraction + sampling.uni_fraction must be > 0")

    # Normalize user-provided fractions so values like 40/50 are valid.
    p_multi = multi_frac / denom
    p_uni = uni_frac / denom

    base_dataset = train_set.dataset
    subset_indices = list(train_set.indices)

    class_labels = [_safe_read_road_type(base_dataset, i) for i in subset_indices]
    n_multi = sum(1 for c in class_labels if c == 'multi')
    n_uni = sum(1 for c in class_labels if c == 'uni')

    if n_multi == 0 or n_uni == 0:
        warnings.warn(
            f"sampling.enabled=True but class counts are uni={n_uni}, multi={n_multi}; "
            "falling back to DistributedSampler"
        )
        return torch.utils.data.DistributedSampler(
            train_set, num_replicas=world_sz, rank=rank, shuffle=True
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

    num_samples = int(sampling_cfg.get('num_samples', len(train_set)))
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
    persistance_workers=None,
    pin_memory=False,
    world_sz=1,
    rank=0,
    **kwargs
):
    if persistance_workers is None:
        persistance_workers = kwargs.get('persistent_workers', False)

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

    train_sampler, sampler_info = _build_probe_train_sampler(
        train_set=train_set,
        train_cfg=train_cfg,
        world_sz=world_sz,
        rank=rank,
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
        "sampling": sampler_info,
    })

    return train_loader, val_loader, train_sampler, val_sampler, {}