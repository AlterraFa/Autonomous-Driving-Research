"""Pre-compute and cache action latents from the frozen world model.

Usage (from LAWM/):
    python -m app.probe.cache_latents \
        --fname cfgs/probe/probe-action-latent-256px-cartesian.yaml \
        --devices cuda:0 \
        --batch_size 64 \
        --cache_dir ./cached_latents/probe

The script:
  1. Builds the frozen world model + dataset (same as probe training).
  2. Iterates over ALL samples (no shuffle, no sampling weights).
  3. For each sample, saves {a_latent, midlane_wp, gate_score} as an .npz file
     indexed by the sample's position in the dataset.
  4. Also saves a manifest.json mapping index -> original .npy path.

During probe training, set `train.cached_latents_dir` in the YAML to point
to this directory, and the trainer will load pre-computed latents instead of
running the world model.
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

import torch
from ruamel.yaml import YAML

from app.probe.compile import (
    compile_model,
    compile_transform,
)
from datasets.dataset import StraighteningProbeDataset
from utils.logger import Logger

logger = Logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cache action latents for probe training")
    parser.add_argument("--fname", type=str, required=True, help="Path to probe YAML config")
    parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Output directory for cached latents (default: ./cached_latents/probe)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dtype", type=str, default=None, help="Override dtype (bfloat16, float16, float32)")
    args = parser.parse_args()

    # Load config
    yaml = YAML(typ="safe")
    with open(args.fname) as f:
        cfg = yaml.load(f)

    device = torch.device(args.devices[0])
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    augment_cfg = cfg.get("data_aug", {})
    meta_cfg = cfg.get("meta", {})

    dtype_str = args.dtype or meta_cfg.get("dtype", "bfloat16")
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    cache_dir = args.cache_dir or "./cached_latents/probe"
    os.makedirs(cache_dir, exist_ok=True)

    # Build transform (same as training — no augmentation)
    crop_size = train_cfg.get("crop_size", 256)
    transform = compile_transform(
        random_horizontal_flip=augment_cfg.get("horizontal_flip", False),
        random_resize_aspect_ratio=augment_cfg.get("random_resize_aspect_ratio", (1.0, 1.0)),
        random_resize_scale=augment_cfg.get("random_resize_scale", (1.0, 1.0)),
        reprob=augment_cfg.get("reprob", 0.0),
        auto_augment=augment_cfg.get("auto_augment", False),
        motion_shift=augment_cfg.get("motion_shift", False),
        crop_size=crop_size,
    )

    # Build dataset (full, no split)
    dataset = StraighteningProbeDataset(
        data_paths=train_cfg["datasets_path"],
        shared_transform=transform,
        waypoint_key=train_cfg.get("waypoint_key", "midlane_wp"),
        n_waypoints=train_cfg.get("n_waypoints", 12),
        wp_clip=train_cfg.get("wp_clip", None),
        wp_normalize=train_cfg.get("wp_normalize", False),
        wp_center=train_cfg.get("wp_center", None),
    )
    logger.INFO(f"Dataset has {len(dataset)} samples")

    # Build world model only (decoder not needed)
    enc_cfg = model_cfg.get("enc", {})
    probe_cfg = model_cfg.get("probe", {})
    world_model_cfg = model_cfg.get("world_model", {})
    world_model, _ = compile_model(
        enc_cfg=enc_cfg,
        probe_cfg=probe_cfg,
        world_model_cfg=world_model_cfg,
        device=device,
    )
    world_model.eval()

    # Wrap dataset to return index alongside data
    indexed_dataset = _IndexedDataset(dataset)
    loader = torch.utils.data.DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_skip_none_collate,
    )

    # Cache latents
    manifest = {}
    cached_count = 0

    logger.INFO(f"Caching latents to {cache_dir} ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Caching latents"):
            if batch is None:
                continue
            global_indices, clips, gt_dict = batch

            clips = clips.to(device)
            with torch.amp.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                a_latent, h_goal_pool = world_model(clips, return_goal_pool=True)

            # Save as float32 for maximum compatibility
            a_latent_np = a_latent.float().cpu().numpy()
            h_goal_pool_np = h_goal_pool.float().cpu().numpy()
            wp_np = gt_dict["midlane_wp"].numpy()
            gate_np = gt_dict["gate_score"].numpy()

            for i, gidx in enumerate(global_indices.tolist()):
                out_path = os.path.join(cache_dir, f"{gidx:08d}.npz")
                np.savez_compressed(
                    out_path,
                    a_latent=a_latent_np[i],
                    h_goal_pool=h_goal_pool_np[i],
                    midlane_wp=wp_np[i],
                    gate_score=gate_np[i],
                )
                manifest[str(gidx)] = dataset.samples[gidx]
                cached_count += 1

    # Save manifest
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest["__total__"] = len(dataset)
    manifest["__cached__"] = cached_count
    manifest["__config__"] = args.fname
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.INFO(f"Done. Cached {cached_count}/{len(dataset)} samples to {cache_dir}")
    logger.INFO(f"Manifest saved to {manifest_path}")


def _skip_none_collate(batch):
    """Collate that filters out None samples (from failed loads)."""
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    indices = torch.stack([item[0] for item in valid])
    clips = torch.stack([item[1] for item in valid])
    gt_keys = valid[0][2].keys()
    gt = {k: torch.stack([item[2][k] for item in valid]) for k in gt_keys}
    return indices, clips, gt


class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return (global_index, *original_return) tuples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = self.dataset[index]
        if result is None:
            return None
        clip, gt = result
        return torch.tensor(index, dtype=torch.long), clip, gt


if __name__ == "__main__":
    main()
