import os, sys

script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
import pandas as pd
import numpy as np
import glob
import torch
import re
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import Subset
from .utils.decode import decode_batch
from .utils.load_helper import (
    _calculate_frame_indices, 
    _ensure_list, 
    _load_samples_and_labels,
    _extract_metadata,
    _check_structure
)




class VideoDataset(Dataset):
    """Dataset for loading video sequences without action conditioning"""
    
    def __init__(
        self,
        data_paths,
        dataset_fpc=None, 
        frames_per_clips=16,
        nclips=1,
        frame_step=1,
        shared_transform=None,
        individual_transform=None,
        allow_clip_overlap=False,  
        random_jiggle_part=True,
        random_part=True,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.frames_per_clips = frames_per_clips
        self.nclips = nclips
        self.frame_step = frame_step
        self.allow_clip_overlap = allow_clip_overlap
        self.random_jiggle_part = random_jiggle_part
        self.random_part = random_part
        self.individual_transform = individual_transform
        self.shared_transform = shared_transform
        
        # Set up dataset-specific frames per clip
        if dataset_fpc is None:
            self.datasets_fpc = [frames_per_clips] * len(data_paths)
        else:
            self.datasets_fpc = dataset_fpc
            
        # Load data from CSV files
        self.samples, self.labels, self.video_indices_map = _load_samples_and_labels(data_paths)
        self.stats_cache = None
        self.apply_gt_transform = False
    
    def __getitem__(self, index):
        """Load sample with retry logic"""
        for retry in range(5):
            try:
                sample = self.load_image_sequences(index)
                if sample is not None:
                    return sample
            except Exception as e:
                if retry < 4:
                    print(f"Error loading sample at {index=}, retrying ({retry+1}/5): {e}")

        print(f"Failed to load sample at {index=}, dataset_path={self.samples[index]} after 5 retries")

    def load_image_sequences(self, index):
        """Load and process image sequences for given index"""
        sample = self.samples[index]
        dataset_idx = self.video_indices_map[index]
        fpc = self.datasets_fpc[dataset_idx]
        
        # Get image paths
        image_paths = glob.glob(os.path.join(sample, "*"))
        image_paths = sorted(image_paths, key=lambda x: int(re.findall(r'\d+', x.rsplit('.', 1)[0])[-1]))
        
        if not image_paths:
            return None
        
        # Calculate frame indices using helper function
        buffer_indices, clip_indices = _calculate_frame_indices(
            len(image_paths), fpc, self.nclips, self.frame_step,
            self.allow_clip_overlap, self.random_jiggle_part
        )
        
        # Load frames
        buffer = decode_batch(np.array(image_paths)[buffer_indices])
        if len(buffer) == 0:
            return None
        
        # Apply transforms
        if self.shared_transform is not None:
            self.shared_transform(buffer)
        
        # Reshape buffer by clips
        clips = []
        for idx, indices in enumerate(clip_indices):
            start_idx = idx * len(indices)
            end_idx = (idx + 1) * len(indices)
            clip = buffer[start_idx:end_idx]
            
            if self.individual_transform is not None:
                clip = self.individual_transform(clip)
            clips.append(clip)
        
        return clips, clip_indices
    
    def __len__(self):
        return len(self.samples)

    def split(self, train=0.9, val=0.1):
        """Split each source CSV independently into train/val/test sets"""
        train_indices = []
        val_indices = []
        test_indices = []

        # Group sample indices by originating CSV/dataset
        indices_by_dataset = {}
        for sample_idx, dataset_idx in enumerate(self.video_indices_map):
            if dataset_idx not in indices_by_dataset:
                indices_by_dataset[dataset_idx] = []
            indices_by_dataset[dataset_idx].append(sample_idx)

        # Split each CSV independently, then merge
        for dataset_idx in sorted(indices_by_dataset.keys()):
            dataset_indices = np.array(indices_by_dataset[dataset_idx], dtype=np.int64)
            if len(dataset_indices) == 0:
                continue

            shuffled = np.random.permutation(dataset_indices)
            n_total = len(shuffled)
            n_train = int(n_total * train)
            n_val = int(n_total * val)

            train_indices.extend(shuffled[:n_train].tolist())
            val_indices.extend(shuffled[n_train:n_train + n_val].tolist())
            test_indices.extend(shuffled[n_train + n_val:].tolist())

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )
    
class ActVideoDataset(Dataset):
    """Action-conditioned video dataset with context/prediction separation and metadata"""
    
    def __init__(
        self,
        data_paths,
        ctx_frames_per_clips=16,
        pred_frames_per_clips=8,
        nclips=1,
        frame_step=1,
        shared_transform=None,
        individual_transform=None,
        allow_clip_overlap=False,  
        random_jiggle_part=True,
        random_part=True,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.nclips = nclips
        self.allow_clip_overlap = allow_clip_overlap
        self.random_jiggle_part = random_jiggle_part
        self.random_part = random_part
        self.individual_transform = individual_transform
        self.shared_transform = shared_transform
        
        # Handle context and prediction frames per clip
        self.ctx_fpcs = _ensure_list(ctx_frames_per_clips, len(data_paths))
        self.pred_fpcs = _ensure_list(pred_frames_per_clips, len(data_paths))
        self.frame_step = _ensure_list(frame_step, len(data_paths))
        
        if len(self.ctx_fpcs) != len(self.pred_fpcs):
            raise ValueError("Number of context fpcs must match prediction fpcs")
        
        self.datasets_fpc = [ctx + pred for ctx, pred in zip(self.ctx_fpcs, self.pred_fpcs)]
        
        # Load data from CSV files
        self.samples, self.labels, self.video_indices_map = _load_samples_and_labels(data_paths)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Load sample with retry logic"""
        for retry in range(5):
            try:
                sample = self.load_image_sequences(index)
                if sample is not None:
                    return sample
            except Exception as e:
                if retry < 4:
                    print(f"Error loading sample at {index=}, retrying ({retry+1}/5): {e}")
                else:
                    print(f"Failed to load sample at {index=} after 5 retries")
        return None

    def load_image_sequences(self, index):
        """Load image sequences with actions and metadata"""
        sample = self.samples[index]
        dataset_idx = self.video_indices_map[index]
        fpc = self.datasets_fpc[dataset_idx]
        ctx_fpcs = self.ctx_fpcs[dataset_idx]
        fps = self.frame_step[dataset_idx]
        
        # Check structure and get metadata paths
        metadata_paths = _check_structure(sample)
        if not metadata_paths:
            return None
        
        # Get metadata file paths
        meta_paths = glob.glob(os.path.join(metadata_paths, "*"))
        meta_paths = sorted(meta_paths, key=lambda x: int(re.findall(r'\d+', x.rsplit('.', 1)[0])[-1]))
        
        if not meta_paths:
            return None
        
        # Calculate frame indices using helper function
        buffer_indices, clip_indices = _calculate_frame_indices(
            len(meta_paths), fpc, self.nclips, fps,
            self.allow_clip_overlap, self.random_jiggle_part
        )
        
        # Load frames and metadata
        selected_paths = np.array(meta_paths)[buffer_indices]
        buffer = decode_batch(selected_paths)
        gt_data = _extract_metadata(selected_paths, ("steer", "velocity"))
        
        if len(buffer) == 0:
            return None
        
        # Apply shared transforms
        if self.shared_transform is not None:
            self.shared_transform(buffer)
        
        # Process clips and separate context/prediction frames
        ctx_buffers, pred_buffers, gt_clips = [], [], []
        
        for idx, indices in enumerate(clip_indices):
            start_idx = idx * len(indices)
            end_idx = (idx + 1) * len(indices)
            
            # Extract clip and ground truth data
            clip = buffer[start_idx:end_idx]
            gt_clip = gt_data[start_idx:end_idx]
            
            # Split into context and prediction frames
            ctx_clip = clip[:ctx_fpcs]
            pred_clip = clip[ctx_fpcs:]
            
            # Apply individual transforms
            if self.individual_transform is not None:
                ctx_clip = self.individual_transform(ctx_clip)
                pred_clip = self.individual_transform(pred_clip)
            
            ctx_buffers.append(ctx_clip)
            pred_buffers.append(pred_clip)
            gt_clips.append(torch.utils.data.default_collate(gt_clip))
        
        return ctx_buffers, pred_buffers, gt_clips
    


class ProbeDataset(Dataset):
    """Action-conditioned video dataset with context/prediction separation and metadata"""
    
    def __init__(
        self,
        data_paths,
        frames_per_clips=16,
        nclips=1,
        frame_step=1,
        shared_transform=None,
        individual_transform=None,
        allow_clip_overlap=False,  
        random_jiggle_part=True,
        random_part=True,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.nclips = nclips
        self.allow_clip_overlap = allow_clip_overlap
        self.random_jiggle_part = random_jiggle_part
        self.random_part = random_part
        self.individual_transform = individual_transform
        self.shared_transform = shared_transform
        
        # Handle context and prediction frames per clip
        self.datasets_fpc = _ensure_list(frames_per_clips, len(data_paths))
        self.frame_step = _ensure_list(frame_step, len(data_paths))
        
        # Load data from CSV files
        self.samples, self.labels, self.video_indices_map = _load_samples_and_labels(data_paths)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """Load sample with retry logic"""
        for retry in range(5):
            try:
                sample = self.load_image_sequences(index)
                if sample is not None:
                    return sample
            except Exception as e:
                if retry < 4:
                    print(f"Error loading sample at {index=}, retrying ({retry+1}/5): {e}")
                else:
                    print(f"Failed to load sample at {index=} after 5 retries")
        return None

    def load_image_sequences(self, index):
        """Load image sequences with actions and metadata"""
        sample = self.samples[index]
        dataset_idx = self.video_indices_map[index]
        fpc = self.datasets_fpc[dataset_idx]
        fps = self.frame_step[dataset_idx]
        
        # Check structure and get metadata paths
        metadata_paths = _check_structure(sample)
        if not metadata_paths:
            return None
        
        # Get metadata file paths
        meta_paths = glob.glob(os.path.join(metadata_paths, "*"))
        meta_paths = sorted(meta_paths, key=lambda x: int(re.findall(r'\d+', x.rsplit('.', 1)[0])[-1]))
        
        if not meta_paths:
            return None
        
        # Calculate frame indices using helper function
        buffer_indices, clip_indices = _calculate_frame_indices(
            len(meta_paths), fpc, self.nclips, fps,
            self.allow_clip_overlap, self.random_jiggle_part
        )
        
        # Load frames and metadata
        selected_paths = np.array(meta_paths)[buffer_indices]
        buffer = decode_batch(selected_paths)
        gt_data = _extract_metadata(selected_paths, ("steer", "velocity", "lat_err"))

        if hasattr(self, 'stats_cache'):
            gt_data = [self._transform_gt_values(frame_gt) for frame_gt in gt_data]
        
        if len(buffer) == 0:
            return None
        
        # Apply shared transforms
        if self.shared_transform is not None:
            self.shared_transform(buffer)
        
        # Process clips and separate context/prediction frames
        clip_buffers, gt_clips = [], []
        
        for idx, indices in enumerate(clip_indices):
            start_idx = idx * len(indices)
            end_idx = (idx + 1) * len(indices)
            
            # Extract clip and ground truth data
            clip = buffer[start_idx:end_idx]
            gt_clip = gt_data[start_idx:end_idx]
            
            # Apply individual transforms
            if self.individual_transform is not None:
                clip = self.individual_transform(clip)
            
            clip_buffers.append(clip)
            gt_clips.append(torch.utils.data.default_collate(gt_clip))
        
        return clip_buffers, gt_clips

    def _transform_gt_values(self, frame_gt):
        if not isinstance(frame_gt, dict):
            return frame_gt

        transformed = dict(frame_gt)
        for key, stats in self.stats_cache.items():
            if key not in transformed:
                continue
            value = transformed[key]
            if value is None:
                continue
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is None or std is None or std == 0:
                continue
            try:
                transformed[key] = (float(value) - mean) / std
            except (TypeError, ValueError):
                continue

        return transformed

    def split(self, train = 0.9, val = 0.1):
        train_indices = []
        val_indices = []
        test_indices = []

        # Group sample indices by originating CSV/dataset
        indices_by_dataset = {}
        for sample_idx, dataset_idx in enumerate(self.video_indices_map):
            if dataset_idx not in indices_by_dataset:
                indices_by_dataset[dataset_idx] = []
            indices_by_dataset[dataset_idx].append(sample_idx)

        # Split each CSV independently, then merge
        for dataset_idx in sorted(indices_by_dataset.keys()):
            dataset_indices = np.array(indices_by_dataset[dataset_idx], dtype=np.int64)
            if len(dataset_indices) == 0:
                continue

            shuffled = np.random.permutation(dataset_indices)
            n_total = len(shuffled)
            n_train = int(n_total * train)
            n_val = int(n_total * val)

            train_indices.extend(shuffled[:n_train].tolist())
            val_indices.extend(shuffled[n_train:n_train + n_val].tolist())
            test_indices.extend(shuffled[n_train + n_val:].tolist())

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )

    def statistics(self, gt_types=("steer", "velocity", "lat_err"), unbiased=False, indices=None):
        """Compute per-ground-truth mean and variance across all samples.

        Args:
            gt_types: Iterable of metadata keys to aggregate.
            unbiased: If True, variance uses N-1 denominator (sample variance).
            indices: Optional iterable of sample indices to restrict aggregation.

        Returns:
            Dict[str, Dict[str, float | int | None]] with count, mean, variance, std.
        """
        gt_types = tuple(gt_types)
        running = {
            key: {"count": 0, "mean": 0.0, "m2": 0.0}
            for key in gt_types
        }

        def _meta_sort_key(path):
            stem = path.rsplit('.', 1)[0]
            nums = re.findall(r'\d+', stem)
            if nums:
                return (0, int(nums[-1]))
            return (1, stem)

        if indices is None:
            sample_indices = range(len(self.samples))
        else:
            sample_indices = indices

        for sample_index in sample_indices:
            sample = self.samples[sample_index]
            metadata_paths = _check_structure(sample)
            if not metadata_paths:
                continue

            meta_paths = glob.glob(os.path.join(metadata_paths, "*"))
            meta_paths = sorted(meta_paths, key=_meta_sort_key)
            if not meta_paths:
                continue

            gt_data = _extract_metadata(meta_paths, gt_types)
            for frame_gt in gt_data:
                if not isinstance(frame_gt, dict):
                    continue
                for key in gt_types:
                    value = frame_gt.get(key)
                    if value is None:
                        continue
                    try:
                        x = float(value)
                    except (TypeError, ValueError):
                        continue

                    stat = running[key]
                    stat["count"] += 1
                    delta = x - stat["mean"]
                    stat["mean"] += delta / stat["count"]
                    delta2 = x - stat["mean"]
                    stat["m2"] += delta * delta2

        stats = {}
        for key, stat in running.items():
            count = stat["count"]
            if count == 0:
                stats[key] = {
                    "count": 0,
                    "mean": None,
                    "variance": None,
                    "std": None,
                }
                continue

            denom = (count - 1) if unbiased else count
            variance = stat["m2"] / denom if denom > 0 else 0.0
            stats[key] = {
                "count": count,
                "mean": stat["mean"],
                "variance": variance,
                "std": float(np.sqrt(variance)),
            }

        self.stats_cache = stats
        return stats

if __name__ == "__main__":
    import yaml
    import cv2

    with open("./cfgs/probe/probe-384px-1024.24e.yaml", "r") as f:
        args = yaml.safe_load(f)

    train_arg = args['train']
    
    dset = ProbeDataset(
        data_paths = [dataset['path'] for dataset in train_arg['datasets']],    
        frame_step = [dataset['fps'] for dataset in train_arg['datasets']],
        frames_per_clips = train_arg['fpcs'],
        nclips = 1,
        allow_clip_overlap = train_arg['allow_clip_overlap'],
        random_jiggle_part = train_arg['random_jiggle']
    )
    train, val, _ = dset.split(0.85, 0.15)
    
    stats = val.dataset.statistics(indices=val.indices)
    print("Val Ground-truth statistics:")
    for key, values in stats.items():
        print(
            f"{key}: "
            f"count={values['count']}, "
            f"mean={values['mean']}, "
            f"variance={values['variance']}, "
            f"std={values['std']}"
        )

    stats = train.dataset.statistics(indices=train.indices)
    print("Train Ground-truth statistics:")
    for key, values in stats.items():
        print(
            f"{key}: "
            f"count={values['count']}, "
            f"mean={values['mean']}, "
            f"variance={values['variance']}, "
            f"std={values['std']}"
        )