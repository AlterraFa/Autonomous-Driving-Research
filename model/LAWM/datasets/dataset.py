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
import random
from concurrent.futures import ThreadPoolExecutor
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
import traceback


IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm


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

        metadata_paths = _check_structure(sample)
        if not metadata_paths:
            print("Not valid metadata structure", sample)
            return None
        
        # Get image paths
        image_paths = glob.glob(os.path.join(metadata_paths, "*"))
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

        print(f"Failed to load sample at {index=} after 5 retries {e}")
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
    

from typing import Literal
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
        agg_method: Literal['first', 'last', 'interpolate', 'sequence', 'mean'] = "first"
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
        self.agg_method = agg_method
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
        print(f"Failed to load sample at {index=} after 5 retries: {e}")
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
            print("Not valid metadata structure", sample)
            return None
        
        # Get metadata file paths
        meta_paths = glob.glob(os.path.join(metadata_paths, "*"))
        meta_paths = sorted(meta_paths, key=lambda x: int(re.findall(r'\d+', x.rsplit('.', 1)[0])[-1]))
        
        if not meta_paths:
            print("No metadata file found")
            return None
        
        # Calculate frame indices using helper function
        buffer_indices, clip_indices = _calculate_frame_indices(
            len(meta_paths), fpc, self.nclips, fps,
            self.allow_clip_overlap, self.random_jiggle_part
        )
        
        # Load frames and metadata
        selected_paths = np.array(meta_paths)[buffer_indices]
        buffer = decode_batch(selected_paths)
        meta_windows = [
            meta_paths[idx : min(idx + fps, len(meta_paths))] 
            for idx in buffer_indices
        ]
        
        gt_data = _extract_metadata(
            meta_windows, 
            ("steer", "velocity", "lat_err"), 
            aggregation=self.agg_method 
        )

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

    @staticmethod
    def _meta_sort_key(path):
        """Helper to sort metadata files numerically (frame_2 before frame_10)."""
        stem = os.path.basename(path).rsplit('.', 1)[0]
        nums = re.findall(r'\d+', stem)
        if nums:
            return (0, int(nums[-1]))
        return (1, stem)

    def _process_single_sample(self, sample_index, gt_types):
        """Worker function to be run in threads."""
        sample = self.samples[sample_index]
        metadata_root = _check_structure(sample) # Ensure this is accessible
        if not metadata_root:
            return None

        meta_paths = glob.glob(os.path.join(metadata_root, "*"))
        if not meta_paths:
            return None
        
        meta_paths = sorted(meta_paths, key=self._meta_sort_key)
        gt_data = _extract_metadata(meta_paths, gt_types) # Ensure this is accessible

        # Collect all valid values from this specific sample
        local_values = {key: [] for key in gt_types}
        for frame_gt in gt_data:
            if not isinstance(frame_gt, dict):
                continue
            for key in gt_types:
                value = frame_gt.get(key)
                if value is not None:
                    try:
                        local_values[key].append(float(value))
                    except (TypeError, ValueError):
                        continue
        return local_values

    def statistics(self, gt_types=("steer", "velocity", "lat_err"), unbiased=False, indices=None, max_samples=None, num_workers=os.cpu_count()):
        """
        Compute statistics using ThreadPoolExecutor for faster I/O.
        """
        gt_types = tuple(gt_types)
        running = {key: {"count": 0, "mean": 0.0, "m2": 0.0} for key in gt_types}

        if indices is None:
            sample_indices = list(range(len(self.samples)))
        else:
            sample_indices = list(indices)

        if max_samples is not None and len(sample_indices) > max_samples:
            sample_indices = random.sample(sample_indices, max_samples)

        # Use ThreadPoolExecutor to overlap I/O operations
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # We use list(executor.map) wrapped in tqdm to see progress
            # The map ensures we process samples in parallel
            results = list(tqdm(
                executor.map(lambda idx: self._process_single_sample(idx, gt_types), sample_indices),
                total=len(sample_indices),
                desc="Parallel Stats Calculation"
            ))

        # Reduction step: Update the global running stats with results from threads
        for local_values in results:
            if local_values is None:
                continue
                
            for key in gt_types:
                for x in local_values[key]:
                    stat = running[key]
                    stat["count"] += 1
                    delta = x - stat["mean"]
                    stat["mean"] += delta / stat["count"]
                    delta2 = x - stat["mean"]
                    stat["m2"] += delta * delta2

        # Finalize Results
        stats = {}
        for key, stat in running.items():
            count = stat["count"]
            if count == 0:
                stats[key] = {"count": 0, "mean": None, "variance": None, "std": None}
                continue

            denom = (count - 1) if unbiased else count
            variance = stat["m2"] / denom if denom > 0 else 0.0
            stats[key] = {
                "count": count,
                "mean": stat["mean"],
                "variance": variance,
                "std": float(np.sqrt(variance)),
            }
        return stats


class StraighteningDataset(Dataset):
    """Action-conditioned video dataset with context/prediction separation and metadata"""
    
    def __init__(
        self,
        data_paths,
        shared_transform=None,
        individual_transform=None,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.individual_transform = individual_transform
        self.shared_transform = shared_transform

        self._load_samples()
       
    def _load_samples(self):
        self.samples = []
        self.mapping = []
        for idx, path in enumerate(self.data_paths):
            seq_paths = _check_structure(path)
            if not seq_paths:
                raise ValueError("No sequence path found to match the structure", path)
            samples = glob.glob(os.path.join(seq_paths, "*"))
            self.samples += samples    
            self.mapping.extend([idx] * len(samples))
            
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
        print(f"Failed to load sample at {index=} after 5 retries: {e}")
        return None

    def load_image_sequences(self, index):
        """Load image sequences with actions and metadata"""
        sample   = self.samples[index]
        abs_path = self.data_paths[self.mapping[index]]
        
        buffer = self._load_seq(abs_path, sample)

        if self.individual_transform is not None:
            buffer = np.array([self.individual_transform(image) for image in buffer])
        
        # Apply shared transforms
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        
        return buffer
    
    def _load_seq(self, abs_path, path):
        data = np.load(path, allow_pickle = True).item()
        
        img_dict = data['img_file']
        image_paths = [os.path.join(abs_path, value) for value in img_dict.values()]

        return decode_batch(image_paths)
        
if __name__ == "__main__":
    import yaml
    import cv2
    from augmenter.transforms_builder import VideoTransform
    from torch.utils.data import DataLoader

    
    transform = VideoTransform(
        random_horizontal_flip = False,
        reprob = 0.1,
        random_resize_aspect_ratio = (0.75, 4/3),
        random_resize_scale = (0.7, 1.2),
        auto_augment = True,
        motion_shift = True,
        normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )

    dataset = StraighteningDataset(
        data_paths = [
            "./../Autonomous_Dataset/carla/LAWM/recording_20251025_142727_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260204_010805_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260308_212005_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260317_214033_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260317_233603_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260318_083409_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260323_200940_best_spatial/",
            "./../Autonomous_Dataset/carla/LAWM/recording_20260329_233141_best_spatial/",
        ],
        shared_transform = transform
    )
    
    
    DataLoader()