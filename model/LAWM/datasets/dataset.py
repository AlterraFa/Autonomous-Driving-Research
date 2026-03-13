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
from PIL import Image
from pathlib import Path
from turbojpeg import TurboJPEG
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import random_split

try: 
    from turbojpeg import TurboJPEG
    _jpeg_loader = TurboJPEG("/usr/lib/libturbojpeg.so0")
    _use_turbo = True
    print(f"Loaded TurboJPEG")
except Exception as e:
    _use_turbo = False
    print(f"Error while loading TurboJPEG, Falling back to opencv: {e}")
    import cv2
_image_ext = ('.jpg', '.jpeg', '.png')
_find_meta = ("steer", "velocity")


def _find_metadata_values(data, target_keys):
    """Recursively search for target keys in nested data structure"""
    found_values = {}
    
    def recursive_search(obj, keys_to_find):
        if not keys_to_find:
            return
            
        if isinstance(obj, np.ndarray):
            # Handle np.ndarray cases
            if obj.ndim == 0:
                recursive_search(obj.item(), keys_to_find)
            elif obj.dtype == object:
                for item in obj:
                    recursive_search(item, keys_to_find)
            return
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Check if this key is one we're looking for
                if key in keys_to_find and key not in found_values:
                    found_values[key] = float(value)
                    keys_to_find = [k for k in keys_to_find if k != key]
                    if not keys_to_find:
                        return
                # Continue searching in nested values
                recursive_search(value, keys_to_find)
        
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                recursive_search(item, keys_to_find)
    
    recursive_search(data, list(target_keys))
    return found_values


def _decode_metadata(path):
    metadata = np.load(path, allow_pickle = True)
    
    def find_image_path(data):
        if isinstance(data, np.ndarray):
            # -- Data is loaded via np.load => Handle np.ndarray cases
            if data.ndim == 0:
                return find_image_path(data.item())
            if data.dtype == object:
                for item in data:
                    res = find_image_path(item)
                    if res is not None: return res
            return None

        if isinstance(data, str):
            if data.lower().endswith(_image_ext):
                return data
            return None

        if isinstance(data, dict):
            for value in data.values():
                result = find_image_path(value)
                if result is not None:
                    return result
            return None

        if isinstance(data, (list, tuple)):
            for item in data:
                result = find_image_path(item)
                if result is not None:
                    return result
            return None

        return None
            
    return find_image_path(metadata)
    
def _decode_image(path):
    if path.lower().endswith(_image_ext[:-1]):
        if _use_turbo:
            # Method 1: TurboJPEG (Fastest)
            with open(path, "rb") as f:
                img_bytes = f.read()
            # pixel_format=0 typically refers to TJPF_RGB in most turbojpeg wrappers
            return _jpeg_loader.decode(img_bytes, pixel_format=0)
        else:
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                with Image.open(path) as img:
                    return np.array(img.convert('RGB'))
    else: 
        with Image.open(path) as img:
            return np.array(img.convert('RGB'))
    
def _decode(path):
    try:
        if path.lower().endswith(('.npz', '.npy')):
            path = os.path.join(
                os.path.dirname(os.path.dirname(path)),
                _decode_metadata(path)
            )

        if path.lower().endswith(_image_ext):
            return _decode_image(path)
    
    except Exception as e:
        print(f"Error processing {path=}: {e}")
        return None
 
    
def decode_batch(paths):
    with ThreadPoolExecutor(max_workers=1) as executor:
        frames = np.asarray(list(executor.map(_decode, paths)), dtype=np.uint8)
    return frames


def _calculate_frame_indices(seq_length, fpc, nclips, frame_step, allow_clip_overlap, random_jiggle_part):
    """Calculate frame indices for video clips - common logic for both dataset types"""
    target_len = int(frame_step * fpc)
    part_len = seq_length // nclips
    
    buffer_indices, clip_indices = [], []
    
    for i in range(nclips):
        if part_len > target_len:
            end_idx = target_len
            if random_jiggle_part:
                end_idx = np.random.randint(target_len, part_len)
            start_idx = end_idx - target_len
            
            local_indices = np.linspace(start_idx, end_idx, fpc, dtype=np.int64)
            local_indices = np.clip(local_indices, start_idx, end_idx - 1)
            global_indices = local_indices + i * part_len
            
        else:
            if not allow_clip_overlap:
                local_indices = np.linspace(0, part_len, num=part_len // frame_step, dtype=np.int64)
                # Pad if needed
                if len(local_indices) < fpc:
                    padding = np.full(fpc - len(local_indices), part_len - 1)
                    local_indices = np.concatenate([local_indices, padding])
                local_indices = np.clip(local_indices, 0, part_len - 1)
                global_indices = local_indices + i * part_len
            else:
                sample_length = min(target_len, seq_length)
                local_indices = np.linspace(0, sample_length, num=sample_length // frame_step, dtype=np.int64)
                local_indices = np.clip(local_indices, 0, sample_length - 1)
                
                step = 0 if seq_length < target_len else (seq_length - target_len) // max(1, nclips - 1)
                global_indices = local_indices + i * step
        
        clip_indices.append(global_indices.tolist())
        buffer_indices.extend(global_indices.tolist())
    
    return buffer_indices, clip_indices

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
        self._load_samples_and_labels()
    
    def _load_samples_and_labels(self):
        """Load samples and labels from CSV files"""
        samples, labels = [], []
        self.nsamples_per_dataset = []
        
        for data_path in self.data_paths:
            df = pd.read_csv(data_path, header=None, delimiter=",")            
            samples.extend(df.values[:, 1])
            labels.extend(df.values[:, 2])
            self.nsamples_per_dataset.append(len(df))
            
        # Create mapping from sample index to dataset index
        self.video_indices_map = []
        for idx, nsamples in enumerate(self.nsamples_per_dataset):
            self.video_indices_map.extend([idx] * nsamples)
            
        self.samples = samples
        self.labels = labels

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
        """Split dataset into train/val/test sets"""
        n_total = len(self)
        n_train = int(n_total * train)
        n_val = int(n_total * val)
        n_test = n_total - n_train - n_val
        return random_split(self, [n_train, n_val, n_test])
    
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
        self.ctx_fpcs = self._ensure_list(ctx_frames_per_clips, len(data_paths))
        self.pred_fpcs = self._ensure_list(pred_frames_per_clips, len(data_paths))
        self.frame_step = self._ensure_list(frame_step, len(data_paths))
        
        if len(self.ctx_fpcs) != len(self.pred_fpcs):
            raise ValueError("Number of context fpcs must match prediction fpcs")
        
        self.datasets_fpc = [ctx + pred for ctx, pred in zip(self.ctx_fpcs, self.pred_fpcs)]
        
        # Load data from CSV files
        self._load_samples_and_labels()
    
    def _ensure_list(self, value, length):
        """Convert single value to list of given length"""
        if not isinstance(value, (list, tuple)):
            return [value] * length
        return value
    
    def _load_samples_and_labels(self):
        """Load samples and labels from CSV files"""
        samples, labels = [], []
        self.nsamples_per_dataset = []
        
        for data_path in self.data_paths:
            df = pd.read_csv(data_path, delimiter=",")            
            samples.extend(df.values[:, 1])
            labels.extend(df.values[:, 0])
            self.nsamples_per_dataset.append(len(df))
        
        # Create mapping from sample index to dataset index
        self.video_indices_map = []
        for idx, nsamples in enumerate(self.nsamples_per_dataset):
            self.video_indices_map.extend([idx] * nsamples)
        
        self.samples = samples
        self.labels = labels

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
        metadata_paths = self._check_structure(sample)
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
        gt_data = self._extract_metadata(selected_paths)
        
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
    
    def _extract_metadata(self, meta_paths):
        """Extract ground truth metadata from .npy files"""
        extracted_data = []
        for path in meta_paths:
            try:
                metadata = np.load(path, allow_pickle=True)
                gt_values = _find_metadata_values(metadata, _find_meta)
                extracted_data.append(gt_values)
            except Exception as e:
                print(f"Error extracting metadata from {path}: {e}")
                extracted_data.append({})
        return extracted_data

    def _check_structure(self, root_path):
        """Check if directory structure contains valid images and metadata files"""
        root = Path(root_path)
        
        # Check for image files
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        has_images = any(f for f in root.rglob("*") if f.suffix.lower() in img_exts)
        
        if not has_images:
            return False

        # Find directory containing .npy files
        first_npy = next(root.rglob("*.npy"), None)
        return str(first_npy.parent) if first_npy else False

    def __len__(self):
        return len(self.samples)