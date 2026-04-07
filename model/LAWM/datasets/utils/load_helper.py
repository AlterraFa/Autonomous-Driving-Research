import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from pathlib import Path

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

def _ensure_list(value, length):
    """Convert single value to list of given length"""
    if not isinstance(value, (list, tuple)):
        return [value] * length
    return value

def _load_samples_and_labels(data_paths):
    """Load samples and labels from CSV files"""
    samples, labels = [], []
    nsamples_per_dataset = []
    
    for data_path in data_paths:
        df = pd.read_csv(data_path, delimiter=",")            
        samples.extend(df.values[:, 1])
        labels.extend(df.values[:, 0])
        nsamples_per_dataset.append(len(df))
    
    # Create mapping from sample index to dataset index
    mapping = []
    for idx, nsamples in enumerate(nsamples_per_dataset):
        mapping.extend([idx] * nsamples)
    
    return samples, labels, mapping


def _extract_metadata(meta_paths, meta_keys, aggregation = "mean"):
    """Extract ground truth metadata from .npy files"""
    from .decode import _find_metadata_values
    extracted_data = []
    for window_paths in meta_paths:
        window_values = {key: [] for key in meta_keys}
        valid_indices = []
        for i, path in enumerate(window_paths):
            try:
                metadata = np.load(path, allow_pickle=True)
                gt_values = _find_metadata_values(metadata, meta_keys)
                found_any = False
                for key in meta_keys:
                    val = gt_values.get(key)
                    if val is not None:
                        window_values[key].append(float(val))
                        found_any = True
                if found_any:
                    valid_indices.append(i)
            except Exception as e:
                continue
    agg_result = {}
    num_expected = len(window_paths)
    
    for key in meta_keys:
        vals = window_values[key]
        if not vals:
            agg_result[key] = 0.0
            continue
        if aggregation == "mean":
            agg_result[key] = np.mean(vals)
        elif aggregation == "sequence":
            if len(vals) < num_expected:
                x = np.array(valid_indices)
                f = interp1d(x, vals, kind = "linear", fill_value = "extrapolate")
                agg_result[key] = f(np.arange(num_expected))
            else:
                agg_result[key] = np.array(vals)
        elif aggregation == "interpolate":
                # Returns the 'ideal' midpoint value or a smoothed estimate
                # Useful if you want the value exactly at the center of the frame skip
                if len(vals) > 1:
                    x = np.array(valid_indices)
                    f = interp1d(x, vals, kind='linear', fill_value="extrapolate")
                    # Sample at the mathematical center of the skip
                    agg_result[key] = float(f((num_expected - 1) / 2))
                else:
                    agg_result[key] = vals[0]

        elif aggregation == "last":
            agg_result[key] = vals[-1]
        elif aggregation == "first":
            agg_result[key] = vals[0]
        extracted_data += [agg_result]
    return extracted_data

def _check_structure(root_path):
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