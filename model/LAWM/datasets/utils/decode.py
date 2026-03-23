import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

try: 
    from turbojpeg import TurboJPEG
    _jpeg_loader = TurboJPEG("/usr/lib/libturbojpeg.so.0")
    _use_turbo = True
except Exception as e:
    _use_turbo = False
    import cv2
_image_ext = ('.jpg', '.jpeg', '.png')


def _find_metadata_values(data, find_meta):
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
    
    recursive_search(data, list(find_meta))
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
 
 
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.cpu_count() // 2))
def decode_batch(paths):
    frames = np.asarray(list(_EXECUTOR.map(_decode, paths)), dtype=np.uint8)
    return frames