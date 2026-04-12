import os
import re
import numpy as np
import torch
import cv2
from rich import print

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '../..', '.env'))
    if 'PYTHONPATH' in os.environ:
        sys.path.insert(0, os.environ['PYTHONPATH'])


from PIL import Image
from turbojpeg import TurboJPEG
from tqdm import tqdm
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

from utils.messages.logger import Logger
from pathlib import Path

jpeg_loader = TurboJPEG("/usr/lib/libturbojpeg.so.0")

def decode_image(path):
    # -- All of this is RGB
    try:
        if path.lower().endswith(('.jpg', '.jpeg')):
            with open(path, "rb") as f:
                img_bytes = f.read()

            return jpeg_loader.decode(img_bytes, pixel_format = 0)
        else: 
            with Image.open(path) as img:
                return np.array(img.convert('RGB'))

    except Exception as e:
        print(f"Error processing {path=}: {e}")
        return None

class PolySystemLoader:
    
    meta_name = "metadata"
    
    def __init__(self, dataset_dir: str | list[str], fraction: float = 1.0, ram_caching = True, shuffle = True):
        self.log = Logger()
        self.ram_cache = ram_caching

        # Handle both single path and list of paths
        if isinstance(dataset_dir, (str, Path)):
            dataset_dir = [dataset_dir]
        
        self.dataset_dirs = [Path(p) for p in dataset_dir]

        self.samples_dir = []; failed = 0
        for d in self.dataset_dirs:
            img_dir = d / "images"
            meta_dir = d / "metadata"
            if not img_dir.exists() or not meta_dir.exists():
                failed += 1
                self.log.ERROR(f"Missing 'images/' or 'metadata/' in {d}.")
            metas = list(meta_dir.glob("*.npy"))
            self.samples_dir.extend(metas)
            
        if failed == len(self.dataset_dirs):
            self.log.ERROR(f"All datasets folder does not have valid structure", exit_code = -1)
        

        self.samples_dir = np.array(self.samples_dir)
        num_samples = len(self.samples_dir)

        if fraction < 1.0:
            load_size = int(fraction * num_samples)
            idx = np.random.choice(num_samples, load_size, replace=False) if shuffle else np.arange(load_size)
            self.samples_dir = self.samples_dir[idx]

        self.log.INFO(f"Found {num_samples} samples across {len(self.dataset_dirs)} dataset(s). Using {len(self.samples_dir)} samples.")

        self._preload()

    def _preload(self):
        self.log.INFO(f"Preloading metadata for {len(self.samples_dir)} samples")
        self.samples = [] 
        self.multimodal_idx = []; self.unimodal_idx = []
        for idx, meta_path in enumerate(tqdm(self.samples_dir, desc = "Preloading")):
            try:
                meta = np.load(meta_path, allow_pickle=True).item()
                
                dataset_base = Path(meta_path).parent.parent
                img_path_raw = meta['img_file']
                
                if isinstance(img_path_raw, dict):
                    meta['is_multi'] = True
                    meta['abs_img_paths'] = {k: str(dataset_base / v) for k, v in img_path_raw.items()}
                else:
                    meta['is_multi'] = False
                    meta['abs_img_paths'] = str(dataset_base / img_path_raw)

                if meta['metadata']['condition']['road_type'] == 'multi':
                    self.multimodal_idx.append(idx)
                elif meta['metadata']['condition']['road_type'] == 'uni':
                    self.unimodal_idx.append(idx)
                
                del meta['img_file']
                if self.ram_cache:
                    meta[self.meta_name] = _flatten_meta(meta[self.meta_name])
                    self.samples.append(meta)
            except Exception as e:
                self.log.WARNING(f"Skipping corrupt metadata {meta_path}: {e}")

        if self.ram_cache:
            self.log.CUSTOM("SUCCESS", f"Successfully cached {len(self.samples)} samples.")
        self.log.WARNING(f"{len(self.multimodal_idx) / (len(self.multimodal_idx) + len(self.unimodal_idx)) * 100: .2f}% of data is at fork on the road. Heavily unbalanced")
        
    def __len__(self):
        return len(self.samples) if self.ram_cache else len(self.samples_dir)
    
    def read_image(self, img_path: Path):
        # # -- Read using opencv first
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image.ndim == 2:
            image = image[..., None]

        return image

    def __getitem__(self, idx):
        if self.ram_cache:
            return self._load_cache(idx)
        else: 
            return self._load_non_cache(idx)

    def _load_non_cache(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index out of range")

        meta_file = Path(self.samples_dir[idx])
        meta = np.load(meta_file, allow_pickle=True).item()

        # -- Determine the base dataset directory from the metadata file path
        # -- meta_file structure: /path/to/dataset/metadata/xxxxx.npy
        dataset_base = meta_file.parent.parent

        imgs_path = meta['img_file']

        if not isinstance(imgs_path, dict):
            img_file = dataset_base / imgs_path
            image    = self.read_image(str(img_file))
            return image, meta[self.meta_name]
        else:
            inp_images = {}
            for index, key_name in enumerate(imgs_path.keys()):
                img_file = dataset_base / imgs_path[key_name]
                image    = self.read_image(str(img_file))
                inp_images.update({key_name: image})
            return inp_images, meta[self.meta_name]
    
    def _load_cache(self, idx):
        meta = self.samples[idx]
        try:
            
            if meta['is_multi']:
                inp_images = {}
                # paths are already absolute strings
                for index, (key_name, img_path) in enumerate(meta['abs_img_paths'].items()):
                    inp_images[key_name] = self.read_image(img_path)
                return inp_images, meta[self.meta_name]
            else:
                image = self.read_image(meta['abs_img_paths'])
                return image, meta[self.meta_name]
        except Exception as e:
            print(f"[Error] Failed loading sample {idx}: {e}. Retrying index 0.")
            return self.__getitem__(0)
    

def _flatten_meta(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        
        # 1. Handle nested dictionaries
        if isinstance(v, dict):
            items.extend(_flatten_meta(v, new_key, sep=sep).items())
        
        elif isinstance(v, (np.generic, np.ndarray)):
            if v.size == 1 and not v.shape:
                items.append((new_key, v.item()))
            else:
                items.append((new_key, v))
                
        else:
            items.append((new_key, v))
            
    return dict(items)

def get_next_run(FILE_DIR: str) -> int:
    exp_dir = os.path.join(FILE_DIR, "Experiment")  # anchor to project root
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        return 1

    runs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    run_nums = []
    for r in runs:
        match = re.match(r"run(\d+)", r)
        if match:
            run_nums.append(int(match.group(1)))

    return max(run_nums, default=0) + 1

    
if __name__ == "__main__":
    from .sampler import BalancedRoadSampler
    from .image_transform import AugmentContext, compose_augment, compose_component
    from .collator import RoadCollator, ImageType, MetaType
    from torch.utils.data import DataLoader
    
    ctx = AugmentContext(flip_prob = 0.5)
    augment = compose_component("horizontal_flip", ctx)
    
    dataset = PolySystemLoader(
        [
            "./data/PolySystemV1/recording_20251019_161905_best_spatial",
            "./data/PolySystemV1/recording_20251025_142727_best_spatial",
            "./data/PolySystemV1/recording_20251029_140108_extra_spatial",
            "./data/PolySystemV1/recording_20251029_163431_extra_spatial",
            "./data/PolySystemV1/recording_20260116_213318_extra_spatial",
        ],
        fraction = 0.1,
        ram_caching = True
    )
    sampler = BalancedRoadSampler(dataset, 32)
    collator = RoadCollator(
        image_types = [ImageType.Mask, ImageType.I0], 
        meta_types = [MetaType.Polyline, MetaType.Waypoint, MetaType.TurnSignal],
        transform = augment
    )

    loader = DataLoader(dataset, batch_sampler = sampler, collate_fn = collator)
    
    print(next(iter(loader))[0]["I0"].shape)
