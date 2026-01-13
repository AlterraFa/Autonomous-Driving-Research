import os
import re
import numpy as np
import torch
import cv2


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

class CarlaDatasetLoader:
    def __init__(self, dataset_dir: str | list[str], fraction: float = 1.0, ram_caching = True, shuffle = True, pad_value = -10000):
        self.log = Logger()
        self.ram_cache = ram_caching
        self.pad_value = pad_value

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

        if ram_caching: self._cache()

    def _cache(self):
        self.log.INFO(f"Loading metadata for {len(self.samples_dir)} samples into RAM... (This speeds up training)")
        self.samples = []
        for meta_path in tqdm(self.samples_dir, desc = "Ram Caching"):
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
                
                del meta['img_file']
                self.samples.append(meta)
            except Exception as e:
                self.log.WARNING(f"Skipping corrupt metadata {meta_path}: {e}")

        self.log.CUSTOM("SUCCESS", f"Successfully cached {len(self.samples)} samples.")
        
    def __len__(self):
        return len(self.samples) if self.ram_cache else len(self.samples_dir)
    
    def read_image(self, img_path: Path, index = None):
        # # -- Read using opencv first
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image.ndim == 2:
            image = image[..., None]

        # image = decode_image(str(img_path))

        return image

    def _get_samples(self, idx):
        if self.ram_cache:
            meta = self.samples[idx]
            try:
                if meta['is_multi']:
                    inp_images = {}
                    # paths are already absolute strings
                    for index, (key_name, img_path) in enumerate(meta['abs_img_paths'].items()):
                        inp_images[key_name] = self.read_image(img_path, index)
                    return inp_images, meta['control']
                else:
                    image = self.read_image(meta['abs_img_paths'])
                    return image, meta['control']
            except Exception as e:
                print(f"[Error] Failed loading sample {idx}: {e}. Retrying index 0.")
                return self.__getitem__(0)
        
        else: 
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
                return image, meta['control']
            else:
                inp_images = {}
                for index, key_name in enumerate(imgs_path.keys()):
                    img_file = dataset_base / imgs_path[key_name]
                    image    = self.read_image(str(img_file), index)
                    inp_images.update({key_name: image})
                return inp_images, meta['control']
    
    def __getitem__(self, idx):
        return self._get_samples(idx)

    CONTROL_KEYS = ["exp_wp", "midlane_wp", "aux_wp", "steer", "throttle", "brake", "velocity", "turn_signal"]

    def collate_fn(self, batch):
        images, controls = zip(*batch)

        # -- We are using v2 transform so permute beforehand is needed
        batched_images = torch.utils.data.default_collate(images)
        batched_images = {name: value.permute(0, 3, 1, 2) for name, value in batched_images.items()}
        
        # -- auxiliary waypoint processing for gmm head
        aux_list = [torch.tensor(c["aux_wp"], dtype=torch.float32) for c in controls]
        aux_padded = pad_sequence(aux_list, batch_first=True, padding_value = self.pad_value)
        

        # -- other control metadata are batched as usual
        clean_controls = [{k: torch.as_tensor(v, dtype = torch.float32) for k, v in c.items() if k != "aux_wp"} for c in controls]
        batched_controls = torch.utils.data.default_collate(clean_controls)
        
        # -- add back the aux waypoint
        batched_controls["aux_wp"] = aux_padded

        return batched_images, batched_controls

    def split(self, train = 0.9, val = 0.1):
        
        n_total = self.__len__()
        n_train = int(n_total * train)
        n_val   = int(n_total * val)
        n_test  = n_total - n_train - n_val

        return random_split(self, [n_train, n_val, n_test])


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