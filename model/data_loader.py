import os
import re
import numpy as np
import torch
import cv2
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

from .logger import Logger
from pathlib import Path

class CarlaDatasetLoader:
    def __init__(self, dataset_dir: str | list[str], images_key: list[str] = None, downsize_ratio = 1, load_size: int = -1, shuffle = True):
        self.log = Logger()
        self.images_key = images_key
        self.downsize_ratio = downsize_ratio

        if isinstance(dataset_dir, (str, Path)):
            dataset_dir = [dataset_dir]

        self.dataset_dirs = [Path(p) for p in dataset_dir]

        for d in self.dataset_dirs:
            img_dir = d / "images"
            meta_dir = d / "metadata"
            if not img_dir.exists() or not meta_dir.exists():
                raise FileNotFoundError(f"Missing 'images/' or 'metadata/' in {d}.")
        
        self.samples_dir = []
        for d in self.dataset_dirs:
            metas = list((d / "metadata").glob("*.npy"))
            self.samples_dir.extend(metas)

        self.samples_dir = np.array(self.samples_dir)
        num_samples = len(self.samples_dir)

        if load_size != -1 and load_size < num_samples:
            idx = np.random.choice(num_samples, load_size, replace=False) if shuffle else np.arange(load_size)
            self.samples_dir = self.samples_dir[idx]

        self.log.INFO(f"Found {num_samples} samples across {len(self.dataset_dirs)} dataset(s). Using {len(self.samples_dir)} samples.")

    def __len__(self):
        return len(self.samples_dir)
    
    def read_image(self, img_path: Path):
        """Reads image as RGB or grayscale, ensures shape (..., 1 or 3)."""
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        if len(image.shape) == 2:
            image = image[..., None]

        if self.downsize_ratio != 1:
            H, W = image.shape[:2]
            image = cv2.resize(image, (W // self.downsize_ratio, H // self.downsize_ratio))
            if image.ndim == 2:
                image = image[..., None]

        return image

    def _get_samples(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index out of range")

        meta_file = Path(self.samples_dir[idx])
        meta = np.load(meta_file, allow_pickle=True).item()

        # Determine the base dataset directory from the metadata file path
        # meta_file structure: /path/to/dataset/metadata/xxxxx.npy
        dataset_base = meta_file.parent.parent

        if self.images_key is None:
            img_file = dataset_base / meta["img_file"]
            image = self.read_image(str(img_file))
            if self.downsize_ratio != 1:
                H, W, _ = image.shape
                image = cv2.resize(image, (W // self.downsize_ratio, H // self.downsize_ratio))

            return {
                "image": image,
                "control": meta["control"],
                "timestamp": meta["timestamp"],
            }
        else:
            inp_images = {}
            for key_name in self.images_key:
                img_file = dataset_base / meta["img_file"][key_name]
                image = self.read_image(str(img_file))
                if self.downsize_ratio != 1:
                    H, W, _ = image.shape
                    image = cv2.resize(image, (W // self.downsize_ratio, H // self.downsize_ratio))
                inp_images.update({key_name: image})

            return {
                "images": inp_images,
                "control": meta["control"],
                "timestamp": meta["timestamp"],
            }
    
    def __getitem__(self, idx):
        return self._get_samples(idx)

    CONTROL_KEYS = ["exp_wp", "midlane_wp", "aux_wp", "steer", "throttle", "brake", "velocity", "turn_signal"]

    @classmethod
    def collate_fn(cls, batch):
        first_item = batch[0]["images"]

        if isinstance(first_item, np.ndarray):
            images = torch.stack([
                torch.from_numpy(np.ascontiguousarray(data["images"])) for data in batch
            ]).permute(0, 3, 1, 2) / 255.0 # Normalize 

        elif isinstance(first_item, dict):
            images = {}
            input_keys = list(first_item.keys())

            for key_name in input_keys:
                imgs = [
                    torch.from_numpy(np.ascontiguousarray(data["images"][key_name])) for data in batch
                ]
                imgs = torch.stack(imgs).permute(0, 3, 1, 2) / 255.0
                images[key_name] = imgs
        else:
            raise TypeError(f"Unsupported image type in batch: {type(first_item)}")

        controls = {}
        for key in cls.CONTROL_KEYS:
            if key == "aux_wp":
                aux_list = [torch.tensor(data["control"][key], dtype=torch.float32) for data in batch]
                aux_padded = pad_sequence(aux_list, batch_first = True) # Batch, max_branch, num_wp, 2
                controls[key] = aux_padded
            else:
                ctrl_arr = np.array([data["control"][key] for data in batch])
                controls[key] = torch.from_numpy(ctrl_arr).float()

        return images, controls

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