import cv2
import time
import torch
import numpy as np
import threading, queue

from torch.utils.data import random_split
from rich import print
from pathlib import Path
from typing import Dict, Any

class TrajectoryBuffer:
    def __init__(self, save_dir: str, init_cap = 8192, dist_thresh_m = 0, min_dt_s = 0.05):
        self.arr = np.empty((init_cap, 4), dtype=np.float32)
        self.n = 0
        self.last = None
        self.last_t = 0.0
        self.dist_thresh = float(dist_thresh_m)
        self.min_dt = float(min_dt_s)
        self.save_dir = save_dir

    @staticmethod
    def _dist3(a, b):
        dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    def update(self, loc: np.ndarray) -> None:
        t = time.time()
        p = [loc[0], loc[1], loc[2]]
        if self.last is not None:
            if (t - self.last_t) < self.min_dt:
                return
            if self._dist3(p, self.last) <= self.dist_thresh:
                return
            
        p.append(t - self.last_t)
        self.last, self.last_t = p, t
        if self.n >= self.arr.shape[0]:
            new = np.empty((self.arr.shape[0]*2, 4), dtype=np.float32)
            new[:self.n] = self.arr[:self.n]
            self.arr = new
        self.arr[self.n] = p
        self.n += 1

    def finalize(self):
        np.save(self.save_dir + "/trajectory", self.arr[:self.n])

class CarlaDatasetCollector:
    """
    Collects dataset samples from CARLA simulation.
    Each sample includes:
      - RGB image (from active camera)
      - Ego waypoints
      - Control inputs (steer, throttle, brake, speed)
      - Turn signals / labels
    Samples are not saved continuously, but occasionally (every N frames).
    """

    def __init__(self, save_dir: str, save_interval: int = 10):
        """
        Args:
            save_dir (str): Base directory to save dataset.
            save_interval (int): Save every N frames (to avoid flooding disk).
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.img_dir = self.save_dir / "images"
        self.img_dir.mkdir(exist_ok=True)

        self.meta_dir = self.save_dir / "metadata"
        self.meta_dir.mkdir(exist_ok=True)

        self.save_interval = save_interval
        self.frame_count = 0
        self.sample_idx = 0
        
        self.saver = AsyncSaver()
        self.time_start = time.time()

    def maybe_save(
        self,
        frame: np.ndarray,
        ego_waypoints: np.ndarray,
        control: Dict[str, Any],
        turn_signal: str,
    ) -> None:
        """
        Save dataset sample occasionally.

        Args:
            frame (np.ndarray): RGB image (H, W, 3).
            ego_waypoints (np.ndarray): Waypoints in ego coordinates, shape (N, 2).
            control (dict): Control signals (steer, throttle, brake, speed).
            turn_signal (str): Turn classification label.
        """
        self.frame_count += 1
        if self.frame_count % self.save_interval != 0:
            return  False

        img_name = f"{self.sample_idx:06d}.png"
        img_path = self.img_dir / img_name
        self.saver.save(cv2.imwrite, str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        meta = {
            "img_file": str(img_path.relative_to(self.save_dir)),
            "ego_waypoints": ego_waypoints.tolist(),
            "control": control,
            "turn_signal": turn_signal,
            "timestamp": time.time() - self.time_start,
        }
        np.save(self.meta_dir / f"{self.sample_idx:06d}.npy", meta, allow_pickle=True)

        print(f"[cyan][INFO[/] [purple]({self.__class__.__name__})[/]]: Saved sample {self.sample_idx} → {img_path}")
        self.sample_idx += 1
        return True
    
class CarlaDatasetLoader:
    def __init__(self, dataset_dir: str, downsize_ratio = 1, load_size: int = -1, shuffle = True):
        self.dataset_dir = Path(dataset_dir)
        self.img_dir     = self.dataset_dir / "images"
        self.meta_dir    = self.dataset_dir / "metadata"
        
        if not self.img_dir.exists() or not self.meta_dir.exists():
            raise FileNotFoundError(f"Dataset directories not found: expected 'images/' and 'metadata/'.")
        
        self.samples_dir = [f_name for f_name in self.meta_dir.glob("*.npy")]
        self.samples_dir = np.array(self.samples_dir)
        num_samples      = len(self.samples_dir)
        if load_size != -1 and load_size != len(self.samples_dir):
            if shuffle:
                rand_idx = np.random.randint(0, len(self.samples_dir), load_size)
                self.samples_dir = self.samples_dir[rand_idx]
            else: 
                self.samples_dir = self.samples_dir[np.arange(0, load_size, 1)]

        print(f"[[green]INFO[/] [purple]({self.__class__.__name__})[/]]: Found {num_samples} samples in {self.dataset_dir}. Using {len(self.samples_dir)} samples")

        self.loader = AsyncLoader()
        self.downsize_ratio = downsize_ratio

    def __len__(self):
        return len(self.samples_dir)

    def _get_samples(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index out of range")

        meta_file = Path(self.samples_dir[idx])
        meta      = np.load(meta_file, allow_pickle = True).item()

        img_file  = self.dataset_dir / meta["img_file"]
        self.loader.load(cv2.imread, str(img_file))
        image     = self.loader.get_result(True)[:, :, ::-1]
        if self.downsize_ratio != 1:
            H, W, _   = image.shape
            image = cv2.resize(image, (W // self.downsize_ratio, H // self.downsize_ratio))

        return {
            "image": image,
            "ego_waypoints": np.array(meta["ego_waypoints"], dtype=np.float32),
            "control": meta["control"],
            "turn_signal": meta["turn_signal"],
            "timestamp": meta["timestamp"],
        }

    
    def __getitem__(self, idx):
        return self._get_samples(idx)

    CONTROL_KEYS = ["steer", "throttle", "brake", "velocity"]

    @classmethod
    def collate_fn(cls, batch):
        images   = torch.stack([torch.from_numpy(np.ascontiguousarray(data["image"])) for data in batch]).permute(0, 3, 1, 2) / 255.0
        wp       = torch.stack([torch.from_numpy(data["ego_waypoints"]) for data in batch])
        controls = torch.tensor([
            [data['control'][key] for key in cls.CONTROL_KEYS] for data in batch
        ], dtype = torch.float32)
        turn_signals = torch.tensor([data['turn_signal'] for data in batch], dtype = torch.long)
        
        return images, wp, controls, turn_signals

    def split(self, train = 0.9, val = 0.1):
        
        n_total = self.__len__()
        n_train = int(n_total * train)
        n_val   = int(n_total * val)
        n_test  = n_total - n_train - n_val

        return random_split(self, [n_train, n_val, n_test])
        
        
class AsyncSaver:
    def __init__(self):
        self.q = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while self.running:
            try:
                func, args = self.q.get(timeout=1)
                func(*args)
            except queue.Empty:
                continue

    def save(self, func, *args):
        self.q.put((func, args))

    def stop(self):
        self.running = False
        self.worker.join()

class AsyncLoader:
    def __init__(self):
        self.q = queue.Queue()         # tasks
        self.results = queue.Queue()   # results
        self.running = True
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while self.running:
            try:
                func, args, kwargs = self.q.get(timeout=1)
                result = func(*args, **kwargs)
                self.results.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                self.results.put(e)

    def load(self, func, *args, **kwargs):
        """Enqueue a function call. Returns immediately."""
        self.q.put((func, args, kwargs))

    def get_result(self, block=True, timeout=None):
        """
        Retrieve the next available result.
        If no result is ready:
          - block=True waits until available (or timeout).
          - block=False returns immediately (or raises queue.Empty).
        """
        return self.results.get(block=block, timeout=timeout)

    def stop(self):
        self.running = False
        self.worker.join()
