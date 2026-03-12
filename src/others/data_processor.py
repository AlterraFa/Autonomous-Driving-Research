import os, sys
import toml
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import cv2
import time
import numpy as np
import threading, queue

from pathlib import Path
from typing import Dict, Any

from src.messages.logger import Logger

conf = toml.load(os.path.join(parent, "../config/config.toml"))
quality = conf['Picture']['quality']
class TrajectoryBuffer:
    def __init__(self, save_dir: str, init_cap = 8192, dist_thresh_m = 0, min_dt_s = 0.05):
        self.log = Logger()
        self.log.DEBUG("SAVING VEHICLE TRAJECTORY")
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

from src.messages.message_handler import MessageSubscriber
from src.messages.all_messages import ServerRuntime
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

    def __init__(self, save_dir: str, save_interval: int = None, fps: int = None):
        """
        Args:
            save_dir (str): Base directory to save dataset.
            save_interval (int): Save every N frames (to avoid flooding disk).
        """
        self.log = Logger()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.img_dir = self.save_dir / "images"
        self.img_dir.mkdir(exist_ok=True)

        self.meta_dir = self.save_dir / "metadata"
        self.meta_dir.mkdir(exist_ok=True)

        self.save_interval = save_interval
        self.fps = fps
        self.frame_count = 0
        self.sample_idx = 0
        
        self.server_runtime = MessageSubscriber(ServerRuntime)
        self.saver = AsyncSaver()
        self.time_start = self.server_runtime.receive()
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        

    def maybe_save(
        self,
        meta: Dict[str, Any],
        **images: np.ndarray,
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
        if self.fps is None:
            if self.frame_count % self.save_interval != 0:
                return  False
        else:
            server_time = self.server_runtime.receive()
            if server_time - self.time_start < (1 / self.fps):
                return False
            else:
                self.time_start = server_time

        # lock image keys on first run
        if not hasattr(self, "_image_keys"):
            self._image_keys = list(images.keys())
            self._savers = {}
            for key in self._image_keys:
                saver_dir = self.img_dir / key
                saver_dir.mkdir(parents = True, exist_ok=True)
                # each saver handles its own queue of jobs
                self._savers[key] = AsyncSaver()  # assume AsyncSaver(self, queue, etc.)
        else:
            if set(images.keys()) != set(self._image_keys):
                self.log.ERROR(f"Image keys mismatch. Expected {self._image_keys}, got {list(images.keys())}", exit_code = -1) 
        
        saved_files = {}
        for key in self._image_keys:
            img = images[key]
            fname = f"{key}/{self.sample_idx:06d}_{key}.jpg"
            fpath = self.img_dir / fname
            img_copy = img.copy() 
            # self._savers[key].save(
            #     cv2.imwrite, str(fpath), img_copy, self.encode_param
            # )
            cv2.imwrite(str(fpath), img_copy, self.encode_param)
            saved_files[key] = str(fpath.relative_to(self.save_dir))


        meta = {
            "img_file": saved_files,
            "metadata": meta,
        }

        np.save(self.meta_dir / f"{self.sample_idx:06d}.npy", meta, allow_pickle=True)

        self.sample_idx += 1
        return True
    
        
        
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
