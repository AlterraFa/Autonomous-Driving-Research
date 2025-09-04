import time
import numpy as np

class TrajectoryBuffer:
    def __init__(self, init_cap = 8192, dist_thresh_m = 0, min_dt_s = 0.05):
        self.arr = np.empty((init_cap, 4), dtype=np.float32)
        self.n = 0
        self.last = None
        self.last_t = 0.0
        self.dist_thresh = float(dist_thresh_m)
        self.min_dt = float(min_dt_s)

    @staticmethod
    def _dist3(a, b):
        dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    def add_if_needed(self, loc):
        t = time.time()
        p = [float(loc.x), float(loc.y), float(loc.z)]
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

    def save(self, path_no_ext: str):
        np.save(path_no_ext, self.arr[:self.n])