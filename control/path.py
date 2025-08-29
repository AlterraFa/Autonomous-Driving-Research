import numpy as np
from scipy.interpolate import interp1d

def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi
    
class NodeFinder:
    def __init__(self, Ld, path, **kwargs):
        super().__init__(**kwargs)
        self.Ld = Ld
        self.position_idx = 0
        self.path = path
        
    def update_state(self, distance):
        in_range_path_idx = np.where(np.abs(distance - self.Ld) <= self.Ld)[0]
        split_indices = np.where(np.diff(in_range_path_idx) != 1)[0] + 1
        consec_groups = np.split(in_range_path_idx, split_indices)

        for group_indices in consec_groups:
            if self.position_idx in group_indices:
                min_index_group = np.argmin(np.abs(distance[group_indices]))
                if group_indices[min_index_group] > self.position_idx:
                    self.position_idx = group_indices[min_index_group]
                return self.position_idx
        
        return self.position_idx

class PathHandler(NodeFinder):
    """
    defined_path: shape (N,3) -> [x, y, z] or (N,4) -> [x, y, z, psi]
    psi is heading in radians if provided.
    """
    def __init__(self, defined_path: np.ndarray):
        super().__init__(40, defined_path)
        
        assert defined_path.ndim == 2 and defined_path.shape[1] in (3, 4), \
            "defined_path must be (N,3) [x,y,z] or (N,4) [x,y,z,psi]"
        self.path_xyz = defined_path[:, :3].astype(float)
        self.has_yaw = defined_path.shape[1] == 4

        diffs   = np.diff(self.path_xyz, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        s       = np.concatenate(([0.0], np.cumsum(seg_len)))
        keep    = np.r_[True, seg_len > 1e-9]
        
        self.path_xyz = self.path_xyz[keep]
        self.s = s[keep]
        self.seg_vec = np.diff(self.path_xyz, axis=0)
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)

        self.x_of_s = interp1d(self.s, self.path_xyz[:, 0], kind="linear",
                               bounds_error=False, fill_value="extrapolate")
        self.y_of_s = interp1d(self.s, self.path_xyz[:, 1], kind="linear",
                               bounds_error=False, fill_value="extrapolate")
        self.z_of_s = interp1d(self.s, self.path_xyz[:, 2], kind="linear",
                               bounds_error=False, fill_value="extrapolate")

        if self.has_yaw:
            yaw = np.unwrap(defined_path[keep, 3].astype(float))
            self._yaw_unwrapped_of_s = interp1d(self.s, yaw, kind="linear",
                                                bounds_error=False, fill_value="extrapolate")
        else:
            self._yaw_unwrapped_of_s = None

        self.s_min = float(self.s[0])
        self.s_max = float(self.s[-1])
    
    def pose(self, s_query: float):
        """Return [x, y, z, yaw] at arc-length s_query (yaw=None if not available)."""
        x = float(self.x_of_s(s_query))
        y = float(self.y_of_s(s_query))
        z = float(self.z_of_s(s_query))
        if self._yaw_unwrapped_of_s is None:
            return np.array([x, y, z, np.nan])
        yaw = wrap_to_pi(float(self._yaw_unwrapped_of_s(s_query)))
        return np.array([x, y, z, yaw])

    def project(self, point_xyz: np.ndarray):
        """
        Project 3D point onto the path polyline.
        Returns:
          s_star: arc-length at projection,
          p_star: projected point [x,y,z],
          d_star: distance to path,
          idx: segment index used
        """

        p = np.asarray(point_xyz, dtype=float)
        distances = np.linalg.norm(self.path_xyz - p, axis=1)

        idx = self.update_state(distances)

        i = min(idx, len(self.path_xyz) - 2)

        a = self.path_xyz[i]
        b = self.path_xyz[i + 1]
        ab = b - a
        ab_len2 = np.dot(ab, ab)

        if ab_len2 < 1e-18:
            t = 0.0
            proj = a
            best_i = i
        else:
            t_raw = np.dot(p - a, ab) / ab_len2
            proj = a + np.clip(t_raw, 0.0, 1.0) * ab
            best_i = i

            if not self._edge_opposite_test(p, a, b):
                if t_raw <= 0.0 and i > 0:
                    a = self.path_xyz[i - 1]
                    b = self.path_xyz[i]
                    ab = b - a
                    ab_len2 = np.dot(ab, ab)
                    if ab_len2 > 1e-18:
                        t_raw = np.dot(p - a, ab) / ab_len2
                        proj = a + np.clip(t_raw, 0.0, 1.0) * ab
                    best_i = i - 1
                elif t_raw >= 1.0 and i + 2 < len(self.path_xyz):
                    a = self.path_xyz[i + 1]
                    b = self.path_xyz[i + 2]
                    ab = b - a
                    ab_len2 = np.dot(ab, ab)
                    if ab_len2 > 1e-18:
                        t_raw = np.dot(p - a, ab) / ab_len2
                        proj = a + np.clip(t_raw, 0.0, 1.0) * ab
                    best_i = i + 1


        d2 = np.dot(p - proj, p - proj)
        seg_len = np.linalg.norm(b - a)
        t = np.linalg.norm(proj - a) / (seg_len + 1e-12)
        s_star = float(self.s[best_i] + t * seg_len)

        return s_star, proj, float(np.sqrt(d2)), best_i

    @staticmethod
    def _edge_opposite_test(pt, A, B):
        P = np.asarray(pt, dtype=float)
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        AB = B - A
        denom = np.dot(AB, AB)
        if denom == 0:
            return False   # edge of length zero
        
        t = np.dot(P - A, AB) / denom
        return (0.0 < t) and (t < 1.0)

    