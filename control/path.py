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
        super().__init__(10, defined_path)
        
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

    def waypoints(self, position: np.ndarray, offsets: list[float], yaw: float, return_global = False):
        dist_travelled, *_ = self.project(position)
        wp = []
        for offset in offsets:
            wp += [self.pose(dist_travelled + 2 + offset)[:-1]]
        wp = np.asarray(wp)
        if not return_global:
            return self._ego_transform(wp, yaw, position)
        else:
            return self._ego_transform(wp, yaw, position), wp
        
    def _ego_transform(self, point: np.ndarray, rot: float, trans: np.ndarray):
        x, y, z = trans
        c, s = np.cos(rot), np.sin(rot)

        T = np.array([
            [ c,  s, -x*c - y*s],
            [ s, -c, -x*s + y*c],
            [ 0,  0,        1 ]
        ])

        pts = np.atleast_2d(point)
        pts = np.hstack([pts[:, :2], np.ones((pts.shape[0], 1))])
        local_pts = (T @ pts.T).T
        return local_pts[:, :2] if len(local_pts) > 1 else local_pts[0, :2]

class TurnClassify:
    def __init__(self, threshold: float):
        self.thresh = threshold
        self.first_filter = []
        self.second_filter = []
        self.signal = None
        pass

    @staticmethod
    def consecutive_angles(points: np.ndarray, signed: bool = False) -> np.ndarray:
        pts = points[:, :2]
        A, B, C = pts[:-2], pts[1:-1], pts[2:]
        
        AB = B - A
        BC = C - B
        
        # normalize
        ABn = AB / np.linalg.norm(AB, axis=1, keepdims=True)
        BCn = BC / np.linalg.norm(BC, axis=1, keepdims=True)
        
        dot = np.sum(ABn * BCn, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        
        angles = np.arccos(dot)
        
        if signed:
            cross = ABn[:,0]*BCn[:,1] - ABn[:,1]*BCn[:,0]
            angles *= np.sign(cross)
        
        return angles
    
    def turning_type(self, enable: bool, disable: bool, waypoints: np.ndarray):
        if enable:
            curve_deg  = np.degrees(self.consecutive_angles(waypoints, True))
            dominant   = curve_deg[np.argmax(np.abs(curve_deg))]
            direction  = dominant > 0
            is_turning = np.any(np.abs(curve_deg) > self.thresh)
            
            if not is_turning:
                cmd = 0
            elif direction > 0:
                cmd = 1
            else:
                cmd = 2
            self.first_filter.append(cmd)
            first_signal  = np.argmax(np.bincount(self.first_filter, minlength = 3))
            self.second_filter.append(first_signal)
            second_signal = np.argmax(np.bincount(self.second_filter, minlength = 3))
            
            self.signal = second_signal
        if disable and not enable:
            self.signal = -1
            self.first_filter.clear()
            self.second_filter.clear()
        
        return self.signal