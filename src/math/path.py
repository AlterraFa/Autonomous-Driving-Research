import os, sys
import toml
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import numpy as np
import carla
import networkx as nx
import math
import time

from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from src.messages.logger import Logger
from src.control.world import World

from numba import njit
import pyclothoids
# --- Global Tuning Configurations ---
 
SPLINE_MIN_POINTS = 20
SPLINE_POINTS_MULTIPLIER = 5
SMOOTHING_BLEND_HALF_WINDOW = 4
SMOOTHING_WINDOW_SIZE = 3
ALIGN_TOLERANCE = 1e-2
SPLINE_DEDUPLICATION_TOLERANCE = 1e-3
B_SMOOTH_S = 2.0
B_SMOOTH_K = 3


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

@njit(fastmath = True)
def compute_distance(path_xyz, position):
    n = path_xyz.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        dx = path_xyz[i,0] - position[0]
        dy = path_xyz[i,1] - position[1]
        dz = path_xyz[i,2] - position[2]
        out[i] = (dx*dx + dy*dy + dz*dz) ** 0.5
    return out
    
class NodeFinder:
    """
    A utility class for tracking the closest position along a predefined list of nodes (e.g. coordinates).
    It maintains the current positional index to efficiently limit search distances dynamically.
    """
    def __init__(self, Ld, path, update_dist = .5, **kwargs):
        super().__init__(**kwargs)
        self.Ld = Ld
        self.position_idx = 0
        self.update_dist = update_dist
        self.path = path
        self.path_length = len(path)
        
        self.kdtree = cKDTree(self.path)
    
    # Keeping this just in case
    def update_state(self, p):

        distance = compute_distance(self.path, p)
        in_range_path_idx = np.where(np.abs(distance - self.Ld) <= self.Ld)[0]
        
        split_indices = np.where(np.diff(in_range_path_idx) != 1)[0] + 1
        consec_groups = np.split(in_range_path_idx, split_indices)

        for group_indices in consec_groups:
            if self.position_idx in group_indices:
                min_index_group = np.argmin(np.abs(distance[group_indices]))
                candidate_idx = group_indices[min_index_group]
                if candidate_idx > self.position_idx and abs(distance[candidate_idx] - distance[self.position_idx]) > self.update_dist:
                    self.position_idx = candidate_idx
                return self.position_idx
        
        return self.position_idx
    
    def update_state(self, p):
        """Optimized version using KDTree"""
        dists, idxs = self.kdtree.query(p, k=self.path_length, distance_upper_bound=2*self.Ld)
        mask = np.isfinite(dists)
        idxs, dists = idxs[mask], dists[mask]

        # Sort by index (path order)
        order = np.argsort(idxs)
        idxs, dists = idxs[order], dists[order]

        
        split_indices = np.where(np.diff(idxs) != 1)[0] + 1
        consec_groups_idx   = np.split(idxs, split_indices)
        consec_groups_dists = np.split(dists, split_indices)

        # Split into groups where candidate index(local minimum distance) resides
        # Choose based on if position index is in within a group (exploting high update frequency) and if the candidate index is greater than the current onek
        for group_indices, group_distance in zip(consec_groups_idx, consec_groups_dists):
            if self.position_idx in group_indices:
                
                min_index_group = np.argmin(np.abs(group_distance))
                candidate_idx = group_indices[min_index_group]

                relative_curr      = np.where(self.position_idx == group_indices)[0][0]
                relative_candidate = np.where(candidate_idx == group_indices)[0][0]

                
                if candidate_idx > self.position_idx and abs(group_distance[relative_candidate] - dists[relative_curr]) > self.update_dist:
                    self.position_idx = candidate_idx
                return self.position_idx
        return self.position_idx

class PathHandler(NodeFinder):
    """
    Core handler for trajectory state abstraction and parameterization along a physical layout.
    
    This class builds continuous polynomial interpolations (PchipInterpolator) across spatial distances (`s`) 
    and temporal deltas (`t`). It supports complex projection queries for location and array offsets,
    along with returning smoothly interpolated 6-DoF Roll-Pitch-Yaw matrices seamlessly synced with the topology.
    
    defined_path configurations: 
      (N,3) -> [x, y, z]
      (N,4) -> [x, y, z, t]   (t = delta time recording)
      (N,6) -> [x, y, z, r, p, y]
      (N,7) -> [x, y, z, r, p, y, t]
    """
    def __init__(self, defined_path: np.ndarray, extrapolate: bool = True):
        self.log = Logger()
        super().__init__(15, defined_path[:, :3])

        assert defined_path.ndim == 2 and defined_path.shape[1] in (3, 4, 6, 7), \
            "defined_path must be (N,3) [x,y,z], (N,4) [x,y,z,t], (N,6) [x,y,z,r,p,y], or (N,7) [x,y,z,r,p,y,t]"
        
        self.path_xyz = defined_path[:, :3].astype(float)
        self.has_rot = defined_path.shape[1] in (6, 7)
        if self.has_rot:
            self.path_rpy = defined_path[:, 3:6].astype(float)
            self.path_rpy[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(self.path_rpy[:, 0])))
            self.path_rpy[:, 1] = np.rad2deg(np.unwrap(np.deg2rad(self.path_rpy[:, 1])))
            self.path_rpy[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(self.path_rpy[:, 2])))
            
        self.has_time = defined_path.shape[1] in (4, 7)

        # --- arc-length for projection ---
        diffs   = np.diff(self.path_xyz, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        s       = np.concatenate(([0.0], np.cumsum(seg_len)))

        # Ensure strictly increasing for Pchip by pushing identical distances forward slightly
        eps_dist = 1e-5
        for i in range(1, len(s)):
            if s[i] <= s[i-1]:
                s[i] = s[i-1] + eps_dist

        self.s = s
        self.seg_vec = diffs
        self.seg_len = seg_len

        # --- interpolation in s ---
        from scipy.interpolate import PchipInterpolator
        if len(self.s) >= 4:
            self.x_of_s = PchipInterpolator(self.s, self.path_xyz[:, 0], extrapolate=extrapolate)
            self.y_of_s = PchipInterpolator(self.s, self.path_xyz[:, 1], extrapolate=extrapolate)
            self.z_of_s = PchipInterpolator(self.s, self.path_xyz[:, 2], extrapolate=extrapolate)
            if self.has_rot:
                self.roll_of_s = PchipInterpolator(self.s, self.path_rpy[:, 0], extrapolate=extrapolate)
                self.pitch_of_s = PchipInterpolator(self.s, self.path_rpy[:, 1], extrapolate=extrapolate)
                self.yaw_of_s = PchipInterpolator(self.s, self.path_rpy[:, 2], extrapolate=extrapolate)
        else:
            self.x_of_s = interp1d(self.s, self.path_xyz[:, 0], kind="linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 0], self.path_xyz[-1, 0]))
            self.y_of_s = interp1d(self.s, self.path_xyz[:, 1], kind="linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 1], self.path_xyz[-1, 1]))
            self.z_of_s = interp1d(self.s, self.path_xyz[:, 2], kind="linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 2], self.path_xyz[-1, 2]))
            if self.has_rot:
                self.roll_of_s = interp1d(self.s, self.path_rpy[:, 0], kind="linear", bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_rpy[0, 0], self.path_rpy[-1, 0]))
                self.pitch_of_s = interp1d(self.s, self.path_rpy[:, 1], kind="linear", bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_rpy[0, 1], self.path_rpy[-1, 1]))
                self.yaw_of_s = interp1d(self.s, self.path_rpy[:, 2], kind="linear", bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_rpy[0, 2], self.path_rpy[-1, 2]))

        # --- interpolation in t if available ---
        if self.has_time:
            self.log.INFO("Found time vector. Enabling spatial and temporal mode")
            
            self.timer = 0
            # Calculate absolute time
            t_col = defined_path[:, -1].astype(float)
            self.t = np.cumsum(t_col)
            
            # Ensure strictly increasing for Pchip
            eps_t = 1e-5
            for i in range(1, len(self.t)):
                if self.t[i] <= self.t[i-1]:
                    self.t[i] = self.t[i-1] + eps_t
            
            if len(self.t) >= 4:
                self.x_of_t = PchipInterpolator(self.t, self.path_xyz[:, 0], extrapolate=extrapolate)
                self.y_of_t = PchipInterpolator(self.t, self.path_xyz[:, 1], extrapolate=extrapolate)
                self.z_of_t = PchipInterpolator(self.t, self.path_xyz[:, 2], extrapolate=extrapolate)
                if self.has_rot:
                    self.roll_of_t = PchipInterpolator(self.t, self.path_rpy[:, 0], extrapolate=extrapolate)
                    self.pitch_of_t = PchipInterpolator(self.t, self.path_rpy[:, 1], extrapolate=extrapolate)
                    self.yaw_of_t = PchipInterpolator(self.t, self.path_rpy[:, 2], extrapolate=extrapolate)
            else:
                self.x_of_t = interp1d(self.t, self.path_xyz[:, 0], kind="linear",
                                       bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 0], self.path_xyz[-1, 0]))
                self.y_of_t = interp1d(self.t, self.path_xyz[:, 1], kind="linear",
                                       bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 1], self.path_xyz[-1, 1]))
                self.z_of_t = interp1d(self.t, self.path_xyz[:, 2], kind="linear",
                                       bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 2], self.path_xyz[-1, 2]))
                if self.has_rot:
                    self.roll_of_t = interp1d(self.t, self.path_rpy[:, 0], kind="linear", bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_rpy[0, 0], self.path_rpy[-1, 0]))
                    self.pitch_of_t = interp1d(self.t, self.path_rpy[:, 1], kind="linear", bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_rpy[0, 1], self.path_rpy[-1, 1]))
                    self.yaw_of_t = interp1d(self.t, self.path_rpy[:, 2], kind="linear", bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_rpy[0, 2], self.path_rpy[-1, 2]))
            
            # keep time interpolation linear to prevent non-monotonic mappings
            self.t_of_s = interp1d(self.s, self.t, kind="linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (t_col[0], t_col[-1]))

        else:
            self.log.WARNING("Didn't find time vector. Disabling temporal mode")

            self.t = None
            
    
    def pose(self, query: float, use_time: bool = False):
        """
        Return [x, y, z] or [x, y, z, roll, pitch, yaw].
        If use_time=False -> query is arc-length s.
        If use_time=True  -> query is timestamp t (requires time column).
        """
        if use_time:
            if self.t is None:
                raise RuntimeError("This path has no time column.")
            xyz = np.stack([
                self.x_of_t(query),
                self.y_of_t(query),
                self.z_of_t(query)
            ], axis = -1)
            
            if self.has_rot:
                rpy = np.stack([
                    self.roll_of_t(query),
                    self.pitch_of_t(query),
                    self.yaw_of_t(query)
                ], axis = -1)
                rpy = (rpy + 180) % 360 - 180
                return xyz, rpy
            return xyz, None
        else:
            xyz = np.stack([
                self.x_of_s(query),
                self.y_of_s(query),
                self.z_of_s(query)
            ], axis = -1)
            
            if self.has_rot:
                rpy = np.stack([
                    self.roll_of_s(query),
                    self.pitch_of_s(query),
                    self.yaw_of_s(query)
                ], axis = -1)
                rpy = (rpy + 180) % 360 - 180
                return xyz, rpy
            return xyz, None

    def project(self, point_xyz: np.ndarray):
        """
        Project 3D point onto the path polyline.
        Returns:
          s_star: arc-length at projection,
          p_star: projected point [x,y,z],
          d_star: distance to path,
          idx: segment index used
        """

        p = np.asarray(point_xyz, dtype=float)[:3]

        idx = self.update_state(p)

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

        side_val = self.get_side(p, proj, a, b)

        d2 = np.dot(p - proj, p - proj)
        seg_len = np.linalg.norm(b - a)
        t = np.linalg.norm(proj - a) / (seg_len + 1e-12)
        s_star = float(self.s[best_i] + t * seg_len)
        
        return s_star, proj, float(np.sqrt(d2)), idx / self.path_length, side_val

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

    @staticmethod
    def _project_point_to_segment(P, A, B):
        AB = B - A
        denom = float(np.dot(AB, AB))
        if denom < 1e-9:
            return A.copy(), 0.0
        t = float(np.dot(P - A, AB) / denom)
        t = float(np.clip(t, 0.0, 1.0))  # remove clip to project onto infinite line
        Q = A + t * AB
        return Q

    def get_side(self, p, proj, a, b):
        forward = b - a
        
        up = np.array([0, 0, 1]) 
        right_vec = np.cross(forward, up)
        
        norm = np.linalg.norm(right_vec)
        if norm < 1e-9:
            return 0.0
        right_vec /= norm

        error_vec = p - proj

        side_dist = np.dot(error_vec, right_vec)
        
        return side_dist

    def waypoints(self, position: np.ndarray, offsets: list[float], use_time: bool = False, merge = False):
        dist_travelled, *_, side_val = self.project(position)
        if not use_time:
            dist_offset = dist_travelled + np.array(offsets)
            wp, wp_rpy = self.pose(dist_offset)
        else: 
            if self.has_time == False:
                self.log.ERROR(f"Temporal mode was disabled but you enabled `use_time` argument. Exiting...", exit_code = -1)
                
            current_time = self.t_of_s(dist_travelled)
            time_offset = current_time + offsets
            wp, wp_rpy = self.pose(time_offset, True)

        if merge:
            wp = self._loc_merging(wp, side_val, offsets)
            if self.has_rot:
                wp_rpy = self._rot_merging(wp, wp_rpy)
                
        wp = np.asarray(wp)
        return wp, wp_rpy

    def _rot_merging(self, merged_wps: list, centerline_rpy: np.ndarray):
        """
        Updates the yaw in centerline_rpy to match the new trajectory tangent defined by merged_wps.
        """
        if centerline_rpy is None: 
            return None
            
        merged_rpy = np.copy(centerline_rpy)
        for i in range(len(merged_wps)):
            if i < len(merged_wps) - 1:
                tangent = merged_wps[i+1][:3] - merged_wps[i][:3]
            else:
                tangent = merged_wps[i][:3] - merged_wps[i-1][:3]
                
            norm = np.linalg.norm(tangent)
            if norm > 1e-6:
                yaw = np.degrees(np.arctan2(tangent[1], tangent[0]))
                merged_rpy[i, 2] = yaw
                
        return merged_rpy

    def _loc_merging(self, centerline_wps: np.ndarray, current_side_val: float, offsets: list[float], merge_fraction: float = 1.0):
        """
        Shifts centerline waypoints laterally.
        merge_fraction: 0.0 to 1.0. Determines how far into the offsets the path 
                        should return to the centerline.
        """
        offsets_np = np.array(offsets)
        horizon = offsets_np[-1] + 1e-6 
        
        # This is the distance at which we want to be perfectly at 0 offset
        merge_target_dist = horizon * merge_fraction
        
        merged_pts = []
        for i, p_center in enumerate(centerline_wps):
            # 1. Calculate ratio relative to the merge_target_dist instead of full horizon
            # We clip it at 1.0 so that points beyond the merge fraction stay at 0 offset
            ratio = min(1.0, offsets_np[i] / (merge_target_dist + 1e-6))
            
            # 2. Quadratic decay
            # If ratio is 1.0 (at or beyond merge point), decay is 0.0
            decay = (1.0 - ratio) ** 2
            active_offset = current_side_val * decay

            # 3. Find the 'Right' vector
            if i < len(centerline_wps) - 1:
                tangent = centerline_wps[i+1][:3] - p_center[:3]
            else:
                tangent = p_center[:3] - centerline_wps[i-1][:3]

            up = np.array([0, 0, 1]) 
            right_vec = np.cross(tangent, up)
            
            norm = np.linalg.norm(right_vec)
            if norm > 1e-6:
                right_vec /= norm
                p_merged = np.copy(p_center)
                p_merged[:3] = p_merged[:3] + (right_vec * active_offset)
            else:
                p_merged = p_center
                
            merged_pts.append(p_merged)

        return merged_pts
        
class BranchingPath:
    """
    Provides multi-lane junction evaluation by forecasting alternate entry and exit topology paths.
    
    Given a driving trajectory, this class isolates map junctions, groups entry clusters, and artificially 
    spawns side-lane branching waypoints by interpolating their connectivity parallel to the ego vehicle.
    """
    def __init__(self, world: World):
        self.virt_world = world

    def brancher(self, global_wp: np.ndarray, scout_pts: np.ndarray, persist_dist = np.inf):
        junctions_metadata = self.virt_world.get_multi_junctions(global_wp) # this is already sorted based on distance (closest -> furthest)
        
        
        branch_wp = []
        for junction in junctions_metadata:
            wp_pairs = junction.get_waypoints(carla.LaneType.Driving)
            pairs = _find_entry_clusters(wp_pairs, scout_pts)
            
            if len(pairs) == 1: continue # If there's only 1 pair then just skip it
            
            entry = carla_waypoints_to_np(pairs[0][0])[0]
            insert_idx = self.insertion_point(global_wp, entry)

            path_in_junction = global_wp[insert_idx: ]
            remainder_path   = global_wp[: insert_idx]
            
            if len(path_in_junction) != len(global_wp):
                path_in_junction = np.vstack([entry[None, ...], path_in_junction])
                extra_length = 0
            else: extra_length = np.linalg.norm(entry - global_wp[0]) 

            if extra_length > persist_dist: continue
            segment_length = self.compute_seg_length(path_in_junction) + extra_length 
            
            branch_candidates = []
            for pair in pairs:
                entry_wp, exit_wp = pair
                wp_junction        = waypoints_between(entry_wp, exit_wp, step = 2)
                wp_junction_loc    = carla_waypoints_to_np(wp_junction)
                junction_seglength = self.compute_seg_length(wp_junction_loc)
                
                z_of_s = interp1d(junction_seglength, wp_junction_loc[:, 2],
                                bounds_error=False,
                                fill_value=(wp_junction_loc[0, 2], wp_junction_loc[-1, 2]))

                y_of_s = interp1d(junction_seglength, wp_junction_loc[:, 1],
                                bounds_error=False,
                                fill_value=(wp_junction_loc[0, 1], wp_junction_loc[-1, 1]))

                x_of_s = interp1d(junction_seglength, wp_junction_loc[:, 0],
                                bounds_error=False,
                                fill_value=(wp_junction_loc[0, 0], wp_junction_loc[-1, 0]))                

                interp_wp = np.stack([
                    x_of_s(segment_length),
                    y_of_s(segment_length),
                    z_of_s(segment_length),
                ], axis = -1)
                if extra_length == 0:
                    padded_wp = np.concatenate([remainder_path, interp_wp[1: ]]) # Remove the entry point
                else:
                    padded_wp = np.concatenate([remainder_path, interp_wp[0: ]])
                    
                if np.linalg.norm(padded_wp[-1] - carla_waypoints_to_np(exit_wp)) < 0.5: continue
                    
                dists = np.linalg.norm(global_wp[:, None, :] - padded_wp[None, :, :], axis=2)
                min_dist = dists.mean()
                branch_candidates.append((min_dist, padded_wp))

            branch_candidates.sort(key=lambda x: x[0])
            branch_candidates = branch_candidates[1:]  # remove closest branch

            for _, branch in branch_candidates:
                branch_wp.append(branch)
        
        branch_wp += [global_wp]
        return np.array(branch_wp)
                    
            
            
    @staticmethod
    def insertion_point(waypoints: np.ndarray, compare_point: np.ndarray) -> int:
        closest_idx = np.argmin(np.linalg.norm(waypoints - compare_point, axis = 1))
        closest_wp = waypoints[closest_idx]

        j = closest_idx
        closest_backward = closest_wp
        while j != 0:
            closest_backward = waypoints[j]
            if not np.allclose(closest_wp, closest_backward, atol = 1e-2):
                break
            j -= 1
            
        k = closest_idx
        closest_forward = np.array(closest_wp)
        while k != len(waypoints):
            closest_forward = waypoints[k]
            if not np.allclose(closest_wp, closest_forward, atol = 1e-2):
                break
            k += 1
            
        choose_backward = PathHandler._edge_opposite_test(compare_point, closest_backward, closest_wp)
        choose_forward  = PathHandler._edge_opposite_test(compare_point, closest_forward, closest_wp)
        if choose_backward == choose_forward: # Projected point too close to coordinates
            insert_idx  = closest_idx
        elif choose_backward:
            insert_idx  = closest_idx 
        elif choose_forward:
            insert_idx  = closest_idx + 1
            
        return insert_idx
    
    @staticmethod
    def compute_seg_length(waypoints: np.ndarray) -> float:
        if len(waypoints) < 2:
            return 0.0
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.cumsum(np.concatenate([[0], segment_lengths]))
                
from src.messages.all_messages import ServerFps
from src.messages.message_handler import MessageSubscriber
class TurnClassify:
    """
    Analyzes the geometric profiles of intersection trajectories to flag structural characteristics.
    
    Uses CARLA's driving layout to project and fit junction sequences, distinguishing 
    straight-through roads from complex turns (e.g. left/right turn classifications) based 
    on heading differentials and tangent thresholds.
    """
    def __init__(self, world: World, threshold_deg: float = 45):
        self.thresh_deg = threshold_deg
        self.signal = None
        self.virt_world = world
        
        # Cache for interpolated junction waypoints and clothoid data
        self._cache = {
            'interpolated_points': None,
            'clothoid': None,
            'entry_heading': None,
            'exit_heading': None,
            'entry_wp_id': None,
            'exit_wp_id': None
        }
        self.log = Logger()
        self.sub_server_fps = MessageSubscriber(ServerFps)

    def _interpolate_junction_path(self, entry_wp, exit_wp, step: float = 0.5):
        """
        Interpolate points between entry and exit waypoints within the junction.
        
        Parameters
        ----------
        entry_wp : carla.Waypoint
            Entry waypoint
        exit_wp : carla.Waypoint
            Exit waypoint
        step : float
            Distance step between interpolated points
            
        Returns
        -------
        np.ndarray
            (N, 3) array of interpolated points [x, y, z]
        """
        wp_list = waypoints_between(entry_wp, exit_wp, step=step)
        points = carla_waypoints_to_np(wp_list)
        return points

    def _fit_clothoid_and_extract_headings(self, entry_wp, exit_wp, points: np.ndarray):
        """
        Fit a clothoid curve using G1Hermite interpolation and extract headings at entry/exit.
        
        Uses the Hermite interpolation method with entry/exit positions and headings to fit
        a clothoid curve. Headings are computed by numerical differentiation along the curve.
        
        Parameters
        ----------
        entry_wp : carla.Waypoint
            Entry waypoint
        exit_wp : carla.Waypoint
            Exit waypoint
        points : np.ndarray
            (N, 3) array of interpolated points (used for validation)
            
        Returns
        -------
        tuple
            (entry_heading, exit_heading) in radians
        """
        try:
            # Get entry and exit positions
            entry_loc = entry_wp.transform.location
            exit_loc = exit_wp.transform.location
            
            x_start = entry_loc.x
            y_start = entry_loc.y
            theta_start = waypoint_heading(entry_wp)
            
            x_end = exit_loc.x
            y_end = exit_loc.y
            theta_end = waypoint_heading(exit_wp)
            
            # Fit clothoid using G1Hermite (Hermite interpolation with positions and headings)
            try:
                clothoid = pyclothoids.Clothoid.G1Hermite(
                    x_start, y_start, theta_start,
                    x_end, y_end, theta_end
                )
                self._cache['clothoid'] = clothoid
                
                # Extract headings at entry and exit using numerical differentiation
                calc_ds = 1e-4
                
                # Entry heading: differentiate X, Y at s=0
                dx_start = clothoid.X(calc_ds) - clothoid.X(0.0)
                dy_start = clothoid.Y(calc_ds) - clothoid.Y(0.0)
                entry_heading = np.arctan2(dy_start, dx_start)
                
                # Exit heading: differentiate X, Y at s=length
                s_end = clothoid.length
                dx_end = clothoid.X(s_end) - clothoid.X(s_end - calc_ds)
                dy_end = clothoid.Y(s_end) - clothoid.Y(s_end - calc_ds)
                exit_heading = np.arctan2(dy_end, dx_end)
                
                return entry_heading, exit_heading
            except Exception as e:
                self.log.WARNING(f"G1Hermite clothoid fitting failed: {e}, falling back to waypoint headings")
                return theta_start, theta_end
                
        except Exception as e:
            self.log.WARNING(f"Error in clothoid extraction: {e}, falling back to waypoint headings")
            return waypoint_heading(entry_wp), waypoint_heading(exit_wp)

    def _clear_cache(self):
        """Clear the junction cache."""
        self._cache = {
            'interpolated_points': None,
            'clothoid': None,
            'entry_heading': None,
            'exit_heading': None,
            'entry_wp_id': None,
            'exit_wp_id': None
        }

    def turning_type(self, enable: bool, junction, disable: bool, waypoints: np.ndarray, debug=False):
        if enable:
            wp_pairs = junction.get_waypoints(carla.LaneType.Driving)
            possible_pairs = _find_entry_clusters(wp_pairs, waypoints)                    
            choosen_pairs = _find_exit(possible_pairs, waypoints)

            server_fps = self.sub_server_fps.receive()
            if debug:
                self.virt_world.world.debug.draw_point(
                    choosen_pairs[0].transform.location,
                    size=0.18,
                    color=carla.Color(0, 0, 255),
                    life_time=3.0 * (1 / server_fps)
                )
                self.virt_world.world.debug.draw_point(
                    choosen_pairs[1].transform.location,
                    size=0.18,
                    color=carla.Color(0, 0, 255),
                    life_time=3.0 * (1 / server_fps)
                )
            
            entry_wp = choosen_pairs[0]
            exit_wp = choosen_pairs[1]
            
            # Check if we need to recompute or use cache using stable waypoint properties
            entry_wp_key = (entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id)
            exit_wp_key = (exit_wp.road_id, exit_wp.section_id, exit_wp.lane_id)
            
            if (self._cache['entry_wp_id'] != entry_wp_key or 
                self._cache['exit_wp_id'] != exit_wp_key or
                self._cache['entry_heading'] is None):
                
                # Interpolate points between entry and exit
                try:
                    interpolated_pts = self._interpolate_junction_path(entry_wp, exit_wp)
                    self._cache['interpolated_points'] = interpolated_pts
                    
                    # Fit clothoid and extract headings
                    entry_heading, exit_heading = self._fit_clothoid_and_extract_headings(
                        entry_wp, exit_wp, interpolated_pts
                    )
                    self._cache['entry_heading'] = entry_heading
                    self._cache['exit_heading'] = exit_heading
                    self._cache['entry_wp_id'] = entry_wp_key
                    self._cache['exit_wp_id'] = exit_wp_key
                except Exception as e:
                    self.log.WARNING(f"Failed to process junction path: {e}")
                    entry_heading = waypoint_heading(entry_wp)
                    exit_heading = waypoint_heading(exit_wp)
                    self._cache['entry_heading'] = entry_heading
                    self._cache['exit_heading'] = exit_heading
                    self._cache['entry_wp_id'] = entry_wp_key
                    self._cache['exit_wp_id'] = exit_wp_key
            else:
                # Use cached headings
                entry_heading = self._cache['entry_heading']
                exit_heading = self._cache['exit_heading']

            delta = np.arctan2(
                np.sin(exit_heading - entry_heading),
                np.cos(exit_heading - entry_heading)
            )
            
            if abs(delta) < np.radians(self.thresh_deg):
                signal = 0
            elif delta < 0:
                signal = 1
            else:
                signal = 2
            if self.signal != signal:
                if signal == 0:
                    self.log.DEBUG("[blue]ENTER[/blue]:", f"Entering intersection ID: [cyan]{entry_wp_key[0]}[/cyan]. Mode: [bold]Straight[/bold]")
                if signal == 1:
                    self.log.DEBUG("[blue]ENTER[/blue]:", f"Entering intersection ID: [cyan]{entry_wp_key[0]}[/cyan]. Mode: [bold]Turn Left[/bold]")
                if signal == 2:
                    self.log.DEBUG("[blue]ENTER[/blue]:", f"Entering intersection ID: [cyan]{entry_wp_key[0]}[/cyan]. Mode: [bold]Turn Right[/bold]")

                self.signal = signal
                
        if disable and not enable:
            if self._cache['entry_wp_id'] is not None:
                self.log.DEBUG("[color(100)]EXITED[/color(100)]:", f"Exited intersection ID: [cyan]{self._cache['entry_wp_id'][0]}[/cyan]. Mode: [bold]Keep lane[/bold]")
            self._clear_cache()
            self.signal = -1

        return self.signal

from scipy.interpolate import splprep, splev
class WaypointsAlign:
    """
    Handles map-level routing, smoothing, and geometric alignment of recorded trace coordinates.
    
    Acts as a middleware filter to map unaligned raw GPS or recording sequences firmly against CARLA's 
    core skeleton structure. It utilizes KD-Trees and spline evaluations (splprep) to fuse and snap
    coordinates onto logical pathways accurately without erratic positional shifts.
    """
    def __init__(self, world: World, waypoint_distance):
        self.world = world

        carla_map = world.world.get_map()
        waypoints = carla_map.generate_waypoints(distance=waypoint_distance)

        wp_dict = {(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z): wp for wp in waypoints}
        _wp_list = list(wp_dict.values())
        _wps = np.array([[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp in _wp_list])
        _tree = cKDTree(_wps)

        self._wp_list = _wp_list
        self._tree = _tree

    @staticmethod
    def b_smooth(points: np.ndarray):
        # NOTE: splprep will crash if consecutive points are identical. 
        # Add deduplication here if you ever experience a ValueError.
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
                    
        tck, u = splprep([x, y, z], s=B_SMOOTH_S, k=B_SMOOTH_K)
        
        new_u = np.linspace(0, 1, len(points))
        smooth_x, smooth_y, smooth_z = splev(new_u, tck)
        
        dx = np.gradient(smooth_x)
        dy = np.gradient(smooth_y)
        smooth_yaw = np.degrees(np.arctan2(dy, dx))
        
        group_meta = np.column_stack((smooth_x, smooth_y, smooth_z, smooth_yaw))
        return group_meta

    def get_jid_for_point(self, point):
        segs = self.world.get_segments_from_points("junction", np.array([point]))
        return segs[0].id if segs else None

    def _filter_valid_junctions(self, coordinates: np.ndarray):
        
        junctions = self.world.get_segments_from_points("junction", coordinates)

        last_jid = None
        for junction_id in range(len(junctions) - 1, -1, -1):
            if junctions[junction_id].id == last_jid:
                junctions.pop(junction_id)
            else:
                last_jid = junctions[junction_id].id

        jids = [self.get_jid_for_point(p) for p in coordinates]
        junction_by_id = {j.id: j for j in junctions}
        
        return jids, junction_by_id

    def _group_coordinates_by_junction(self, coordinates: np.ndarray, heading: np.ndarray, jids: list):
        groups: list[tuple] = []
        current_group: list = []
        current_jid = jids[0]
        group_start = 0

        if current_jid is not None:
            current_group.append(np.concatenate([coordinates[0], heading[0]]))

        for ci, (pt, head, jid) in enumerate(zip(coordinates[1:], heading[1:], jids[1:]), start=1):
            transform = np.concatenate([pt, head])
            if jid == current_jid:
                if jid is not None:
                    current_group.append(transform)
            else:
                if current_jid is not None and current_group:
                    groups.append((current_jid, np.array(current_group), ci))
                current_group = [transform] if jid is not None else []
                current_jid = jid
                group_start = ci

        if current_jid is not None and current_group:
            groups.append((current_jid, np.array(current_group), len(coordinates)))

        return groups

    def _precompute_junction_paths(self, groups: list, junction_by_id: dict):
        junctions_metadata_groups = []

        for idx, (group_jid, coordinate_group, group_end) in enumerate(groups):
            junction = junction_by_id.get(group_jid)
            if junction is None:
                junctions_metadata_groups.append(np.array([]))
                continue

            wp_pairs = junction.get_waypoints(carla.LaneType.Driving)
            best_pair, min_score = _unified_entry_exit_finder(wp_pairs, coordinate_group)
            
            if best_pair is not None:
                entry_wp, exit_wp = best_pair
                wp_in_junctions = waypoints_between(entry_wp, exit_wp)
            else:
                wp_in_junctions = []

            raw_pts = []
            for wp in wp_in_junctions:
                loc = wp.transform.location
                rot = wp.transform.rotation
                raw_pts.append([loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw])

            junctions_metadata_groups.append(np.array(raw_pts))

        return junctions_metadata_groups

    def _align_coordinates(self, coordinates: np.ndarray, jids: list, junctions_metadata_groups: list):
        combined_meta = []
        group_iter = iter(junctions_metadata_groups)

        i = 0
        while i < len(coordinates):
            jid = jids[i]
            if jid is None:
                x, y, z = coordinates[i]
                _, idx = self._tree.query([x, y, z])
                closest_wp = self._wp_list[idx]
                loc = closest_wp.transform.location
                rot = closest_wp.transform.rotation
                combined_meta.append([loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw])
                i += 1
            else:
                try:
                    group_meta = next(group_iter)
                except StopIteration:
                    break

                if len(group_meta) > 0:
                    junc_xyz = group_meta[:, :3]
                    junc_seg = np.diff(junc_xyz, axis=0)
                    junc_seglen = np.linalg.norm(junc_seg, axis=1)
                    junc_arc = np.concatenate(([0.0], np.cumsum(junc_seglen)))
                    junc_prev_s = 0.0
                    junc_search = 0

                start_jid = jid
                while i < len(jids) and jids[i] == start_jid:
                    x, y, z = coordinates[i]
                    if len(group_meta) > 0:
                        pt = np.array([x, y, z], dtype=float)
                        best_s = junc_prev_s
                        best_dist2 = np.inf
                        best_seg = junc_search

                        for si in range(junc_search, len(junc_seg)):
                            A = junc_xyz[si]
                            AB = junc_seg[si]
                            ab_len2 = junc_seglen[si] ** 2

                            if ab_len2 < 1e-18:
                                tc = 0.0
                            else:
                                tc = float(np.clip(np.dot(pt - A, AB) / ab_len2, 0.0, 1.0))

                            proj = A + tc * AB
                            d2 = float(np.dot(pt - proj, pt - proj))
                            cand_s = junc_arc[si] + tc * junc_seglen[si]

                            if cand_s >= junc_prev_s - 1e-6 and d2 < best_dist2:
                                best_dist2 = d2
                                best_s = cand_s
                                best_seg = si

                        best_s = max(best_s, junc_prev_s)
                        junc_prev_s = best_s
                        junc_search = max(0, best_seg - 1)

                        seg_idx = min(np.searchsorted(junc_arc[1:], best_s), len(junc_seg) - 1)
                        local_t = (best_s - junc_arc[seg_idx]) / (junc_seglen[seg_idx] + 1e-12)
                        local_t = float(np.clip(local_t, 0.0, 1.0))

                        row_a = group_meta[seg_idx]
                        row_b = group_meta[seg_idx + 1]
                        interp_xyz = row_a[:3] + local_t * (row_b[:3] - row_a[:3])
                        
                        rpy_A, rpy_B = row_a[3:6], row_b[3:6]
                        diff_rpy = (rpy_B - rpy_A + 180) % 360 - 180
                        interp_rpy = rpy_A + local_t * diff_rpy
                        
                        combined_meta.append([float(interp_xyz[0]), float(interp_xyz[1]), float(interp_xyz[2]), 
                                              float(interp_rpy[0]), float(interp_rpy[1]), float(interp_rpy[2])])
                    else:
                        _, idx = self._tree.query([x, y, z])
                        closest_wp = self._wp_list[idx]
                        loc = closest_wp.transform.location
                        rot = closest_wp.transform.rotation
                        combined_meta.append([x, y, z, rot.roll, rot.pitch, rot.yaw])
                    i += 1

        return combined_meta

    @staticmethod
    def _project_and_interpolate(P, A, B, rpy_A, rpy_B):
        AB = B - A
        denom = float(np.dot(AB, AB))
        if denom < 1e-9:
            return A.copy(), np.array(rpy_A).copy()
        
        t = float(np.dot(P - A, AB) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        
        Q = A + t * AB
        
        rpy_A = np.array(rpy_A, dtype=float)
        rpy_B = np.array(rpy_B, dtype=float)
        
        diff = (rpy_B - rpy_A + 180) % 360 - 180
        rpy_Q = rpy_A + t * diff
        
        return Q, rpy_Q

    def _filter_and_smooth_trajectory(self, coordinates: np.ndarray, combined_meta: list, jids: list):
        filtered_meta = []
        tol = ALIGN_TOLERANCE
        for i in range(len(combined_meta)):
            closest_wp = np.array(combined_meta[i], dtype=float)[:3]
            rpy_closest = np.array(combined_meta[i], dtype=float)[3:6]

            j = i
            closest_backward = np.array(closest_wp)
            rpy_backward = rpy_closest
            while j != 0:
                closest_backward = np.array(combined_meta[j], dtype=float)[:3]
                if not np.allclose(closest_wp, closest_backward, atol=tol):
                    rpy_backward = np.array(combined_meta[j], dtype=float)[3:6]
                    break
                j -= 1
            
            k = i
            closest_forward = np.array(closest_wp)
            rpy_forward = rpy_closest
            while k < len(combined_meta) - 1:
                k += 1
                closest_forward = np.array(combined_meta[k], dtype=float)[:3]
                if not np.allclose(closest_wp, closest_forward, atol=tol):
                    rpy_forward = np.array(combined_meta[k], dtype=float)[3:6]
                    break

            choose_backward = PathHandler._edge_opposite_test(coordinates[i], closest_backward, closest_wp)
            choose_forward  = PathHandler._edge_opposite_test(coordinates[i], closest_forward, closest_wp)
            P = np.array(coordinates[i], dtype=float)

            if choose_backward and choose_forward:
                Q_bk, rpy_bk = WaypointsAlign._project_and_interpolate(P, closest_wp, closest_backward, rpy_closest, rpy_backward)
                Q_fw, rpy_fw = WaypointsAlign._project_and_interpolate(P, closest_wp, closest_forward, rpy_closest, rpy_forward)
                d_bk = np.dot(P - Q_bk, P - Q_bk)
                d_fw = np.dot(P - Q_fw, P - Q_fw)
                if d_bk < d_fw:
                    Q, rpy = Q_bk, rpy_bk
                else:
                    Q, rpy = Q_fw, rpy_fw
            elif choose_backward:
                Q, rpy = WaypointsAlign._project_and_interpolate(P, closest_wp, closest_backward, rpy_closest, rpy_backward)
            elif choose_forward:
                Q, rpy = WaypointsAlign._project_and_interpolate(P, closest_wp, closest_forward, rpy_closest, rpy_forward)
            else:
                Q = closest_wp
                rpy = rpy_closest
                
            filtered_meta += [[Q[0], Q[1], Q[2], rpy[0], rpy[1], rpy[2]]]
        
        filtered_meta = np.array(filtered_meta)
        
        N = SMOOTHING_BLEND_HALF_WINDOW
        boundaries = [b for b in range(1, len(jids)) if (jids[b] is None) != (jids[b-1] is None)]
        
        for b in boundaries:
            start = max(0, b - N)
            end = min(len(filtered_meta), b + N)
            if end - start < 3: 
                continue
            
            window_size = SMOOTHING_WINDOW_SIZE
            pad = window_size // 2
            
            for _ in range(2):
                window = filtered_meta[start:end, :3]
                new_window = np.copy(window)
                for dim in range(3):
                    padded = np.pad(window[:, dim], (pad, pad), mode='edge')
                    new_window[:, dim] = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
                filtered_meta[start:end, :3] = new_window

        return filtered_meta

    def spatial_align(self, coordinates: np.ndarray, headings: np.ndarray):
        jids, junction_by_id = self._filter_valid_junctions(coordinates)
        groups = self._group_coordinates_by_junction(coordinates, headings, jids)
        
        junctions_metadata_groups = self._precompute_junction_paths(groups, junction_by_id)
        combined_meta = self._align_coordinates(coordinates, jids, junctions_metadata_groups)
        return self._filter_and_smooth_trajectory(coordinates, combined_meta, jids)
    
    def temporal_align(self, trajectories: np.ndarray, time_vect: np.ndarray):

        num_original = len(time_vect)
        num_filtered = len(trajectories)
        
        original_indices = np.linspace(0, num_original - 1, num_original)
        filtered_indices = np.linspace(0, num_original - 1, num_filtered)
        temporal_aligned = np.zeros((num_filtered, 7))
        temporal_aligned[:, :6] = trajectories[:, :6]
        temporal_aligned[:, 6] = np.interp(filtered_indices, original_indices, time_vect)
        return temporal_aligned

    
    def align(self, trajectories: np.ndarray):
        spatial_aligned = self.spatial_align(trajectories[:, :3], trajectories[:, 4: ])

        original_time = trajectories[:, 3]
        temporal_aligned = self.temporal_align(spatial_aligned, original_time)

        return spatial_aligned, temporal_aligned

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

def _find_entry_clusters(wp_pairs, waypoints):
    
    best_dist = float("inf")
    best_entry = None
    best_loc = None
    best_exit = None

    for entry_wp, exit_wp in wp_pairs:
        loc = entry_wp.transform.location
        entry_xyz = np.array([loc.x, loc.y, loc.z])
        dists = np.linalg.norm(waypoints - entry_xyz, axis=1)
        min_d = dists.min()
        if min_d < best_dist:
            best_dist = min_d
            best_entry = entry_wp
            best_exit = exit_wp
            best_loc = entry_xyz

    if best_entry is None:
        return []

    cluster = []
    for entry_wp, exit_wp in wp_pairs:
        loc = entry_wp.transform.location
        loc_xyz = np.array([loc.x, loc.y, loc.z])
        if np.allclose(loc_xyz, best_loc, atol=1e-6):  # exact same point
            cluster.append((entry_wp, exit_wp))
            
    return cluster

def _find_exit(wp_pairs, waypoints, trajectory_dir=None):
    """
    Find the exit waypoint pair closest to waypoints.
    If trajectory_dir (2D unit vector) is provided, exits whose heading
    doesn't match the trajectory direction receive a distance penalty,
    preventing overshoot from selecting the wrong roundabout exit.
    """
    best_score = float("inf")
    best_entry = None
    best_loc = None
    best_exit = None

    for entry_wp, exit_wp in wp_pairs:
        loc = exit_wp.transform.location
        exit_xyz = np.array([loc.x, loc.y, loc.z])
        dists = np.linalg.norm(waypoints - exit_xyz, axis=1)
        min_d = dists.min()

        score = min_d
        if trajectory_dir is not None:
            exit_fwd = exit_wp.transform.get_forward_vector()
            exit_dir = np.array([exit_fwd.x, exit_fwd.y])
            exit_norm = np.linalg.norm(exit_dir)
            if exit_norm > 1e-6:
                exit_dir /= exit_norm
                cos_sim = np.dot(trajectory_dir, exit_dir)
                # cos_sim: 1 = same dir, -1 = opposite
                # Penalty: 0 for perfect match, up to dir_weight for opposite
                dir_weight = 5.0
                score += dir_weight * (1.0 - cos_sim) / 2.0

        if score < best_score:
            best_score = score
            best_entry = entry_wp
            best_exit = exit_wp
            best_loc = exit_xyz

    return best_entry, best_exit

def get_fwd_vec(pitch_deg, yaw_deg):
    """Converts CARLA Euler angles (degrees) to a normalized forward vector."""
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    
    fx = np.cos(p) * np.cos(y)
    fy = np.cos(p) * np.sin(y)
    fz = np.sin(p)
    return np.array([fx, fy, fz])

def _unified_entry_exit_finder(wp_pairs, wp_metadata: np.ndarray):
    # waypoints include location and may or may not have rotation (use cos sim)
    traj_start_loc = wp_metadata[0, :3]
    traj_end_loc   = wp_metadata[-1, :3]
    
    # Calculate vehicle heading at start and end
    traj_start_fwd = get_fwd_vec(wp_metadata[0, 4], wp_metadata[0, 5])
    traj_end_fwd   = get_fwd_vec(wp_metadata[-1, 4], wp_metadata[-1, 5])

    best_pair = None
    min_score = float('inf')
    
    for entry_wp, exit_wp in wp_pairs:
        en_loc = np.array([entry_wp.transform.location.x, 
                           entry_wp.transform.location.y, 
                           entry_wp.transform.location.z])
        ex_loc = np.array([exit_wp.transform.location.x, 
                           exit_wp.transform.location.y, 
                           exit_wp.transform.location.z])
        
        en_fwd_carla = entry_wp.transform.get_forward_vector()
        en_fwd = np.array([en_fwd_carla.x, en_fwd_carla.y, en_fwd_carla.z])
        
        ex_fwd_carla = exit_wp.transform.get_forward_vector()
        ex_fwd = np.array([ex_fwd_carla.x, ex_fwd_carla.y, ex_fwd_carla.z])
        
        dist_score = np.linalg.norm(en_loc - traj_start_loc) + \
                     np.linalg.norm(ex_loc - traj_end_loc)
                     
        start_align = np.dot(en_fwd, traj_start_fwd)
        end_align   = np.dot(ex_fwd, traj_end_fwd)
        
        orientation_penalty = (1.0 - start_align) + (1.0 - end_align)
        
        total_score = dist_score + (orientation_penalty * 5.0)

        if total_score < min_score:
            min_score = total_score
            best_pair = (entry_wp, exit_wp)
            
    return best_pair, min_score
        
def waypoint_heading(wp):
    fwd = wp.transform.get_forward_vector()
    yaw = np.arctan2(fwd.y, fwd.x)
    return yaw

def waypoints_between(entry_wp, exit_wp, step=0.5):
    """
    Returns a list of waypoints between entry and exit inside a junction.
    Ensures inclusion of entry and exit waypoint
    
    entry_wp, exit_wp : carla.Waypoint
    step : float
        Distance between waypoints when iterating.
    """
    wps = [entry_wp]
    current_wp = entry_wp

    while current_wp.transform.location.distance(exit_wp.transform.location) > step:
        next_wps = current_wp.next(step)
        if not next_wps:
            break
        # choose the next waypoint closest to the exit
        current_wp = min(next_wps, key=lambda wp: wp.transform.location.distance(exit_wp.transform.location))
        wps.append(current_wp)
        if current_wp.id == exit_wp.id:
            break

    # ensure exit is included
    if wps[-1].id != exit_wp.id:
        wps.append(exit_wp)

    return wps

def carla_waypoints_to_np(waypoints):
    if hasattr(waypoints, 'transform'):
        wp_list = [waypoints]  # Wrap single waypoint in a list
    else:
        wp_list = list(waypoints)  # Assume iterable of waypoints
    arr = np.array([[getattr(wp.transform.location, dim) for dim in ['x', 'y', 'z']] for wp in wp_list])
    return arr

class OptimizePath:
    """
    Generates an optimized topological spatial hashing grid structure for fast pathway routing.
    
    Parses complex CARLA waypoints map into custom NetworkX sub-graphs heavily parameterized for quick
    spatial lookups. By deploying customized grid discretization, it achieves O(1) node retrievals to 
    expedite A* routing operations efficiently across continuous topology blocks.
    """
    def __init__(self, world, step: float, exclude_circle: tuple = None):
        self.log = Logger()
        self.virt_world = world
        self.exclude_params = exclude_circle # (cx, cy, radius)
        self.exclude_params = [0, 0, 0]
        
        # Initialize spatial hash grid for O(1) node lookups during building
        self._spatial_hash_grid = {}
        self._grid_size = 1.0  # Grid cell size for hashing

        carla_map = world.world.get_map()
        self.network, self.nodes = self._build_detailed_graph(carla_map, step=step, epsilon=0.1)

        self.log.INFO(f"Built Road network. Nodes: {len(self.network.nodes)}")      

    @staticmethod
    def fast_extract_coordinates(network_or_nodes):
        # Handle NetworkX graph
        if hasattr(network_or_nodes, 'nodes') and hasattr(network_or_nodes.nodes, 'data'):
            nodes_data = network_or_nodes.nodes(data=True)
            if not nodes_data:
                return {}
            
            coord_dict = {}
            
            for nid, attrs in nodes_data:
                if 'pos' in attrs:
                    coord_dict[nid] = attrs['pos']
                elif 'x' in attrs and 'y' in attrs:
                    if 'z' in attrs:
                        coord_dict[nid] = (attrs['x'], attrs['y'], attrs['z'])
                    else:
                        coord_dict[nid] = (attrs['x'], attrs['y'])
                else:
                    raise ValueError(f"Node {nid} has no coordinate data")
            
            return coord_dict
        
        # Handle dictionary
        elif isinstance(network_or_nodes, dict):
            return dict(network_or_nodes)  # Return copy of the dictionary
        
        else:
            raise TypeError("Input must be NetworkX graph or dictionary")

    def _find_or_create_node(self, nodes, loc, epsilon=0.1):
        """Find existing node within epsilon distance or create new one.
        Uses spatial hash grid for O(1) average lookup during building.
        """
        x, y, z = loc.x, loc.y, loc.z
        
        # Compute grid cell coordinates
        grid_x = int(x / self._grid_size)
        grid_y = int(y / self._grid_size)
        grid_z = int(z / self._grid_size)
        
        # Check nearby grid cells (3x3x3 neighborhood for safety)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    cell_key = (grid_x + dx, grid_y + dy, grid_z + dz)
                    if cell_key in self._spatial_hash_grid:
                        # Check all nodes in this cell
                        for nid in self._spatial_hash_grid[cell_key]:
                            node_x, node_y, node_z = nodes[nid]
                            # Use 3D distance
                            dx_node = x - node_x
                            dy_node = y - node_y
                            dz_node = z - node_z
                            if (dx_node*dx_node + dy_node*dy_node + dz_node*dz_node) ** 0.5 < epsilon:
                                return nid
        
        # Create new node
        nid = len(nodes)
        nodes[nid] = (x, y, z)
        
        # Add to spatial hash grid
        cell_key = (grid_x, grid_y, grid_z)
        if cell_key not in self._spatial_hash_grid:
            self._spatial_hash_grid[cell_key] = []
        self._spatial_hash_grid[cell_key].append(nid)
        
        return nid
    
    def _is_inside_circle(self, wp):
        """Checks if a single waypoint is inside the forbidden circle."""
        if not self.exclude_params:
            return False
        cx, cy, radius = self.exclude_params
        loc = wp.transform.location
        return math.hypot(loc.x - cx, loc.y - cy) <= radius

    def _build_detailed_graph(self, cmap, step=3.0, epsilon=0.1):
        G = nx.DiGraph()
        nodes = {}
        
        topology = cmap.get_topology()
        edges_to_add = []  # Batch edges for efficient addition
        
        for start_wp, end_wp in topology:
            wp_list = start_wp.next_until_lane_end(step)
            wp_list = [start_wp] + wp_list
            if end_wp not in wp_list:
                wp_list.append(end_wp)

            # Check if segment is invalid
            is_segment_invalid = False
            if self.exclude_params:
                for wp in wp_list:
                    if self._is_inside_circle(wp):
                        is_segment_invalid = True
                        break
            
            if is_segment_invalid:
                continue

            prev_id = None
            for wp in wp_list:
                nid = self._find_or_create_node(nodes, wp.transform.location, epsilon)
                
                # Only add node if new
                if nid not in G.nodes:
                    G.add_node(nid, x=nodes[nid][0], y=nodes[nid][1], z=nodes[nid][2], pos=nodes[nid])
                
                if prev_id is not None and prev_id != nid:
                    edges_to_add.append((prev_id, nid))
                prev_id = nid
        
        # Batch add all edges at once for efficiency
        if edges_to_add:
            G.add_edges_from(edges_to_add)

        return G, nodes

    def update_coordinates(self, new_coords_dict):
        for nid, coords in new_coords_dict.items():
            if nid in self.network.nodes:
                self.network.nodes[nid]['x'] = coords[0]
                self.network.nodes[nid]['y'] = coords[1]
                if len(coords) > 2:
                    self.network.nodes[nid]['z'] = coords[2]
                self.network.nodes[nid]['pos'] = coords

    def get_positions(self):
        return {nid: data.get('pos', (0, 0, 0)) for nid, data in self.network.nodes(data=True)}

    def bfs_shortest_path(self, start, goal):
        from collections import deque
        
        if start not in self.network.nodes or goal not in self.network.nodes:
            self.log.ERROR(f"Start node {start} or goal node {goal} not found in network")
            return None
            
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited = set()
        
        while queue:
            node, path = queue.popleft()
            
            if node == goal:
                self.log.INFO(f"Found path from {start} to {goal} with {len(path)} nodes")
                return path
                
            if node not in visited:
                visited.add(node)
                
                # Get neighbors from NetworkX graph
                for neighbor in self.network.neighbors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        self.log.WARNING(f"No path found from {start} to {goal}")
        return None

    def get_path_coordinates(self, path):
        if not path:
            return np.array([]).reshape(0, 2)
            
        coords = []
        for node_id in path:
            if node_id in self.nodes:
                coords.append(self.nodes[node_id])
            else:
                self.log.WARNING(f"Node {node_id} not found in coordinates")
                
        return np.array(coords)
    
    def find_nearest_node(self, position, max_distance=50.0):
        coords_dict = self.fast_extract_coordinates(self.network)
        if not coords_dict:
            return None

        pos_array = np.array(position, dtype=float)

        min_dist = float('inf')
        nearest_node = None

        for node_id, coords in coords_dict.items():
            node_pos = np.array(coords[:len(pos_array)])
            dist = np.linalg.norm(pos_array - node_pos)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest_node = node_id

        return nearest_node

    def plan_path(self, start_pos, goal_pos, max_search_distance=50.0):
        start_node = self.find_nearest_node(start_pos, max_search_distance)
        goal_node = self.find_nearest_node(goal_pos, max_search_distance)
        
        if start_node is None:
            self.log.ERROR(f"No start node found near position {start_pos}")
            return None, None
            
        if goal_node is None:
            self.log.ERROR(f"No goal node found near position {goal_pos}")
            return None, None
        
        path_nodes = self.bfs_shortest_path(start_node, goal_node)
        
        if path_nodes is None:
            return None, None
            
        path_coords = self.get_path_coordinates(path_nodes)
        
        self.log.INFO(f"Planned path with {len(path_nodes)} nodes, total distance: {self._calculate_path_distance(path_coords):.2f}m")
        
        return path_nodes, path_coords

    def _calculate_path_distance(self, path_coords):
        """Calculate total distance of a path."""
        if len(path_coords) < 2:
            return 0.0
            
        diffs = np.diff(path_coords, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    
    def find_distant_nodes(self, position, min_distance, max_distance=None):
        coords_dict = self.fast_extract_coordinates(self.network)
        if not coords_dict:
            return []

        pos_array = np.array(position, dtype=float)
        distant_nodes = []

        for node_id, coords in coords_dict.items():
            node_pos = np.array(coords[:len(pos_array)])
            dist = np.linalg.norm(pos_array - node_pos)
            
            if dist >= min_distance:
                if max_distance is None or dist <= max_distance:
                    distant_nodes.append((node_id, dist, coords))

        # Sort by distance
        distant_nodes.sort(key=lambda x: x[1])
        return distant_nodes
    
    def debug_draw_network(self, world, life_time=10.0):
        debug = world.debug
        for u, v in self.network.edges():
            pos_u = self.nodes[u]
            pos_v = self.nodes[v]
            loc_u = carla.Location(x=pos_u[0], y=pos_u[1], z=2.0)
            loc_v = carla.Location(x=pos_v[0], y=pos_v[1], z=2.0)
            debug.draw_line(loc_u, loc_v, thickness=0.1, 
                            color=carla.Color(0, 255, 0), life_time=life_time)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    client = carla.Client("localhost", 2000)
    world = World(client, 10000)
    
    optimizer = OptimizePath(world, step=2.0, exclude_circle=(322.5, -195.5, 17))
    optimizer.debug_draw_network(world.world)