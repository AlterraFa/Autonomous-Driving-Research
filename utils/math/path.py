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

from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from utils.others.data_processor import CarlaDatasetCollector
from utils.math.coordinate_transform import global_2_local
from utils.messages.logger import Logger
from utils.control.world import World

from numba import njit

conf = toml.load(os.path.join(parent, "../config/config.toml"))
temporal_offset      = conf['Offsets']['temporal_offset']
spatial_offset       = conf['Offsets']['spatial_offset']
scout_offset_params  = conf['Offsets']['scout_offset_params']
front_vehicle_offset = conf['Offsets']['front_vehicle_offset']

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
        dists, idxs = self.kdtree.query(p, k=len(self.path), distance_upper_bound=2*self.Ld)
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
    defined_path: 
      (N,3) -> [x, y, z]
      (N,4) -> [x, y, z, t]   (t = delta time recording)
    """
    def __init__(self, defined_path: np.ndarray, extrapolate: bool = True):
        self.log = Logger()
        super().__init__(10, defined_path[:, :3])

        assert defined_path.ndim == 2 and defined_path.shape[1] in (3, 4), \
            "defined_path must be (N,3) [x,y,z] or (N,4) [x,y,z,t]"
        
        self.path_xyz = defined_path[:, :3].astype(float)
        self.has_time = defined_path.shape[1] == 4

        # --- arc-length for projection ---
        diffs   = np.diff(self.path_xyz, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        s       = np.concatenate(([0.0], np.cumsum(seg_len)))
        keep    = np.r_[True, seg_len >= 0]

        # Prevent multiple points with the same distance for interpolation
        eps   = 1e-6
        count = 0
        for i in range(1, len(s)):
            if seg_len[i-1] == 0:
                count += 1
                s[i] += eps * count
            else:
                count = 0

        self.path_xyz = self.path_xyz[keep]
        self.s = s[keep]
        self.seg_vec = np.diff(self.path_xyz, axis=0)
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)

        # --- interpolation in s ---
        self.x_of_s = interp1d(self.s, self.path_xyz[:, 0], kind="linear",
                               bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 0], self.path_xyz[-1, 0]))
        self.y_of_s = interp1d(self.s, self.path_xyz[:, 1], kind="linear",
                               bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 1], self.path_xyz[-1, 1]))
        self.z_of_s = interp1d(self.s, self.path_xyz[:, 2], kind="linear",
                               bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 2], self.path_xyz[-1, 2]))

        # --- interpolation in t if available ---
        if self.has_time:
            self.log.DEBUG("Found time vector. Enabling spatial and temporal mode")
            
            self.timer = 0
            t_col = defined_path[:, -1].astype(float)[keep]
            self.t = np.cumsum(t_col)
            
            self.x_of_t = interp1d(self.t, self.path_xyz[:, 0], kind="linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 0], self.path_xyz[-1, 0]))
            self.y_of_t = interp1d(self.t, self.path_xyz[:, 1], kind="linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 1], self.path_xyz[-1, 1]))
            self.z_of_t = interp1d(self.t, self.path_xyz[:, 2], kind="linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (self.path_xyz[0, 2], self.path_xyz[-1, 2]))
            self.t_of_s = interp1d(self.s, self.t, kind = "linear",
                                   bounds_error=False, fill_value="extrapolate" if extrapolate else (t_col[0], t_col[-1]))

        else:
            self.log.DEBUG("Didn't find time vector. Disabling temporal mode")

            self.t = None
            
    
    def pose(self, query: float, use_time: bool = False):
        """
        Return [x, y, z].
        If use_time=False -> query is arc-length s.
        If use_time=True  -> query is timestamp t (requires time column).
        """
        if use_time:
            if self.t is None:
                raise RuntimeError("This path has no time column.")
            return np.stack([
                self.x_of_t(query),
                self.y_of_t(query),
                self.z_of_t(query)
            ], axis = -1)
        else:
            return np.stack([
                self.x_of_s(query),
                self.y_of_s(query),
                self.z_of_s(query)
            ], axis = -1)

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
            wp = self.pose(dist_offset)
        else: 
            if self.has_time == False:
                self.log.ERROR(f"Temporal mode was disabled but you enabled `use_time` argument. Exiting...", exit_code = -1)
                
            current_time = self.t_of_s(dist_travelled)
            time_offset = current_time + offsets
            wp = self.pose(time_offset, True)

        if merge:
            wp = self._apply_merging(wp, side_val, offsets)
                
        wp = np.asarray(wp)
        return wp

    def _apply_merging(self, centerline_wps: np.ndarray, current_side_val: float, offsets: list[float], merge_fraction: float = 1.0):
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
                tangent = centerline_wps[i+1] - p_center
            else:
                tangent = p_center - centerline_wps[i-1]

            up = np.array([0, 0, 1]) 
            right_vec = np.cross(tangent, up)
            
            norm = np.linalg.norm(right_vec)
            if norm > 1e-6:
                right_vec /= norm
                p_merged = p_center + (right_vec * active_offset)
            else:
                p_merged = p_center
                
            merged_pts.append(p_merged)

        return merged_pts
        
class BranchingPath:
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
        backward_point = closest_wp
        while j != 0:
            backward_point = waypoints[j]
            if not np.allclose(closest_wp, backward_point, atol = 1e-2):
                break
            j -= 1
            
        k = closest_idx
        forward_point = np.array(closest_wp)
        while k != len(waypoints):
            forward_point = waypoints[k]
            if not np.allclose(closest_wp, forward_point, atol = 1e-2):
                break
            k += 1
            
        choose_backward = PathHandler._edge_opposite_test(compare_point, backward_point, closest_wp)
        choose_forward  = PathHandler._edge_opposite_test(compare_point, forward_point, closest_wp)
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
                
class TurnClassify:
    def __init__(self, world: World, threshold_deg: float = 45):
        self.thresh_deg = threshold_deg
        self.signal = None
        self.virt_world  = world
        pass



    def turning_type(self, enable: bool, junction, disable: bool, waypoints: np.ndarray, debug = False):
        """
        Classify the vehicle's maneuver through a junction as straight, left, or right
        based on the heading change between the closest entry and exit waypoints.

        Parameters
        ----------
        enable : bool
            If True, perform classification. When enabled, the method will:
            1. Get all (entry, exit) waypoint pairs from the junction.
            2. Find the entry cluster closest to the vehicle's current path waypoints.
            3. Select the exit waypoint from that cluster that is closest to the path.
            4. Compute the heading (yaw) of both the chosen entry and exit waypoints.
            5. Calculate the wrapped heading difference Δ (radians) using atan2.
            6. Classify the maneuver:
                self.signal = 0 → straight  (|Δ| < thresh_deg)
                self.signal = 1 → right turn (Δ < -thresh_deg)
                self.signal = 2 → left turn  (Δ > +thresh_deg)

        junction : carla.Junction
            The CARLA junction object obtained from a waypoint's `.get_junction()` call.
            Must contain driving lane waypoints.

        disable : bool
            If True and `enable` is False, reset classification state by setting
            `self.signal = -1`.

        waypoints : np.ndarray, shape (N,3)
            Array of vehicle trajectory points [x, y, z] used to determine which
            entry/exit pair is closest to the current path.

        thresh_deg : float, optional (default=45)
            Angular threshold in degrees to decide what counts as "straight".
            Turns smaller than this threshold are treated as going straight.

        Returns
        -------
        signal : int
            -1 if disabled/reset,
            0 if straight maneuver,
            1 if right turn,
            2 if left turn.

        Notes
        -----
        - This method uses only entry/exit waypoint heading difference, so small
        zig-zags or lane curvature inside the junction will still be classified
        correctly by the net heading change.
        - Uses CARLA debug draw to visualize the chosen entry (blue point) and
        exit (blue point) locations for one frame at 70 FPS.
        """
        if enable:
            wp_pairs       = junction.get_waypoints(carla.LaneType.Driving)
            possible_pairs = _find_entry_clusters(wp_pairs, waypoints)                    
            choosen_pairs  = _find_exit(possible_pairs, waypoints)

            if debug:
                self.virt_world.debug.draw_point(choosen_pairs[0].transform.location, size = 0.18, color = carla.Color(0, 0, 255), life_time = 1.5 * (1 / 70))
                self.virt_world.debug.draw_point(choosen_pairs[1].transform.location, size = 0.18, color = carla.Color(0, 0, 255), life_time = 1.5 * (1 / 70))
            
            entry_heading  = waypoint_heading(choosen_pairs[0])
            exit_heading   = waypoint_heading(choosen_pairs[1])

            delta = np.arctan2(np.sin(exit_heading - entry_heading),
                       np.cos(exit_heading - entry_heading))
            
            if abs(delta) < np.radians(self.thresh_deg):
                self.signal = 0
            elif delta < 0:
                self.signal = 1
            else:
                self.signal = 2
        if disable and not enable:
            self.signal = -1

        return self.signal

from utils.messages.message_handler import MessageSubscriber, MessageSender
from utils.messages.all_messages import (
    Location,
    Heading,
    ServerFps,
    ClientFps,
    SteerLog,
    Velocity,
    BrakeLog,
    ThrottleLog,
    TurnSignal
)
class ReplayHandler:

    turn_classify = False
    __slot__ = ["road_type"]
    
    def __init__(self, world: World, true_trajectories: str, midlane_waypoints: np.ndarray = None, data_collect_dir: str = None, use_temporal: bool = False, debug: bool = False):
        self.logger = Logger()
        
        self.true_trajectories = true_trajectories
        self.path_handler = PathHandler(midlane_waypoints)
        self.debug = debug
        self.virt_world = world
        self.use_temporal = use_temporal
        self.scout_points = [i for i in range(*scout_offset_params)]
        if not self.use_temporal:
            self.offset   = spatial_offset
        else:
            self.offset   = temporal_offset
        self.turn_classifier = TurnClassify(world=world, threshold_deg=15)
        self.branching_path  = BranchingPath(self.virt_world)
        self.data_collector = None
        if data_collect_dir:
            self.data_collector = CarlaDatasetCollector(save_dir=data_collect_dir, save_interval=20)

        self.prev_dist = 0
        self.additional_max = 20; self.addition_cnt = 0
        self.start_time = time.time()
        self._init_transmittor()

    def _init_transmittor(self):
        self.sub_location = MessageSubscriber(Location)
        self.sub_heading  = MessageSubscriber(Heading)
        self.sub_server_fps = MessageSubscriber(ServerFps)
        self.sub_client_fps = MessageSubscriber(ClientFps)
        self.sub_steer_logging = MessageSubscriber(SteerLog)
        self.sub_throttle_logging = MessageSubscriber(ThrottleLog)
        self.sub_brake_logging    = MessageSubscriber(BrakeLog)
        self.sub_velocity         = MessageSubscriber(Velocity)
        self.send_turn_signal     = MessageSender(TurnSignal)
        

    def step(self, **frame: np.ndarray):
        vehicle_location = self.sub_location.receive()

        # Convert yaw from degrees to radians for math functions
        heading  = np.radians(self.sub_heading.receive())

        # Distance from the center to the front of the car (adjust as per your vehicle)
        front_offset = front_vehicle_offset  # meters

        # Calculate offset in x and y directions
        offset_x = front_offset * np.cos(heading - np.pi / 2)
        offset_y = front_offset * np.sin(heading - np.pi / 2)

        # Calculate front location coordinates
        position = np.array([
            vehicle_location[0] + offset_x,
            vehicle_location[1] + offset_y,
            vehicle_location[2]  # same height as center
        ])
        server_fps = self.sub_server_fps.receive()
        if server_fps < 1: server_fps = self.sub_client_fps.receive()
        
        
        
        curr_dist, *_ = self.path_handler.project(position)
        mid_global_scout = self.path_handler.waypoints(
            position, self.scout_points
        )
        
        mid_global = self.path_handler.waypoints(
            position, self.offset, use_time = self.use_temporal, merge = False
        )
        mid_ego = global_2_local(vehicle_location, mid_global, heading)
        
        path_branches  = self.branching_path.brancher(mid_global, mid_global_scout, persist_dist = 20)
        ego_branches = np.empty_like(path_branches)[..., :2]
        if path_branches.shape[0] > 1:
            self.road_type = "multi"
        else:
            self.road_type = "uni"
        for idx, branch in enumerate(path_branches):
            ego_branches[idx] = global_2_local(vehicle_location, branch, heading)
                
                
        if self.debug:
            for path in path_branches:
                self.virt_world.draw_waypoints(path, 1.5 * (1 / server_fps), size = .1, color = (255, 0, 0))

        if self.turn_classify:
            global_scout = self.path_handler.waypoints(
                position, self.scout_points
            )
            is_at_junction, junction = self.virt_world.get_waypoint_junction(global_scout[14])
            not_exit_junction, _ = self.virt_world.get_waypoint_junction(global_scout[10])
            is_exit_junction = not not_exit_junction
            turn_signal = self.turn_classifier.turning_type(is_at_junction, junction, is_exit_junction, global_scout)
        else:
            turn_signal = -1
        self.send_turn_signal.send(turn_signal)

        # Only save when it moves (Prevent saving all the time when stopping at red light or stop sign)
        if self.data_collector:
            if self.addition_cnt < self.additional_max:
                steer    = self.sub_steer_logging.receive()
                throttle = self.sub_throttle_logging.receive()
                brake    = self.sub_brake_logging.receive()
                velocity = self.sub_velocity.receive()
                saved = self.data_collector.maybe_save(
                    {
                        "midlane_wp" : mid_ego,
                        "aux_wp"     : ego_branches,
                        "steer"      : steer,
                        "throttle"   : throttle,
                        "brake"      : brake,
                        "velocity"   : velocity,
                        "turn_signal": turn_signal,
                        "GPS"        : vehicle_location,
                        "IMU_radian" : heading,
                        "road_type"  : self.road_type,
                        "timestamp"  : time.time() - self.start_time
                    },
                    **frame
                )
                if saved:
                    if curr_dist - self.prev_dist < 1e-2:
                        self.addition_cnt += 1
            if curr_dist - self.prev_dist > 1e-2:
                self.addition_cnt = 0
            self.prev_dist = curr_dist
        return mid_ego


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

def _find_exit(wp_pairs, waypoints):
    
    best_dist = float("inf")
    best_entry = None
    best_loc = None
    best_exit = None

    for entry_wp, exit_wp in wp_pairs:
        loc = exit_wp.transform.location
        exit_xyz = np.array([loc.x, loc.y, loc.z])
        dists = np.linalg.norm(waypoints - exit_xyz, axis=1)
        min_d = dists.min()
        if min_d < best_dist:
            best_dist = min_d
            best_entry = entry_wp
            best_exit = exit_wp
            best_loc = exit_xyz

    return best_entry, best_exit

def waypoint_heading(wp):
    fwd = wp.transform.get_forward_vector()
    yaw = np.arctan2(fwd.y, fwd.x)
    return yaw

def waypoints_between(entry_wp, exit_wp, step=1.0):
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
    def __init__(self, world, step: float, exclude_circle: tuple = None):
        self.log = Logger()
        self.virt_world = world
        self.exclude_params = exclude_circle # (cx, cy, radius)

        carla_map = world.world.get_map()
        self.network, self.nodes = self._build_detailed_graph(carla_map, step=step, epsilon=0.1)

        self.log.DEBUG(f"Built Road network. Nodes: {len(self.network.nodes)}")   

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
                    coord_dict[nid] = (attrs['x'], attrs['y'])
                else:
                    raise ValueError(f"Node {nid} has no coordinate data")
            
            return coord_dict
        
        # Handle dictionary
        elif isinstance(network_or_nodes, dict):
            return dict(network_or_nodes)  # Return copy of the dictionary
        
        else:
            raise TypeError("Input must be NetworkX graph or dictionary")

    @staticmethod
    def _find_or_create_node(nodes, loc, epsilon=0.1):
        for nid, (x, y) in nodes.items():
            if math.hypot(loc.x - x, loc.y - y) < epsilon:
                return nid
        nid = len(nodes)
        nodes[nid] = (loc.x, loc.y)
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

        for start_wp, end_wp in topology:
            wp_list = start_wp.next_until_lane_end(step)
            wp_list = [start_wp] + wp_list
            if end_wp not in wp_list:
                wp_list.append(end_wp)

            is_segment_invalid = False
            if self.exclude_params:
                for wp in wp_list:
                    if self._is_inside_circle(wp):
                        is_segment_invalid = True
                        break # One bad point spoils the whole segment
            
            if is_segment_invalid:
                continue

            prev_id = None
            for wp in wp_list:
                nid = self._find_or_create_node(nodes, wp.transform.location, epsilon)
                
                if nid not in G.nodes:
                    G.add_node(nid, x=nodes[nid][0], y=nodes[nid][1], pos=nodes[nid])
                
                if prev_id is not None and prev_id != nid:
                    G.add_edge(prev_id, nid)
                prev_id = nid

        return G, nodes

    def update_coordinates(self, new_coords_dict):
        for nid, (x, y) in new_coords_dict.items():
            if nid in self.network.nodes:
                self.network.nodes[nid]['x'] = x
                self.network.nodes[nid]['y'] = y  
                self.network.nodes[nid]['pos'] = (x, y)

    def get_positions(self):
        return {nid: data.get('pos', (0, 0)) for nid, data in self.network.nodes(data=True)}

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

        for node_id, (x, y) in coords_dict.items():
            node_pos = np.array([x, y])
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

        for node_id, (x, y) in coords_dict.items():
            node_pos = np.array([x, y])
            dist = np.linalg.norm(pos_array - node_pos)
            
            if dist >= min_distance:
                if max_distance is None or dist <= max_distance:
                    distant_nodes.append((node_id, dist, (x, y)))

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