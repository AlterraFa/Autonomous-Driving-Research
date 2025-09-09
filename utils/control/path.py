import numpy as np
import carla
from scipy.interpolate import interp1d
from utils.messages.message_handler import MessagingSenders, MessagingSubscribers
from utils.data_processor import CarlaDatasetCollector

def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi
    
class NodeFinder:
    def __init__(self, Ld, path, update_dist = .5, **kwargs):
        super().__init__(**kwargs)
        self.Ld = Ld
        self.position_idx = 0
        self.path = path
        self.update_dist = update_dist

    def update_state(self, distance):

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
class PathHandler(NodeFinder):
    """
    defined_path: 
      (N,3) -> [x, y, z]
      (N,4) -> [x, y, z, t]   (t = delta time recording)
    """
    def __init__(self, defined_path: np.ndarray):
        super().__init__(10, defined_path)

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
                               bounds_error=False, fill_value="extrapolate")
        self.y_of_s = interp1d(self.s, self.path_xyz[:, 1], kind="linear",
                               bounds_error=False, fill_value="extrapolate")
        self.z_of_s = interp1d(self.s, self.path_xyz[:, 2], kind="linear",
                               bounds_error=False, fill_value="extrapolate")

        # --- interpolation in t if available ---
        if self.has_time:
            self.timer = 0
            t_col = defined_path[:, -1].astype(float)[keep]
            self.t = np.cumsum(t_col)
            
            self.x_of_t = interp1d(self.t, self.path_xyz[:, 0], kind="linear",
                                   bounds_error=False, fill_value="extrapolate")
            self.y_of_t = interp1d(self.t, self.path_xyz[:, 1], kind="linear",
                                   bounds_error=False, fill_value="extrapolate")
            self.z_of_t = interp1d(self.t, self.path_xyz[:, 2], kind="linear",
                                   bounds_error=False, fill_value="extrapolate")
            self.t_of_s = interp1d(self.s, self.t, kind = "linear",
                                   bounds_error=False, fill_value="extrapolate")

        else:
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
            return np.array([
                float(self.x_of_t(query)),
                float(self.y_of_t(query)),
                float(self.z_of_t(query))
            ])
        else:
            return np.array([
                float(self.x_of_s(query)),
                float(self.y_of_s(query)),
                float(self.z_of_s(query))
            ])

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

    def waypoints(self, position: np.ndarray, offsets: list[float], yaw: float, use_time: bool = False, return_global = False):
        dist_travelled, *_ = self.project(position)
        if not use_time:
            wp = []
            for offset in offsets:
                wp += [self.pose(dist_travelled + 2 + offset)]
        else: 
            current_time = self.t_of_s(dist_travelled)
            wp = []
            for offset in offsets:
                wp += [self.pose(current_time + offset, True)]
                
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
    def __init__(self, world, threshold_deg: float = 45):
        self.thresh_deg = threshold_deg
        self.signal = None
        self.world  = world
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
    def waypoint_heading(wp):
        fwd = wp.transform.get_forward_vector()
        yaw = np.arctan2(fwd.y, fwd.x)
        return yaw


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
            possible_pairs = self._find_entry_clusters(wp_pairs, waypoints)                    
            choosen_pairs  = self._find_exit(possible_pairs, waypoints)

            if debug:
                self.world.debug.draw_point(choosen_pairs[0].transform.location, size = 0.18, color = carla.Color(0, 0, 255), life_time = 1.5 * (1 / 70))
                self.world.debug.draw_point(choosen_pairs[1].transform.location, size = 0.18, color = carla.Color(0, 0, 255), life_time = 1.5 * (1 / 70))
            
            entry_heading  = self.waypoint_heading(choosen_pairs[0])
            exit_heading   = self.waypoint_heading(choosen_pairs[1])

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

class ReplayHandler(MessagingSubscribers, MessagingSenders):
    def __init__(self, replay_file: str, world, data_collect_dir: str = None, use_temporal: bool = False, debug: bool = False):
        MessagingSubscribers.__init__(self)
        MessagingSenders.__init__(self)
        
        waypoints_storage = np.load(replay_file)
        self.path_handler = PathHandler(waypoints_storage)
        self.debug = debug
        self.world = world
        self.use_temporal = use_temporal
        self.scout_points = [i for i in range(-18, 33, 2)]
        if not self.use_temporal:
            self.offset   = [1, 3, 5, 7, 9]
        else:
            self.offset   = [.2, .4, .6, .8, 1.0]
        self.turn_classifier = TurnClassify(world=world, threshold_deg=15)
        self.data_collector = None
        if data_collect_dir:
            self.data_collector = CarlaDatasetCollector(save_dir=data_collect_dir, save_interval=20)

        self.prev_dist = 0
        self.addtional_max = 20; self.addition_cnt = 0

    def step(self, frame: np.ndarray):
        position   = self.sub_enu.receive()
        heading    = np.radians(self.sub_heading.receive())
        server_fps = self.sub_server_fps.receive()
        
        ego_wp, global_wp = self.path_handler.waypoints(
            position, self.offset, heading, return_global=True, use_time = self.use_temporal
        )
        _, global_scout = self.path_handler.waypoints(
            position, self.scout_points, heading, return_global=True
        )
        curr_dist, *_ = self.path_handler.project(position)

        if self.debug:
            self.world.draw_waypoints(global_wp, 1.5 * (1 / server_fps), size = .1)
            self.world.draw_waypoints(global_scout, 1.5 * (1 / server_fps), color=(255, 0, 0), size=.2)

        is_at_junction, junction = self.world.get_waypoint_junction(global_scout[14])
        not_exit_junction, _ = self.world.get_waypoint_junction(global_scout[10])
        is_exit_junction = not not_exit_junction
        turn_signal = self.turn_classifier.turning_type(is_at_junction, junction, is_exit_junction, global_scout)
        self.send_turn_signal.send(turn_signal)

        # Only save when it moves (Prevent saving all the time when stopping at red light or stop sign)
        if self.data_collector:
            if self.addition_cnt < self.addtional_max:
                steer    = self.sub_steer.receive()
                throttle = self.sub_throttle.receive()
                brake    = self.sub_brake.receive()
                velocity = self.sub_velocity.receive()
                saved = self.data_collector.maybe_save(
                    frame, ego_wp,
                    {
                        "steer": steer,
                        "throttle": throttle,
                        "brake": brake,
                        "velocity": velocity,
                    },
                    turn_signal,
                )
                if saved:
                    if curr_dist - self.prev_dist < 1e-2:
                        self.addition_cnt += 1
            if curr_dist - self.prev_dist > 1e-2:
                self.addition_cnt = 0
            self.prev_dist = curr_dist