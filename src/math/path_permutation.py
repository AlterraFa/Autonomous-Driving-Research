import numpy as np
import carla

from typing import Optional
from src.math import rpy2ypr
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

class ContractingWP:
    """Encapsulates waypoint contraction against nearby vehicles for replay camera safety.

    The class builds a KD-tree of non-ego vehicles in ego-local coordinates, identifies
    the closest containing vehicle for a set of local waypoints (OBB or circle mode),
    projects the detected vehicle center onto the waypoint polyline to recover arc-length
    `s`, and contracts/resamples the waypoint sequence with even spacing up to the
    remaining safe distance.
    """

    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle, containment_mode: str = "circle"):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.containment_mode = containment_mode
        self._static_bbox_data = {}
        self._vehicle_tree = None
        self._vehicle_id_map = {}
        self._vehicle_by_id = {}

    @staticmethod
    def _project2polyline(point_xyz: np.ndarray, polyline_xyz: np.ndarray):
        """Project a point to a discrete polyline and return arc-length s from the polyline root."""
        pts = np.asarray(polyline_xyz, dtype=float)
        p = np.asarray(point_xyz, dtype=float)

        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return 0.0, p, np.inf
        if len(pts) == 1:
            dist = float(np.linalg.norm(p - pts[0]))
            return 0.0, pts[0], dist

        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        seg_len_safe = seg_len + 1e-12

        ap = p - pts[:-1]
        t = np.sum(ap * seg, axis=1) / (seg_len_safe * seg_len_safe)
        t = np.clip(t, 0.0, 1.0)

        proj = pts[:-1] + seg * t[:, None]
        d2 = np.sum((p - proj) ** 2, axis=1)
        best_i = int(np.argmin(d2))

        cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
        s_star = float(cumulative[best_i] + t[best_i] * seg_len[best_i])

        return s_star, proj[best_i], float(np.sqrt(d2[best_i]))

    @staticmethod
    def _contract_polyline_even(points: np.ndarray, end_s: float):
        """Contract a polyline to [0, end_s] and resample with even spacing."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return pts

        seg = pts[1:, :3] - pts[:-1, :3]
        seg_len = np.linalg.norm(seg, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_s = float(cumulative[-1])
        if total_s <= 1e-9:
            return pts.copy()

        target_end_s = float(np.clip(end_s, 0.0, total_s))
        target_s = np.linspace(0.0, target_end_s, pts.shape[0])

        out = np.empty_like(pts)
        for col in range(pts.shape[1]):
            f = interp1d(
                cumulative,
                pts[:, col],
                kind='linear',
                bounds_error=False,
                fill_value=(pts[0, col], pts[-1, col]),
            )
            out[:, col] = f(target_s)
        return out

    def build_vehicle_tree(self, ego_id: int):
        """Build k-nearest search structure for all non-ego vehicles in ego-local frame."""
        ego_transform = self.ego_vehicle.get_transform()
        ego_loc = np.array([ego_transform.location.x, ego_transform.location.y, ego_transform.location.z])
        r_ego = rpy2ypr([ego_transform.rotation.roll, ego_transform.rotation.pitch, ego_transform.rotation.yaw])

        current_vehicles = self.world.get_actors().filter("vehicle.*")

        raw_global_positions = []
        actor_list = []

        for actor in current_vehicles:
            if actor.id != ego_id:
                loc = actor.get_location()
                raw_global_positions.append([loc.x, loc.y, loc.z])
                actor_list.append(actor)

                if actor.id not in self._static_bbox_data:
                    bb = actor.bounding_box
                    self._static_bbox_data[actor.id] = {
                        "extent": np.array([bb.extent.x, bb.extent.y, bb.extent.z]),
                        "relative_center": np.array([bb.location.x, bb.location.y, bb.location.z])
                    }

        if not actor_list:
            self._vehicle_tree = None
            return

        global_pos_np = np.array(raw_global_positions)
        local_pos_np = r_ego.inv().apply(global_pos_np - ego_loc)

        self._vehicle_id_map = {i: actor.id for i, actor in enumerate(actor_list)}
        self._vehicle_by_id = {actor.id: actor for actor in actor_list}
        self._vehicle_tree = cKDTree(local_pos_np)

    def query_closest_containing_vehicle(self, points_local: np.ndarray, ref_point: int = 0, k_nearest: int = 5, containment_mode: Optional[str] = None):
        """Containment test on k-nearest candidates (mode: 'obb' or 'circle')."""
        if self._vehicle_tree is None:
            return None

        mode = (containment_mode or self.containment_mode or "obb").lower()
        points_local = np.atleast_2d(points_local)
        ref_point_local = np.asarray(points_local[ref_point], dtype=float)
        path_local_from_root = np.asarray(points_local, dtype=float) - ref_point_local
        _, indices = self._vehicle_tree.query(points_local, k=min(k_nearest, len(self._vehicle_id_map)))
        indices = np.atleast_1d(indices)
        if indices.ndim == 1:
            indices = indices[None, :]

        ego_transform = self.ego_vehicle.get_transform()
        ego_loc = np.array([ego_transform.location.x, ego_transform.location.y, ego_transform.location.z])
        r_ego = rpy2ypr([ego_transform.rotation.roll, ego_transform.rotation.pitch, ego_transform.rotation.yaw])

        for i, point_row in enumerate(indices):
            p_world = ego_loc + r_ego.apply(points_local[i])

            for neighbor_idx in point_row:
                v_id = self._vehicle_id_map.get(int(neighbor_idx))
                actor = self._vehicle_by_id.get(v_id)
                static = self._static_bbox_data.get(v_id)

                if not actor or not static:
                    continue

                tr = actor.get_transform()
                r_actor = rpy2ypr([tr.rotation.roll, tr.rotation.pitch, tr.rotation.yaw])

                v_world_pos = np.array([tr.location.x, tr.location.y, tr.location.z])
                center_world = v_world_pos + r_actor.apply(static["relative_center"])
                center_ego = r_ego.inv().apply(center_world - ego_loc) - ref_point_local
                center_s, _, _ = self._project2polyline(center_ego, path_local_from_root)

                r_actor_ego = r_ego.inv() * r_actor
                rotation_ego_rpy = r_actor_ego.as_euler("xyz", degrees=True)

                p_bin_local = r_actor.inv().apply(p_world - center_world)
                if mode == "circle":
                    radius = float(np.linalg.norm(static["extent"][:2]))
                    contains = (np.linalg.norm(p_bin_local[:2]) <= radius + 1e-3)
                else:
                    contains = np.all(np.abs(p_bin_local) <= static["extent"] + 1e-3)

                if contains:
                    return v_id, actor, {
                        "center_world": center_world,
                        "center_ego": center_ego,
                        "center_s": center_s,
                        "extent": static["extent"],
                        "rotation": tr.rotation,
                        "rotation_ego": rotation_ego_rpy
                    }
        return None

    def contract_local_wp(self, local_wp: np.ndarray, ref_point: int = 0):
        """Match nearest containing vehicle and contract waypoints by remaining safe arc-length."""
        match = self.query_closest_containing_vehicle(local_wp[:, :3], ref_point=ref_point)
        if not match:
            return local_wp, None

        _, _, cache = match
        remaining_s = float(cache['center_s']) - float(np.linalg.norm(cache['extent'])) * 1.2
        if remaining_s > 0.0:
            local_wp = self._contract_polyline_even(local_wp, remaining_s)
        return local_wp, match