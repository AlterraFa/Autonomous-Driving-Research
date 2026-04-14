import os, sys
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
from src.control.world import World
from src.math.path import (
    PathHandler, 
    BranchingPath, 
    TurnClassify, 
    WaypointsAlign
)
from config.enum import CameraView
from src.math.coordinate_transform import global_2_local_rot, global_2_local_full_rot, rpy2ypr, camera_extrinsic, camera_intrinsic, ego_to_pixel
from config import CONFIG

quality = CONFIG.picture.quality
position_idx = CONFIG.replay_runtime.position_idx

temporal_offset      = CONFIG.offsets.temporal_offset
spatial_offset       = CONFIG.offsets.spatial_offset
scout_offset_params  = CONFIG.offsets.scout_offset_params
front_offset = CONFIG.offsets.front_offset

class TrajectoryBuffer:
    def __init__(
        self,
        save_dir: str,
        init_cap=CONFIG.data_collection.trajectory_buffer_capacity,
        dist_thresh_m=CONFIG.data_collection.trajectory_distance_threshold_m,
        min_dt_s=CONFIG.data_collection.trajectory_min_dt_s,
    ):
        self.log = Logger()
        self.log.DEBUG("SAVING VEHICLE TRAJECTORY")
        self.arr = np.empty((init_cap, 7), dtype=np.float32)
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

    def update(self, loc: np.ndarray, rot: np.ndarray) -> None:
        t = time.time()
        p = [loc[0], loc[1], loc[2]]
        if self.last is not None:
            if (t - self.last_t) < self.min_dt:
                return
            if self._dist3(p, self.last) <= self.dist_thresh:
                return
            
        p.append(t - self.last_t)
        p.extend(rot)
        self.last, self.last_t = p, t
        if self.n >= self.arr.shape[0]:
            new = np.empty((self.arr.shape[0]*2, 7), dtype=np.float32)
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
            self._savers[key].save(
                cv2.imwrite, str(fpath), img_copy, self.encode_param
            )
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


from src.messages.message_handler import MessageSubscriber, MessageSender
from src.messages.all_messages import (
    Location,
    Heading,
    ServerFps,
    ClientFps,
    SteerLog,
    Velocity,
    BrakeLog,
    ThrottleLog,
    TurnSignal,
    PolylinesCmd,
    ServerRuntime,
    SteerAngle,
    GlobalWPSpatial,
    LocalWPSpatial,
    GlobalWPTemporal,
    LocalWPTemporal,
    PixelWPSpatial, 
    PixelWPTemporal,
    Rotation,
    CameraDimension
)
class ReplayHandler:

    turn_classify = False
    __slot__ = ["road_type"]
    
    def __init__(self, world: World, true_trajectories: np.ndarray, data_collect_dir: str = None, use_temporal: bool = False, debug: bool = False):
        self.logger = Logger()
        
        _, midlane_waypoints = WaypointsAlign(world, 2.0).align(true_trajectories)
        self.path_handler = PathHandler(midlane_waypoints)
        self.path_handler.position_idx = position_idx
        self.debug = debug
        self.virt_world = world
        self.scout_points = [i for i in range(*scout_offset_params)]
        self.spatial_offset = spatial_offset
        self.temporal_offset = temporal_offset
        self.turn_classifier = TurnClassify(
            world=world,
            threshold_deg=CONFIG.turn_detection.threshold_deg,
        )
        self.branching_path  = BranchingPath(self.virt_world)
        self.data_collector = None
        if data_collect_dir:
            self.data_collector = CarlaDatasetCollector(
                save_dir=data_collect_dir,
                fps=CONFIG.data_collection.save_fps,
            )

        self._init_transmittor()
        self.prev_dist = 0
        self.additional_max = CONFIG.data_collection.additional_trajectory_max; self.addition_cnt = 0
        self.start_time = self.sub_server_runtime.receive()
        self.temporal = use_temporal

    def _init_transmittor(self):
        self.sub_location         = MessageSubscriber(Location)
        self.sub_rotation         = MessageSubscriber(Rotation)
        self.sub_heading          = MessageSubscriber(Heading)
        self.sub_server_fps       = MessageSubscriber(ServerFps)
        self.sub_client_fps       = MessageSubscriber(ClientFps)
        self.sub_steer_logging    = MessageSubscriber(SteerLog)
        self.sub_throttle_logging = MessageSubscriber(ThrottleLog)
        self.sub_brake_logging    = MessageSubscriber(BrakeLog)
        self.sub_velocity         = MessageSubscriber(Velocity)
        self.sub_polylines        = MessageSubscriber(PolylinesCmd)
        self.sub_server_runtime   = MessageSubscriber(ServerRuntime)
        self.sub_steer_angle      = MessageSubscriber(SteerAngle)
        self.sub_cam_dim          = MessageSubscriber(CameraDimension)
        

        self.send_turn_signal         = MessageSender(TurnSignal)
        self.send_global_wp_spatial   = MessageSender(GlobalWPSpatial)
        self.send_local_wp_spatial    = MessageSender(LocalWPSpatial)
        self.send_global_wp_temporal  = MessageSender(GlobalWPTemporal)
        self.send_local_wp_temporal   = MessageSender(LocalWPTemporal)
        self.send_pixel_wp_temporal   = MessageSender(PixelWPTemporal)
        self.send_pixel_wp_spatial    = MessageSender(PixelWPSpatial)

    def _ego_state(self):
        vehicle_location = self.sub_location.receive()
        vehicle_rotation = self.sub_rotation.receive()
        heading = np.radians(self.sub_heading.receive())

        # Vehicle-forward is local +X in CARLA/Unreal coordinates.
        r_ego = rpy2ypr(vehicle_rotation)
        front_vec = r_ego.apply(np.array([front_offset, 0.0, 0.0]))
        position = vehicle_location + front_vec
        return vehicle_location, vehicle_rotation, heading, position

    def _waypoint_state(self, vehicle_location, vehicle_rotation, position):
        curr_dist, *_, lat_err = self.path_handler.project(position)
        proj_data = (curr_dist, lat_err)
        global_scout, _ = self.path_handler.waypoints(
            position, self.scout_points, precomputed_s_side=proj_data
        )

        global_loc_spatial, global_rot_spatial = self.path_handler.waypoints(
            position, self.spatial_offset, use_time=False, merge=True, precomputed_s_side=proj_data
        )
        global_loc_temporal, global_rot_temporal = self.path_handler.waypoints(
            position, self.temporal_offset, use_time=True, merge=True, precomputed_s_side=proj_data
        )

        local_loc_spatial = global_2_local_full_rot(vehicle_location, global_loc_spatial, vehicle_rotation)
        local_rot_spatial = global_2_local_rot(global_rot_spatial, vehicle_rotation)
        local_loc_temporal = global_2_local_full_rot(vehicle_location, global_loc_temporal, vehicle_rotation)
        local_rot_temporal = global_2_local_rot(global_rot_temporal, vehicle_rotation)

        return (
            lat_err,
            global_scout,
            global_loc_spatial,
            global_rot_spatial,
            global_loc_temporal,
            global_rot_temporal,
            local_loc_spatial,
            local_rot_spatial,
            local_loc_temporal,
            local_rot_temporal,
        )

    def _update_road_type(self, global_loc_spatial, global_scout):
        path_branches = self.branching_path.brancher(global_loc_spatial, global_scout, persist_dist=20)
        self.road_type = "multi" if path_branches.shape[0] > 1 else "uni"

    def _project_pixels(self, local_wp, frame):
        view_metadata = self.sub_cam_dim.receive() or {}
        w = int(view_metadata.get("width", 0))
        h = int(view_metadata.get("height", 0))
        if (w <= 0 or h <= 0) and len(frame) > 0:
            first_frame = next(iter(frame.values()))
            h, w = first_frame.shape[:2]

        if w <= 0 or h <= 0:
            return np.full((len(local_wp), 2), np.nan, dtype=np.float64)

        fov_deg = float(view_metadata.get("fov_deg", view_metadata.get("fov", 90.0)))
        cam_tf = {
            "x": float(view_metadata.get("x", CameraView.FIRST_PERSON.value["x"])),
            "y": float(view_metadata.get("y", CameraView.FIRST_PERSON.value["y"])),
            "z": float(view_metadata.get("z", CameraView.FIRST_PERSON.value["z"])),
            "roll": float(view_metadata.get("roll", CameraView.FIRST_PERSON.value["roll"])),
            "pitch": float(view_metadata.get("pitch", CameraView.FIRST_PERSON.value["pitch"])),
            "yaw": float(view_metadata.get("yaw", CameraView.FIRST_PERSON.value["yaw"])),
        }

        K = camera_intrinsic(w, h, fov_deg)
        E = camera_extrinsic(cam_tf)
        return ego_to_pixel(local_wp, K, E, w, h, clip=False)

    def _debug_draw_waypoints(self, global_loc_spatial, global_loc_temporal):
        if not self.debug:
            return

        server_fps = self.sub_server_fps.receive()
        if server_fps < 1:
            server_fps = self.sub_client_fps.receive()
        wp = global_loc_temporal if self.temporal else global_loc_spatial
        wp[:, -1] += 0.5
        self.virt_world.draw_waypoints(wp, 2.0 * (1 / server_fps), size=0.1, color=(255, 0, 0))

    def _turn_signal(self, global_scout):
        if not self.turn_classify:
            return -1

        is_at_junction, junction = self.virt_world.get_waypoint_junction(global_scout[14])
        switch_junction, other_junction = self.virt_world.get_waypoint_junction(global_scout[19])
        if is_at_junction and switch_junction:
            junction = other_junction
        not_exit_junction, _ = self.virt_world.get_waypoint_junction(global_scout[11])
        is_exit_junction = not not_exit_junction
        return self.turn_classifier.turning_type(
            is_at_junction, junction, is_exit_junction, global_scout, debug=self.debug
        )

    def _publish_waypoints(self, local_loc_spatial, local_rot_spatial, global_loc_spatial, global_rot_spatial,
                           local_loc_temporal, local_rot_temporal, global_loc_temporal, global_rot_temporal, uv_spatial, uv_temporal,
                           turn_signal):
        local_wp_spatial = np.concatenate([local_loc_spatial, local_rot_spatial], axis=1)
        global_wp_spatial = np.concatenate([global_loc_spatial, global_rot_spatial], axis=1)
        local_wp_temporal = np.concatenate([local_loc_temporal, local_rot_temporal], axis=1)
        global_wp_temporal = np.concatenate([global_loc_temporal, global_rot_temporal], axis=1)

        self.send_local_wp_spatial.send(local_wp_spatial)
        self.send_global_wp_spatial.send(global_wp_spatial)
        self.send_local_wp_temporal.send(local_wp_temporal)
        self.send_global_wp_temporal.send(global_wp_temporal)
        self.send_pixel_wp_spatial.send(uv_spatial)
        self.send_pixel_wp_temporal.send(uv_temporal)
        self.send_turn_signal.send(turn_signal)

    def _maybe_save(self, authorized_saving, frame, local_loc_spatial, local_loc_temporal,
                    lat_err, uv_spatial, uv_temporal, turn_signal, vehicle_location, heading):
        if not (self.data_collector and authorized_saving):
            return False

        steer = self.sub_steer_logging.receive()
        throttle = self.sub_throttle_logging.receive()
        brake = self.sub_brake_logging.receive()
        velocity = self.sub_velocity.receive()
        steer_angle = self.sub_steer_angle.receive()

        return self.data_collector.maybe_save(
            {
                "gt_data": {
                    "midlane_wp": local_loc_spatial[:, :2],
                    "midlane_wp_temporal": local_loc_temporal[:, :2],
                    'pixel_wp': uv_spatial,
                    'pixel_wp_temporal': uv_temporal,
                    "steer": steer,
                    "steer_angle": steer_angle,
                    "throttle": throttle,
                    "brake": brake,
                    "velocity": velocity,
                    "lat_err": lat_err,
                },
                "command": {
                    "turn_signal": turn_signal,
                    "polycmd": self.sub_polylines.receive(),
                },
                "condition": {
                    "GPS": vehicle_location,
                    "heading": heading,
                    "road_type": self.road_type,
                },
                "timestamp": self.sub_server_runtime.receive() - self.start_time,
            },
            **frame,
        )
        

    def step(self, authorized_saving = False, **frame: np.ndarray):
        vehicle_location, vehicle_rotation, heading, position = self._ego_state()

        (
            lat_err,
            global_scout,
            global_loc_spatial,
            global_rot_spatial,
            global_loc_temporal,
            global_rot_temporal,
            local_loc_spatial,
            local_rot_spatial,
            local_loc_temporal,
            local_rot_temporal,
        ) = self._waypoint_state(vehicle_location, vehicle_rotation, position)

        self._update_road_type(global_loc_spatial, global_scout)

        uv_spatial  = self._project_pixels(local_loc_spatial, frame)
        uv_temporal = self._project_pixels(local_loc_temporal, frame)
        if self.debug:
            valid_uv = np.isfinite(uv_spatial).all(axis=1).sum()
            self.logger.DEBUG(f"Projected spatial pixels valid: {valid_uv}/{len(uv_spatial)}", frequency=0.5)
            valid_uv = np.isfinite(uv_temporal).all(axis=1).sum()
            self.logger.DEBUG(f"Projected temporal pixels valid: {valid_uv}/{len(uv_temporal)}", frequency=0.5)

        self._debug_draw_waypoints(global_loc_spatial, global_loc_temporal)

        turn_signal = self._turn_signal(global_scout)

        self._publish_waypoints(
            local_loc_spatial,
            local_rot_spatial,
            global_loc_spatial,
            global_rot_spatial,
            local_loc_temporal,
            local_rot_temporal,
            global_loc_temporal,
            global_rot_temporal,
            uv_spatial,
            uv_temporal,
            turn_signal,
        )

        self.logger.DEBUG(f"Lat Err: {lat_err:.3f}m", frequency = 0.5)
        return self._maybe_save(
            authorized_saving,
            frame,
            local_loc_spatial,
            local_loc_temporal,
            lat_err,
            uv_spatial,
            uv_temporal,
            turn_signal,
            vehicle_location,
            heading,
        )