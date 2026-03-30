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
from src.control.world import World
from src.math.path import (
    PathHandler, 
    BranchingPath, 
    TurnClassify, 
    WaypointsAlign
)
from src.math.coordinate_transform import global_2_local


conf = toml.load(os.path.join(parent, "../config/config.toml"))
quality = conf['Picture']['quality']
position_idx = conf['Replay']['position_idx']

temporal_offset      = conf['Offsets']['temporal_offset']
spatial_offset       = conf['Offsets']['spatial_offset']
scout_offset_params  = conf['Offsets']['scout_offset_params']
front_vehicle_offset = conf['Offsets']['front_vehicle_offset']

class TrajectoryBuffer:
    def __init__(self, save_dir: str, init_cap = int(8192 * 8), dist_thresh_m = 0, min_dt_s = 0.05):
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
    GlobalWP
)
class ReplayHandler:

    turn_classify = True
    __slot__ = ["road_type"]
    
    def __init__(self, world: World, true_trajectories: np.ndarray, data_collect_dir: str = None, use_temporal: bool = False, debug: bool = False):
        self.logger = Logger()
        
        _, midlane_waypoints = WaypointsAlign(world, 2.0).align(true_trajectories)
        self.path_handler = PathHandler(midlane_waypoints)
        self.path_handler.position_idx = position_idx
        self.debug = debug
        self.virt_world = world
        self.use_temporal = use_temporal
        self.scout_points = [i for i in range(*scout_offset_params)]
        if not self.use_temporal:
            self.offset   = spatial_offset
        else:
            self.offset   = temporal_offset
        self.turn_classifier = TurnClassify(world=world, threshold_deg=20)
        self.branching_path  = BranchingPath(self.virt_world)
        self.data_collector = None
        if data_collect_dir:
            self.data_collector = CarlaDatasetCollector(save_dir=data_collect_dir, fps=20)

        self._init_transmittor()
        self.prev_dist = 0
        self.additional_max = 20; self.addition_cnt = 0
        self.start_time = self.sub_server_runtime.receive()

    def _init_transmittor(self):
        self.sub_location         = MessageSubscriber(Location)
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
        self.send_turn_signal         = MessageSender(TurnSignal)
        self.send_global_wp           = MessageSender(GlobalWP)
        

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
        
        
        
        curr_dist, *_, lat_err = self.path_handler.project(position)
        mid_global_scout = self.path_handler.waypoints(
            position, self.scout_points
        )
        
        mid_global = self.path_handler.waypoints(
            position, self.offset, use_time = self.use_temporal, merge = True
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
            mid_global[:, -1] += .5
            self.virt_world.draw_waypoints(mid_global, 2.0 * (1 / server_fps), size = .1, color = (255, 0, 0))

        if self.turn_classify:
            global_scout = self.path_handler.waypoints(
                position, self.scout_points
            )
            
            is_at_junction , junction = self.virt_world.get_waypoint_junction(global_scout[14])
            switch_junction, other_junction = self.virt_world.get_waypoint_junction(global_scout[19])
            if is_at_junction and switch_junction:
                junction = other_junction
            not_exit_junction, _ = self.virt_world.get_waypoint_junction(global_scout[11])
            is_exit_junction = not not_exit_junction
            turn_signal = self.turn_classifier.turning_type(is_at_junction, junction, is_exit_junction, global_scout, debug = self.debug)
        else:
            turn_signal = -1

        self.send_global_wp.send(mid_global)
        self.send_turn_signal.send(turn_signal)


        # Only save when it moves (Prevent saving all the time when stopping at red light or stop sign)
        if self.data_collector:
            steer    = self.sub_steer_logging.receive()
            throttle = self.sub_throttle_logging.receive()
            brake    = self.sub_brake_logging.receive()
            velocity = self.sub_velocity.receive()
            steer_angle = self.sub_steer_angle.receive()
            # if self.addition_cnt < self.additional_max:
            #     if saved:
            #         if curr_dist - self.prev_dist < 1e-2:
            #             self.addition_cnt += 1
            # if curr_dist - self.prev_dist > 1e-2:
            #     self.addition_cnt = 0
            self.logger.DEBUG(f"Lat Err: {lat_err:.3f}m", frequency = 5)

            saved = self.data_collector.maybe_save(
                {
                    "gt_data": {
                        "midlane_wp" : mid_ego,
                        "aux_wp"     : ego_branches,
                        "steer"      : steer,
                        "steer_angle": steer_angle,
                        "throttle"   : throttle,
                        "brake"      : brake,
                        "velocity"   : velocity,
                        "lat_err"    : lat_err
                    },
                    "command": {
                        "turn_signal": turn_signal,
                        "polycmd"    : self.sub_polylines.receive(), 
                    },
                    "condition": {
                        "GPS"        : vehicle_location,
                        "heading"    : heading,
                        "road_type"  : self.road_type,
                    }, 
                    "timestamp"  : self.sub_server_runtime.receive() - self.start_time
                },
                **frame
            )
            # self.prev_dist = curr_dist
        return mid_ego