import os
import cv2
import numpy as np
from config import CONFIG

script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

from collections import deque
from typing import Literal

from src.control.world import World
from src.messages.logger import Logger
from src.math.path import PathHandler, WaypointsAlign
from src.math.coordinate_transform import global_2_local
from src.messages.all_messages import (
    PolylinesCmd,
    Enu,
    Heading
)
from src.messages.message_handler import MessageSubscriber, MessageSender

MAX_GPS_DELAY = CONFIG.gps.max_gps_delay
MIN_GPS_DELAY = CONFIG.gps.min_gps_delay

position_idx = CONFIG.replay_runtime.position_idx

class Map:
    def __init__(self, 
                 world: World, 
                 rect_dim: tuple = (1, 1), 
                 map_offset: tuple = (0, 0), 
                 range_ = (50, 50), 
                 resize_to = (200, 200), 
                 scale: int = 10, 
                 relative_pos: Literal["forward", "center"] = "center", 
                 invert_color = False,
                 waypoint_distance: float = 0.5):
        self.logger = Logger()
        self.world = world
        self.relative_pos = relative_pos

        carla_map = world.world.get_map()
        waypoints = carla_map.generate_waypoints(distance=waypoint_distance)
        waypoints_metadata =[]
        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            yaw = wp.transform.rotation.yaw
            waypoints_metadata += [[loc.x, loc.y, loc.z, yaw]]
        waypoints_metadata = np.array(waypoints_metadata)
        
        self._map = carla_map
        self.logger.DEBUG(f"Generated {len(waypoints_metadata)} waypoints at {waypoint_distance}m spacing")
        
        self.world_min_x = waypoints_metadata[:, 0].min()
        self.world_max_x = waypoints_metadata[:, 0].max()
        self.world_min_y = waypoints_metadata[:, 1].min()
        self.world_max_y = waypoints_metadata[:, 1].max()
        
        waypoints_metadata[:, 0] *= scale   
        waypoints_metadata[:, 1] *= scale   

        waypoints_metadata[:, 0] -= self.world_min_x * scale
        waypoints_metadata[:, 1] -= self.world_min_y * scale

        self.min_x = 0
        self.max_x = (self.world_max_x - self.world_min_x) * scale
        self.old_min_y = 0
        self.old_max_y = (self.world_max_y - self.world_min_y) * scale
        self.new_min_y = 0
        self.new_max_y = self.old_max_y

        self.waypoints_metadata = waypoints_metadata

        self.length, self.width      = rect_dim[0] * scale, rect_dim[1] * scale
        self.offset_x, self.offset_y = map_offset[0] * scale, map_offset[1] * scale
        self.scale = scale
        self.range = (range_[0] * self.scale, range_[1] * self.scale)
        self.original_range = range_
        self.resize_to = resize_to
        self.final_scale = (resize_to[0] / self.range[0] * self.scale, resize_to[1] / self.range[1] * self.scale)
        self.gps_buffer = deque(maxlen = MAX_GPS_DELAY)

        # -- Initialize the aligner
        self.aligner = WaypointsAlign(
            world=self.world,
            waypoint_distance = waypoint_distance
        )
        
        self.invert_y = True

        self._render_map(invert = invert_color)
        self._init_transmission()
    
    def _init_transmission(self):
        self.poly_pub    = MessageSender(PolylinesCmd)
        self.enu_sub     = MessageSubscriber(Enu)
        self.heading_sub = MessageSubscriber(Heading)
        
    def register_wp(self, trajectories: np.ndarray):

        spatial_aligned, _ = self.aligner.align(trajectories)

        self.path_handler = PathHandler(spatial_aligned, extrapolate=False)
        self.path_handler.position_idx = position_idx
        
        if self.relative_pos == "forward":
            self.offset_path =[i for i in range(-5, 70, 4)]
        elif self.relative_pos == "center":
            self.offset_path =[i for i in range(-50, 50, 4)]
        else:
            self.offset_path = []


    def draw_map(self, image, box_color: tuple, waypoints_coordinates):
        for cx, cy, _, yaw in waypoints_coordinates:
            rect = ((cx + self.offset_x, cy + self.offset_y), (self.width, self.length), yaw)
            box  = cv2.boxPoints(rect)
            box  = box.astype(int)

            cv2.drawContours(image, [box], 
                             0, 
                             box_color, 
                             cv2.FILLED,
                             lineType = cv2.LINE_AA)
        return image

    def draw_waypoints_lines(self, image, waypoints, color=(0, 0, 255), line_thickness=2):
        pts = np.array(waypoints, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

    def _render_map(self, invert):
        map_padding = int(self.range[0] * 2.5)  
        
        map_height = int(self.new_max_y + map_padding)
        map_width = int(self.max_x + map_padding)
        
        self.logger.DEBUG(f"Creating map image: {map_width}x{map_height}, waypoint bounds: ({self.max_x:.2f}, {self.new_max_y:.2f}), padding: {map_padding}")
        
        self.map_image = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        self.map_image = self.draw_map(self.map_image, (255, 255, 255), self.waypoints_metadata)
        
        self.map_image = cv2.GaussianBlur(self.map_image, (3, 3), sigmaX=0) 
        kernel = np.ones((3, 3), np.uint8)
        self.map_image = cv2.morphologyEx(self.map_image, cv2.MORPH_CLOSE, kernel)
        
        if invert:
            self.map_image = 255 - self.map_image
            
    def retrieve_map(self, display = False):
        location = self.enu_sub.receive()
        if self.invert_y:
            location[1] = -location[1]
        location_bfscale = location.copy()
        heading  = self.heading_sub.receive()
        heading_rad = np.radians(heading)

        if display:
            x = int((location[0] - self.world_min_x) * self.scale + self.offset_x)
            y = int((location[1] - self.world_min_y) * self.scale + self.offset_y)
            
            H, W, _ = self.map_image.shape
            w, h = self.range
            
            if self.relative_pos == "forward":
                base_radius = int((((w / 2) ** 2 + (h / 2) ** 2) ** 0.5) * 2.0)
            elif self.relative_pos == "center":
                base_radius = int(((w / 2) ** 2 + (h / 2) ** 2) ** 0.5)
            
            rotation_padding = int(base_radius * 1.5) 
            radius = base_radius + rotation_padding

            x1, x2 = max(0, x - radius), min(W, x + radius)
            y1, y2 = max(0, y - radius), min(H, y + radius)
            
            if x1 >= x2 or y1 >= y2:
                self.logger.WARNING(f"Empty cutout bounds: x1={x1}, x2={x2}, y1={y1}, y2={y2}. "
                                f"Location: ({location[0]:.1f}, {location[1]:.1f}), "
                                f"Map shape: ({W}, {H}), base_radius: {base_radius}, padded_radius: {radius}", frequency = 10)
                return None, None
            
            cutout = self.map_image[y1:y2, x1:x2]
            
            if cutout.size == 0:
                self.logger.WARNING(f"Empty cutout image generated. Bounds: ({y1}:{y2}, {x1}:{x2})")
                return None, None
            
            cx, cy = x - x1, y - y1
            cos_t, sin_t = np.cos(heading_rad), np.sin(heading_rad)
            M = np.float32([[cos_t, sin_t, (1 - cos_t) * cx - sin_t * cy],[-sin_t, cos_t, sin_t * cx + (1 - cos_t) * cy]])
                            
            try:
                rot_h, rot_w = cutout.shape[:2]
                if not hasattr(self, "src_img") or self.src_img.size() != (rot_w, rot_h):
                    self.src_img = cv2.cuda_GpuMat(rot_h, rot_w, cv2.CV_8UC3)
                    self.dst_img = cv2.cuda_GpuMat(rot_h, rot_w, cv2.CV_8UC3)
                self.src_img.upload(cutout)
                cv2.cuda.warpAffine(
                    src         = self.src_img, M = M, 
                    dsize       = (cutout.shape[1], cutout.shape[0]), 
                    dst         = self.dst_img,
                    flags       = cv2.INTER_LINEAR,
                    borderMode  = cv2.BORDER_CONSTANT,
                    borderValue = (255, 255, 255),
                    stream      = cv2.cuda_Stream.Null()
                )
                rotated = self.dst_img.download()
                self.logger.INFO("Using GPU to rotate map", once = True)
            except:
                rotated = cv2.warpAffine(cutout, M, (cutout.shape[1], cutout.shape[0]), flags=cv2.INTER_LINEAR)
                self.logger.INFO("Falling back to CPU map rotation", once = True)

            if self.relative_pos == "center":
                x1f = max(0, int(cx - w // 2))
                x2f = min(rotated.shape[1], int(cx + w // 2))
                y1f = max(0, int(cy - h // 2))
                y2f = min(rotated.shape[0], int(cy + h // 2))
            elif self.relative_pos == "forward":
                x1f = max(0, int(cx - w // 2))
                x2f = min(rotated.shape[1], int(cx + w // 2))
                y1f = max(0, int(cy - h))
                y2f = min(rotated.shape[0], int(cy))

            if x1f >= x2f or y1f >= y2f:
                self.logger.WARNING(f"Crop region out of rotated bounds: x1f={x1f}, x2f={x2f}, y1f={y1f}, y2f={y2f}. "
                              f"Rotated shape: {rotated.shape}, center: ({cx}, {cy}). Inverting y and returning black frame.")
                black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                self.invert_y = not self.invert_y
                return None, black_frame

            cutout = rotated[y1f:y2f, x1f:x2f]
            
            if cutout.shape[0] < h or cutout.shape[1] < w:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:cutout.shape[0], :cutout.shape[1]] = cutout
                cutout = padded
            
            unrouted_cutout = cutout.copy()
        
        # ================= Draw path on map ====================
        
        if hasattr(self, "path_handler"):
            global_wp, _ = self.path_handler.waypoints(
                location_bfscale, self.offset_path
            )

            self.logger.DEBUG("Current waypoint index: {}", self.path_handler.position_idx, frequency = 3)
            
            local_wp = global_2_local(location, global_wp, heading_rad)[:, :2]
            self.poly_pub.send(local_wp)
            
            if display:
                pts_world = np.atleast_2d(global_wp)[:, :2].astype(float) 
                pts_world[:, 0] = (pts_world[:, 0] - self.world_min_x) * self.scale + self.offset_x
                pts_world[:, 1] = (pts_world[:, 1] - self.world_min_y) * self.scale + self.offset_y

                pts_in_cutout = pts_world - np.array([x1, y1], dtype=float)

                ones = np.ones((pts_in_cutout.shape[0], 1), dtype=float)
                pts_hom = np.hstack([pts_in_cutout, ones])                  
                pts_trans = (pts_hom @ M.T)                                 

                pts_final = pts_trans - np.array([x1f, y1f], dtype=float)  

                self.draw_waypoints_lines(cutout, pts_final, color = (255, 0, 0), line_thickness = 2 * self.scale)
                
        if display:
            if hasattr(self, "path_handler"):
                return cv2.resize(unrouted_cutout, self.resize_to)[..., 0], cv2.resize(cutout, self.resize_to)
            else:
                return None, cv2.resize(cutout, self.resize_to) 
        else:
            return None, None