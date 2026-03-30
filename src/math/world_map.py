import os
import cv2
import carla
import numpy as np
import toml

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

conf = toml.load(os.path.join(parent, "../config/config.toml"))

gps_conf = conf.get("GPS", {})
MAX_GPS_DELAY = gps_conf.get("max_gps_delay", 60)
MIN_GPS_DELAY = gps_conf.get("min_gps_delay", 10)

position_idx = conf['Replay']['position_idx']

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
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(cutout)
                gpu_rotated = cv2.cuda.warpAffine(gpu_img, M, (cutout.shape[1], cutout.shape[0]))
                rotated = gpu_rotated.download()
                
                gpu_img.release()
                gpu_rotated.release()
                self.logger.INFO("Using GPU to rotate map", once = True)
            except:
                rotated = cv2.warpAffine(cutout, M, (cutout.shape[1], cutout.shape[0]))
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
            global_wp = self.path_handler.waypoints(
                location_bfscale, self.offset_path
            )

            self.logger.DEBUG(f"Current waypoint index: {self.path_handler.position_idx}", frequency = 5)
            
            local_wp = global_2_local(location, global_wp, heading_rad)
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

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    
    world = World(client, 10000)
    map_processor = Map(world, rect_dim = (3, 4), map_offset = (5, 5))

    dx, dy = 0, 0
    dragging = False
    prev_x, prev_y = -1, -1

    def mouse_event(event, x, y, flags, param):
        global dx, dy, dragging, prev_x, prev_y, scale

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            prev_x, prev_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx += x - prev_x
            dy += y - prev_y
            prev_x, prev_y = x, y

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:   # scroll up = zoom in
                scale *= 1.1
            else:           # scroll down = zoom out
                scale /= 1.1
            scale = max(0.1, min(scale, 10))  # clamp zoom


    cv2.namedWindow("map_image")
    cv2.setMouseCallback("map_image", mouse_event)

    while True:
        H, W, _ = map_processor.map_image.shape
        # apply scale
        scaled = cv2.resize(map_processor.map_image, (int(W) // 5, int(H) // 5))

        # create black background of original size
        view = np.zeros((H, W, 3), dtype=np.uint8)

        # compute top-left corner with dx, dy applied
        x1 = int(dx)
        y1 = int(dy)
        x2 = x1 + scaled.shape[1]
        y2 = y1 + scaled.shape[0]

        # clip coordinates so we don’t go out of bounds
        x1_clip = max(x1, 0)
        y1_clip = max(y1, 0)
        x2_clip = min(x2, W)
        y2_clip = min(y2, H)

        sx1 = x1_clip - x1
        sy1 = y1_clip - y1
        sx2 = sx1 + (x2_clip - x1_clip)
        sy2 = sy1 + (y2_clip - y1_clip)

        # paste scaled image into view
        view[y1_clip:y2_clip, x1_clip:x2_clip] = scaled[sy1:sy2, sx1:sx2]

        cv2.imshow("map_image", view)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()