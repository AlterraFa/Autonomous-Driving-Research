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
from src.math.path import _find_entry_clusters, _find_exit, waypoints_between, PathHandler
from src.math.coordinate_transform import global_2_local
from scipy.spatial import cKDTree
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
        self.log = Logger()
        self.world = world
        self.relative_pos = relative_pos

        carla_map = world.world.get_map()
        # Generate waypoints at finer spacing for detailed map rendering
        waypoints = carla_map.generate_waypoints(distance=waypoint_distance)
        self.wp_dict = {(wp.transform.location.x, wp.transform.location.y): wp for wp in waypoints}
        waypoints_metadata = []
        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            yaw = wp.transform.rotation.yaw
            waypoints_metadata += [[loc.x, loc.y, loc.z, yaw]]
        waypoints_metadata = np.array(waypoints_metadata)
        
        self._map = carla_map

        self.log.DEBUG(f"Generated {len(waypoints_metadata)} waypoints at {waypoint_distance}m spacing")
        
        # Store raw world bounds BEFORE transformation
        self.world_min_x = waypoints_metadata[:, 0].min()
        self.world_max_x = waypoints_metadata[:, 0].max()
        self.world_min_y = waypoints_metadata[:, 1].min()
        self.world_max_y = waypoints_metadata[:, 1].max()
        
        # Scale up for fine grain detail
        waypoints_metadata[:, 0] *= scale   # x
        waypoints_metadata[:, 1] *= scale   # y

        # Shift BOTH x and y to start from zero
        waypoints_metadata[:, 0] -= self.world_min_x * scale
        waypoints_metadata[:, 1] -= self.world_min_y * scale

        self.min_x = 0
        self.max_x = (self.world_max_x - self.world_min_x) * scale
        self.old_min_y = 0
        self.old_max_y = (self.world_max_y - self.world_min_y) * scale
        self.new_min_y = 0
        self.new_max_y = self.old_max_y

        # Store for later
        self.waypoints_metadata = waypoints_metadata

        # Draw the map_image using rectangles
        self.length, self.width      = rect_dim[0] * scale, rect_dim[1] * scale
        self.offset_x, self.offset_y = map_offset[0] * scale, map_offset[1] * scale
        self.scale = scale
        self.range = (range_[0] * self.scale, range_[1] * self.scale)
        self.original_range = range_
        self.resize_to = resize_to
        self.final_scale = (resize_to[0] / self.range[0] * self.scale, resize_to[1] / self.range[1] * self.scale)

        self.stored_entries = {}  # junction_id -> entry_wp
        self._wp_list = list(self.wp_dict.values())
        self._wps = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in self._wp_list])
        self._tree = cKDTree(self._wps)
        self.gps_buffer = deque(maxlen = MAX_GPS_DELAY)

        self._render_map(invert = invert_color)
        self._init_transmission()
    
    def _init_transmission(self):
        self.poly_pub    = MessageSender(PolylinesCmd)
        self.enu_sub     = MessageSubscriber(Enu)
        self.heading_sub = MessageSubscriber(Heading)
        
    def precompute_waypoints(self, trajectories: np.ndarray):
        
        points = self.waypoints_compute(trajectories[:, :3])
        if len(points) == 0:
            return points

        # Always keep first point, then keep if distance > tol
        self.path_handler = PathHandler(points, extrapolate = False)
        if self.relative_pos == "forward":
            self.offset_path  = [i for i in range(-5, 70, 4)]
        elif self.relative_pos == "center":
            self.offset_path  = [i for i in range(-50, 50, 4)]

        # Interpolate time data from original trajectories to filtered waypoints
        # The filtered points may have fewer entries than the original trajectories
        original_time = trajectories[:, -1]
        num_original = len(trajectories)
        num_filtered = len(points)
        
        # Create indices for interpolation
        original_indices = np.linspace(0, num_original - 1, num_original)
        filtered_indices = np.linspace(0, num_original - 1, num_filtered)
        
        # Interpolate time values
        points[:, -1] = np.interp(filtered_indices, original_indices, original_time)
        
        return points


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
        
        # Create the map_image with generous padding to handle rotation and boundary cases
        # Padding should be at least the maximum radius we'll request during retrieve_map
        map_padding = int(self.range[0] * 2.5)  # Extra large padding for rotation buffer
        
        map_height = int(self.new_max_y + map_padding)
        map_width = int(self.max_x + map_padding)
        
        self.log.DEBUG(f"Creating map image: {map_width}x{map_height}, waypoint bounds: ({self.max_x:.2f}, {self.new_max_y:.2f}), padding: {map_padding}")
        
        self.map_image = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        self.map_image = self.draw_map(self.map_image, (255, 255, 255), self.waypoints_metadata)
        
        self.map_image = cv2.GaussianBlur(self.map_image, (3, 3), sigmaX=0) 
        kernel = np.ones((3, 3), np.uint8)
        self.map_image = cv2.morphologyEx(self.map_image, cv2.MORPH_CLOSE, kernel)
        
        if invert:
            self.map_image = 255 - self.map_image
            
    
    def retrieve_map(self, display = False):
        """Instead of drawing on the larger self.map_image, we draw on the smaller cutout image and apply waypoints transformation"""

        location = self.enu_sub.receive()
        location_bfscale = location.copy()
        heading  = self.heading_sub.receive()
        heading_rad = np.radians(heading)

        # ================ Retrieve Submap ======================
        # Retrieve the normal submap with the same transformation as __init__
        if display:
            # Apply same world-to-map transformation as waypoints
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

            # clamp once
            x1, x2 = max(0, x - radius), min(W, x + radius)
            y1, y2 = max(0, y - radius), min(H, y + radius)
            
            # DEBUG: Check if cutout would be empty
            if x1 >= x2 or y1 >= y2:
                self.log.WARNING(f"Empty cutout bounds: x1={x1}, x2={x2}, y1={y1}, y2={y2}. "
                                f"Location: ({location[0]:.1f}, {location[1]:.1f}), "
                                f"Map shape: ({W}, {H}), base_radius: {base_radius}, padded_radius: {radius}", frequency = 10)
                return None, None
            
            self.log.DEBUG(f"Cutout bounds: x1={x1}, x2={x2}, y1={y1}, y2={y2}, shape={self.map_image.shape}", frequency = 10)

            # First cutout uses radius to avoid missing lanes during rotation
            cutout = self.map_image[y1:y2, x1:x2]
            
            # Verify cutout is valid
            if cutout.size == 0:
                self.log.WARNING(f"Empty cutout image generated. Bounds: ({y1}:{y2}, {x1}:{x2})")
                return None, None
            
            cx, cy = x - x1, y - y1
            cos_t, sin_t = np.cos(heading_rad), np.sin(heading_rad)
            M = np.float32([[cos_t, sin_t, (1 - cos_t) * cx - sin_t * cy],
                            [-sin_t, cos_t, sin_t * cx + (1 - cos_t) * cy]])
                            
            # Much faster because overhead offload to GPU
            # rotated = cv2.warpAffine(cutout, M, (cutout.shape[1], cutout.shape[0]))
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(cutout)
            gpu_rotated = cv2.cuda.warpAffine(gpu_img, M, (cutout.shape[1], cutout.shape[0]))
            rotated = gpu_rotated.download()
            
            # Explicitly release GPU memory to prevent accumulation during replay
            gpu_img.release()
            gpu_rotated.release()

            # one precise crop - handle negative center by clamping to visible region
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

            # If requested region is entirely out of bounds after rotation, skip
            if x1f >= x2f or y1f >= y2f:
                self.log.DEBUG(f"Crop region out of rotated bounds: x1f={x1f}, x2f={x2f}, y1f={y1f}, y2f={y2f}. "
                              f"Rotated shape: {rotated.shape}, center: ({cx}, {cy}). Returning black frame.")
                # Return a black frame instead of crashing
                black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                return None, black_frame

            # Second cutout to refine to the correct range
            cutout = rotated[y1f:y2f, x1f:x2f]
            
            # If cutout is smaller than expected, pad with black
            if cutout.shape[0] < h or cutout.shape[1] < w:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:cutout.shape[0], :cutout.shape[1]] = cutout
                cutout = padded
            
            unrouted_cutout = cutout.copy()
        
        # ================= Draw path on map ====================
        
        if hasattr(self, "path_handler"):
            # Extract the waypoints using interpolation, globals only, locals waypoints embedded in waypoint code will mess up rotation and transformation
            # We don't need yaw, yaw = 0 is a dummy value        
            global_wp = self.path_handler.waypoints(
                location_bfscale, self.offset_path
            )
            
            local_wp = global_2_local(location, global_wp, heading_rad)
            self.poly_pub.send(local_wp)
            
            if display:
                pts_world = np.atleast_2d(global_wp)[:, :2].astype(float)  # (N,2)
                # Apply same world-to-map transformation as waypoints
                pts_world[:, 0] = (pts_world[:, 0] - self.world_min_x) * self.scale + self.offset_x
                pts_world[:, 1] = (pts_world[:, 1] - self.world_min_y) * self.scale + self.offset_y

                pts_in_cutout = pts_world - np.array([x1, y1], dtype=float)   # (N,2)

                ones = np.ones((pts_in_cutout.shape[0], 1), dtype=float)
                pts_hom = np.hstack([pts_in_cutout, ones])                  # (N,3)
                pts_trans = (pts_hom @ M.T)                                 # (N,2)

                pts_final = pts_trans - np.array([x1f, y1f], dtype=float)   # (N,2)

                self.draw_waypoints_lines(cutout, pts_final, color = (255, 0, 0), line_thickness = 2 * self.scale)
                
        if display:
            if hasattr(self, "path_handler"): # If we have path, return both
                return cv2.resize(unrouted_cutout, self.resize_to)[..., 0], cv2.resize(cutout, self.resize_to)
            else:
                return None, cv2.resize(cutout, self.resize_to) # Else we return cutout as it is the same as unrouted_cutout
        else:
            return None, None
    
    def get_jid_for_point(self, point):
        segs = self.world.get_segments_from_points("junction", np.array([point]))
        return segs[0].id if segs else None
    
    def waypoints_compute(self, coordinates: np.ndarray):
        """This code was initially used for running online
        Now it is used for precompute waypoints"""

        junctions = self.world.get_segments_from_points("junction", coordinates)

        # ========== Filter out duplicated adjacent junctions ============
        last_jid = None
        for junction_id in range(len(junctions) - 1, -1, -1):
            if junctions[junction_id].id == last_jid:
                junctions.pop(junction_id)
            else:
                last_jid = junctions[junction_id].id

        jids = [self.get_jid_for_point(p) for p in coordinates]

        # ================ Grouping coordinate to respective junctions =================
        # This is to avoid assigning wrong entry, exit wp when going over junction more than once
        groups: list[np.ndarray] = []
        current_group: list = []
        current_jid = jids[0]

        # start current_group only if first point is in a junction
        if current_jid is not None:
            current_group.append(coordinates[0])

        for pt, jid in zip(coordinates[1:], jids[1:]):
            if jid == current_jid:
                if jid is not None:
                    current_group.append(pt)
            else:
                # boundary: flush previous junction group if any
                if current_jid is not None and current_group:
                    groups.append(np.array(current_group))
                # start next group only if new jid is a junction
                current_group = [pt] if jid is not None else []
                current_jid = jid

        # flush last group
        if current_jid is not None and current_group:
            groups.append(np.array(current_group))

        junctions_metadata_groups = []

        # This loop initially used caching because it was meant to run online
        # Now it is needed to precompute the path through junctions
        # Avoids KDTree snapping errors in dense areas by explicitly following entry→exit pairs
        # Caches entry clusters per junction, but clears them once the ego passes the exit
        for coordinate_group, junction in zip(groups, junctions):

            wp_pairs = junction.get_waypoints(carla.LaneType.Driving)
            possible_pairs = _find_entry_clusters(wp_pairs, coordinate_group)

            # Dynamic exit
            entry_wp, exit_wp = _find_exit(possible_pairs, coordinate_group)

            # Collect waypoints inside junction
            wp_in_junctions = waypoints_between(entry_wp, exit_wp)

            group_meta = []
            for wp in wp_in_junctions:
                loc = wp.transform.location
                yaw = wp.transform.rotation.yaw
                group_meta.append([loc.x, loc.y, loc.z, yaw])
            junctions_metadata_groups.append(np.array(group_meta))

        # ============ Merge non-junction waypoints with junction metadata ============
        combined_meta = []
        group_iter = iter(junctions_metadata_groups)

        i = 0
        while i < len(coordinates):
            jid = jids[i]
            if jid is None:
                # use KDTree for waypoints outside junctions
                x, y, z = coordinates[i]
                _, idx = self._tree.query([x, y])
                closest_wp = self._wp_list[idx]
                loc = closest_wp.transform.location
                yaw = closest_wp.transform.rotation.yaw
                combined_meta.append([loc.x, loc.y, loc.z, yaw])
                i += 1
            else:
                # insert full junction group metadata
                try:
                    group_meta = next(group_iter)
                except:
                    break
                if len(group_meta) > 0:
                    j_tree = cKDTree(group_meta[:, :2])
                start_jid = jid
                while i < len(jids) and jids[i] == start_jid:
                    x, y, z = coordinates[i]
                    if len(group_meta) > 0:
                        _, gi = j_tree.query([x, y])
                        gx, gy, gz, gyaw = group_meta[int(gi)]
                        combined_meta.append([float(gx), float(gy), float(gz), float(gyaw)])
                    else:
                        # fallback to global KDTree if junction group is empty
                        _, idx = self._tree.query([x, y])
                        closest_wp = self._wp_list[idx]
                        loc = closest_wp.transform.location
                        yaw = closest_wp.transform.rotation.yaw
                        combined_meta.append([loc.x, loc.y, loc.z, yaw])
                    i += 1

        filtered_meta = []
        tol = 1e-2
        for i in range(len(combined_meta)):
            closest_wp = np.array(combined_meta[i], dtype = float)[:3]

            j = i
            backward_point = np.array(closest_wp)
            while j != 0:
                backward_point = np.array(combined_meta[j], dtype = float)[:3]
                if not np.allclose(closest_wp, backward_point, atol = tol):
                    break
                j -= 1
            
            k = i
            forward_point = np.array(closest_wp)
            while k < len(combined_meta) - 1:
                k += 1
                forward_point = np.array(combined_meta[k], dtype = float)[:3]
                if not np.allclose(closest_wp, forward_point, atol = tol):
                    break

            choose_backward = PathHandler._edge_opposite_test(coordinates[i], backward_point, closest_wp)
            choose_forward  = PathHandler._edge_opposite_test(coordinates[i], forward_point, closest_wp)
            if choose_backward == choose_forward: # Projected point too close to coordinates
                Q = coordinates[i]
            if choose_backward:
                # Project the coordinates[i] on to the line created by closest_wp and backward_point
                A = closest_wp
                B = backward_point
                P = np.array(coordinates[i], dtype = float)
                Q = PathHandler._project_point_to_segment(P, A, B)

            if choose_forward:
                # Project the coordinates[i] on to the line created by closest_wp and forward_point
                A = closest_wp
                B = forward_point
                P = np.array(coordinates[i], dtype = float)
                Q = PathHandler._project_point_to_segment(P, A, B)
            filtered_meta += [[Q[0], Q[1], Q[2], 0]]


        return np.array(filtered_meta)
        

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