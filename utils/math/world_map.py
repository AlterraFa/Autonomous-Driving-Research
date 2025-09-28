import cv2
import carla
import numpy as np

from utils.control.world import World
from utils.messages.logger import Logger
from utils.math.path import _find_entry_clusters, _find_exit, waypoints_between, PathHandler
from scipy.spatial import cKDTree

class Map:
    def __init__(self, world: World, rect_dim: tuple, map_offset: tuple = (0, 0), scale: int = 10, ):
        self.log = Logger()
        self.world = world

        carla_map = world.world.get_map()
        waypoints = carla_map.generate_waypoints(distance=2.0)
        self.wp_dict = {(wp.transform.location.x, wp.transform.location.y): wp for wp in waypoints}
        waypoints_metadata = []
        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            yaw = wp.transform.rotation.yaw
            waypoints_metadata += [[loc.x, loc.y, loc.z, yaw]]
        waypoints_metadata = np.array(waypoints_metadata)
        
        self._map = carla_map


        self.log.DEBUG("Found waypoints metadata")        
        # Scale up for fine grain detail
        waypoints_metadata[:, 0] *= scale   # x
        waypoints_metadata[:, 1] *= scale   # y

        # Min/max using numpy instead of describe()
        self.min_x, self.max_x = waypoints_metadata[:, 0].min(), waypoints_metadata[:, 0].max()
        self.old_min_y, self.old_max_y = waypoints_metadata[:, 1].min(), waypoints_metadata[:, 1].max()

        # Shift y to start from zero
        waypoints_metadata[:, 1] -= self.old_min_y

        self.new_min_y, self.new_max_y = waypoints_metadata[:, 1].min(), waypoints_metadata[:, 1].max()

        # Store for later
        self.waypoints_metadata = waypoints_metadata

        # Draw the map_image using rectangles
        self.length, self.width      = rect_dim[0] * scale, rect_dim[1] * scale
        self.offset_x, self.offset_y = map_offset[0] * scale, map_offset[1] * scale
        self.scale = scale

        self.stored_entries = {}  # junction_id -> entry_wp
        self._wp_list = list(self.wp_dict.values())
        self._wps = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in self._wp_list])
        self._tree = cKDTree(self._wps)

        self._render_map()
        
    def precompute_waypoints(self, replay_file: str):
        
        trajectories = np.load(replay_file)
        points = self.waypoints_compute(trajectories[:, :3])
        if len(points) == 0:
                return points

        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        # Always keep first point, then keep if distance > tol
        mask = np.insert(dists > 1e-2, 0, True)
        self.path_handler = PathHandler(points[mask][:, :3])
        self.offset_path  = [i for i in range(-100, 100, 2)]

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
        for i in range(len(waypoints)-1):
            x1, y1 = int(waypoints[i][0]), int(waypoints[i][1])
            x2, y2 = int(waypoints[i+1][0]), int(waypoints[i+1][1])
            cv2.line(image, (x1, y1), (x2, y2), color, thickness=line_thickness, lineType = cv2.LINE_AA)
    
    def _render_map(self):
        
        # Create the map_image with padding to avoid going out of range
        self.map_image = np.zeros((int(self.new_max_y + self.offset_x * 2), int(self.max_x + self.offset_y * 2), 3), dtype = np.uint8)
        self.map_image = self.draw_map(self.map_image, (255, 255, 255), self.waypoints_metadata)
        
        self.map_image = cv2.GaussianBlur(self.map_image, (5, 5), sigmaX = 0) 
        kernel         = np.ones((3,3), np.uint8)
        self.map_image = cv2.morphologyEx(self.map_image, cv2.MORPH_CLOSE, kernel)

    
    @profile
    def retrieve_map(self, coordinate, heading, range_, resize_to=(50, 50)):
        """Instead of drawing on the larger self.map_image, we draw on the smaller cutout image and apply waypoints transformation"""
        x, y, z = coordinate
        before_scale = np.array(coordinate)

        # ================ Retrieve Submap ======================
        # Retrieve the normal submap with the same transformation as __init__
        x = int(x * self.scale + self.offset_x)
        y = int(y * self.scale - self.old_min_y + self.offset_y)
        
        H, W, _ = self.map_image.shape
        w, h = range_
        radius = int(((w / 2) ** 2 + (h / 2) ** 2) ** 0.5)

        # clamp once
        x1, x2 = max(0, x - radius), min(W, x + radius)
        y1, y2 = max(0, y - radius), min(H, y + radius)

        # First cutout uses radius to avoid missing lanes during rotation
        cutout = self.map_image[y1:y2, x1:x2]

        cx, cy = x - x1, y - y1
        cos_t, sin_t = np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))
        M = np.float32([[cos_t, sin_t, (1 - cos_t) * cx - sin_t * cy],
                        [-sin_t, cos_t, sin_t * cx + (1 - cos_t) * cy]])

        rotated = cv2.warpAffine(cutout, M, (cutout.shape[1], cutout.shape[0]), flags=cv2.INTER_LINEAR)

        # one precise crop
        x1f, x2f = max(0, cx - w // 2), min(rotated.shape[1], cx + w // 2)
        y1f, y2f = max(0, cy - h // 2), min(rotated.shape[0], cy + h // 2)

        # Second cutout to refine to the correct range
        cutout = rotated[y1f:y2f, x1f:x2f]
        
        # ================= Draw path on map ====================
        
        if hasattr(self, "path_handler"):
            # Extract the waypoints using interpolation, globals only, locals waypoints embedded in waypoint code will mess up rotation and transformation
            # We don't need yaw, yaw = 0 is a dummy value        
            global_wp = self.path_handler.waypoints(
                before_scale, self.offset_path, yaw = 0, return_local = False
            )
            
            pts_world = np.atleast_2d(global_wp)[:, :2].astype(float)  # (N,2)
            # Same transformation
            pts_pix = np.empty_like(pts_world, dtype=float)
            pts_pix[:, 0] = pts_world[:, 0] * self.scale + self.offset_x
            pts_pix[:, 1] = pts_world[:, 1] * self.scale - self.old_min_y + self.offset_y

            pts_in_cutout = pts_pix - np.array([x1, y1], dtype=float)   # (N,2)

            ones = np.ones((pts_in_cutout.shape[0], 1), dtype=float)
            pts_hom = np.hstack([pts_in_cutout, ones])                  # (N,3)
            pts_trans = (pts_hom @ M.T)                                 # (N,2)

            pts_final = pts_trans - np.array([x1f, y1f], dtype=float)   # (N,2)

            self.draw_waypoints_lines(cutout, pts_final, color = (255, 0, 0), line_thickness = 3 * self.scale)

        return cv2.resize(cutout, resize_to)
    
    def waypoints_to_canvas(self, waypoints_metadata):
        waypoints_metadata[:, 0] = waypoints_metadata[:, 0] * self.scale
        waypoints_metadata[:, 1] = waypoints_metadata[:, 1] * self.scale - self.old_min_y
        return waypoints_metadata
    
    def waypoints_compute(self, coordinates: np.ndarray):
        """This code was initially used for running online
        Now it is used for precompute waypoints"""
        waypoints_metadata = []

        junctions = self.world.get_segments_from_points("junction", coordinates)
        junctions_metadata = []
        
        # This loop initially used caching because it was meant to run online
        # Now it is needed to precompute the path through junctions
        # Avoids KDTree snapping errors in dense areas by explicitly following entry→exit pairs
        # Caches entry clusters per junction, but clears them once the ego passes the exit
        for junction in junctions:
            jid = junction.id

            if jid in self.stored_entries:
                possible_pairs = self.stored_entries[jid]
            else:
                wp_pairs = junction.get_waypoints(carla.LaneType.Driving)
                possible_pairs = _find_entry_clusters(wp_pairs, coordinates)
                self.stored_entries[jid] = possible_pairs
            
            # Dynamic exit
            choosen_pairs     = _find_exit(possible_pairs, coordinates)
            entry_wp, exit_wp = choosen_pairs

            # Collect waypoints inside junction
            wp_in_junctions = waypoints_between(entry_wp, exit_wp)

            # Clear stored entry if vehicle passed exit
            ego_pos = coordinates[0][:2]
            exit_pos = np.array([exit_wp.transform.location.x, exit_wp.transform.location.y])
            if np.linalg.norm(exit_pos - ego_pos) < 1.0:
                self.stored_entries.pop(jid, None)           

            
            for wp in wp_in_junctions:
                loc = wp.transform.location
                yaw = wp.transform.rotation.yaw
                junctions_metadata.append([loc.x, loc.y, loc.z, yaw])


        junctions_metadata = np.array(junctions_metadata)

        # Compute waypoints mixed with junctions' waypoints
        for coordinate in coordinates:
            x, y, z = coordinate
            _, idx = self._tree.query([x, y])
            closest_wp = self._wp_list[idx]

            if closest_wp.is_junction and junctions_metadata.size > 0:
                distances = np.linalg.norm(junctions_metadata[:, :2] - np.array([x, y]), axis=1)
                closest_idx = distances.argmin()
                loc_x, loc_y, loc_z, yaw = junctions_metadata[closest_idx]
            else:
                loc = closest_wp.transform.location
                yaw = closest_wp.transform.rotation.yaw
                loc_x, loc_y, loc_z = loc.x, loc.y, loc.z

            waypoints_metadata.append([loc_x, loc_y, loc_z, yaw])

        waypoints_metadata = np.array(waypoints_metadata)
        return waypoints_metadata
        

if __name__ == "__main__":
    map_processor = Map((6, 4), map_offset = (5, 5))

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
        H, W = map_processor.map_image.shape
        # apply scale
        scaled = cv2.resize(map_processor.map_image, (int(W) // 1, int(H) // 1))

        # create black background of original size
        view = np.zeros((H, W), dtype=np.uint8)

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
        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()