import cv2
import carla
import numpy as np

from utils.control.world import World
from utils.messages.logger import Logger
from utils.math.path import _find_entry_clusters, _find_exit, waypoints_between
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
    
    def draw_on_map(self, image, box_color: tuple, waypoints_coordinates):
        for cx, cy, _, yaw in waypoints_coordinates:

            rect = ((cx + self.offset_x, cy + self.offset_y), (self.width, self.length), yaw)
            
            box  = cv2.boxPoints(rect)
            box  = box.astype(int)

            cv2.drawContours(image, [box], 
                             0, 
                             box_color, 
                             cv2.FILLED)
        return image

    def draw_waypoints_lines(self, image, waypoints, color=(0, 0, 255), line_thickness=2):
        for i in range(len(waypoints)-1):
            x1, y1 = int(waypoints[i][0] + self.offset_x), int(waypoints[i][1] + self.offset_y)
            x2, y2 = int(waypoints[i+1][0] + self.offset_x), int(waypoints[i+1][1] + self.offset_y)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness=line_thickness)
        
        # Optionally, draw small arrows to show heading
        for x, y, _, yaw in waypoints:
            start_pt = (int(x + self.offset_x), int(y + self.offset_y))
            end_pt = (
                int(start_pt[0] + 5 * np.cos(np.deg2rad(yaw))),
                int(start_pt[1] + 5 * np.sin(np.deg2rad(yaw)))
            )
            cv2.arrowedLine(image, start_pt, end_pt, color, 1, tipLength=0.3)

        return image
    
    def _render_map(self):
        
        # Create the map_image with padding to avoid going out of range
        self.map_image = np.zeros((int(self.new_max_y + self.offset_x * 2), int(self.max_x + self.offset_y * 2), 3), dtype = np.uint8)
        self.map_image = self.draw_on_map(self.map_image, (255, 255, 255), self.waypoints_metadata)
        
        self.map_image = cv2.GaussianBlur(self.map_image, (5, 5), sigmaX = 0) 
        kernel         = np.ones((3,3), np.uint8)
        self.map_image = cv2.morphologyEx(self.map_image, cv2.MORPH_CLOSE, kernel)

        self.canvas = self.map_image.copy()
    
    def retrieve_map(self, coordinate, heading, range_, resize_to=(50, 50)):
        x, y = coordinate
        x = int(x * self.scale + self.offset_x)
        y = int(y * self.scale - self.old_min_y + self.offset_y)
        
        H, W, _ = self.canvas.shape
        w, h = range_
        radius = int(((w / 2) ** 2 + (h / 2) ** 2) ** 0.5)

        # clamp once
        x1, x2 = max(0, x - radius), min(W, x + radius)
        y1, y2 = max(0, y - radius), min(H, y + radius)

        # First cutout uses radius to avoid missing lanes during rotation
        cutout = self.canvas[y1:y2, x1:x2].copy()

        cx, cy = x - x1, y - y1
        cos_t, sin_t = np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))
        M = np.float32([[cos_t, sin_t, (1 - cos_t) * cx - sin_t * cy],
                        [-sin_t, cos_t, sin_t * cx + (1 - cos_t) * cy]])

        rotated = cv2.warpAffine(cutout, M, (cutout.shape[1], cutout.shape[0]), flags=cv2.INTER_LINEAR)

        # one precise crop
        x1f, x2f = max(0, cx - w // 2), min(rotated.shape[1], cx + w // 2)
        y1f, y2f = max(0, cy - h // 2), min(rotated.shape[0], cy + h // 2)

        # Second cutout to refine to the correct range
        return cv2.resize(rotated[y1f:y2f, x1f:x2f], resize_to)
    
    def waypoints_to_canvas(self, waypoints_metadata):
        transformed = []
        for x, y, z, yaw in waypoints_metadata:
            cx = int(x * self.scale)
            cy = int(y * self.scale - self.old_min_y)
            transformed.append((cx, cy, z, yaw))
        return np.array(transformed)
    
    
    def routed_map(self, coordinates: np.ndarray):
        waypoints_metadata = []

        junctions = self.world.get_segments_from_points("junction", coordinates)
        junctions_metadata = []
        
        
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

        waypoints_metadata = self.waypoints_to_canvas(np.array(waypoints_metadata))
        self.canvas = self.map_image.copy()
        self.canvas = self.draw_waypoints_lines(
            self.canvas, waypoints_metadata, color=(255, 0, 0), line_thickness=4*self.scale
        )
        

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