import carla
import numpy as np
import threading
import time
import math

from traceback import print_exc
from scipy.signal import butter, lfilter, lfilter_zi

class OnlineLowPassFilter:
    def __init__(self, cutoff, fs, order=2, x0=0.0):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)
        self.zi = lfilter_zi(self.b, self.a) * x0

    def step(self, x):
        y, self.zi = lfilter(self.b, self.a, [x], zi=self.zi)
        return y[0]

class EMAFilter:
    def __init__(self, alpha=0.1, x0=0.0):
        self.alpha = alpha
        self.y = x0

    def step(self, x):
        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y

class Vehicle:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World, fps = 70):
        self.vehicle = vehicle
        self.world = world
        self.map = self.world.get_map()
        self._autopilot = True
        self.set_autopilot(self._autopilot)
        self.throttle = 0
        self.steer = 0
        self.brake = 0
        self.throt_delta = 0
        self.steer_delta = 0
        self.brake_delta = 0
        
        self.hand_brake     = False
        self.reverse        = False
        self.regulate_speed = False

        self.decay = 0.2
        
        self.throttle_filter = OnlineLowPassFilter(2.0, fps, 2)
        self.brake_filter    = OnlineLowPassFilter(2.0, fps, 2)

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.tl_thread = threading.Thread(target = self._poll_traffic_light, daemon = True); self.tl_state = None
        self.tl_thread.start()
        self.ts_thread = threading.Thread(target = self._poll_traffic_sign, daemon = True); self.ts_events = []
        self.ts_thread.start()
        self.junction_thread = threading.Thread(target = self._poll_junction, daemon = True); self.junctions = {}
        self.junction_thread.start()
        self.waypoint_thread = threading.Thread(target = self._poll_waypoint, daemon = True); self.waypoint = None
        self.waypoint_thread.start()
        self.location_thread = threading.Thread(target = self._poll_location, daemon = True); self.location = carla.Location(x = 0, y = 0, z = 0)
        self.location_thread.start()
        self.threads = [self.tl_thread, self.ts_thread, self.junction_thread, self.waypoint_thread, self.location_thread]
        
        self.prev_loc = self.vehicle.get_transform().location
        
        
    def stop(self):
        self._stop.set()
        for thread in self.threads:
            if thread.is_alive(): thread.join(timeout = 1.0)

    def set_autopilot(self, enable: bool):
        self.vehicle.set_autopilot(enable)
        self._autopilot = enable

    def literal_name(self):
        id = self.vehicle.type_id
        first_name, last_name = id.split(".")[1:]
        first_name = first_name.capitalize(); 
        last_name = " ".join([name.capitalize() for name in last_name.split("_")])
        return first_name + " " + last_name

    def get_velocity(self, return_vec: bool = True):
        vel_vec = self.vehicle.get_velocity()
        if return_vec:
            return np.array([vel_vec.x, vel_vec.y, vel_vec.z]) * 3.6

        speed = np.sqrt(vel_vec.x ** 2 + vel_vec.y ** 2 + vel_vec.z ** 2) * 3.6
        if speed < 1e-1:
            curr     = self.vehicle.get_transform().location
            distance = curr.distance(self.prev_loc)
            self.prev_loc = curr
            
            dt = self.world.get_snapshot().timestamp.delta_seconds
            speed = (distance / dt) * 3.6 # Scaled by some factor (close to 3.6 (conversion from m/s to km/h))
        return speed
    
    
    def get_ctrl(self, filter = False):
        control = self.vehicle.get_control()

        throttle   = control.throttle
        steer      = control.steer
        brake      = control.brake
        reverse    = control.reverse
        handbrake  = control.hand_brake
        manual     = control.manual_gear_shift
        gear       = control.gear

        if filter:
            throttle   = self.throttle_filter.step(throttle)
            brake      = self.brake_filter.step(brake)

        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "reverse": reverse,
            "handbrake": handbrake,
            "manual": manual,
            "gear": gear,
            "autopilot": self._autopilot,
            "regulate_speed": self.regulate_speed
        }
        
    def apply_control(self, throt_delta: float, steer_delta: float, brake_delta: float, reverse: bool, hand_brake: bool, regulate_speed: bool, use_joystick: bool = False):
        self.regulate_speed = regulate_speed

        if use_joystick == False:
            self.throttle += throt_delta
            self.steer += steer_delta
            self.brake += brake_delta
            self.reverse = reverse
            self.hand_brake = hand_brake

            self.throt_delta = throt_delta
            self.steer_delta = steer_delta
            self.brake_delta = brake_delta

            if throt_delta == 0:   # ease throttle back toward 0
                if self.throttle > 0:
                    self.throttle = max(0.0, self.throttle - 0.03)
                elif self.throttle < 0:
                    self.throttle = min(0.0, self.throttle + 0.03)

            if steer_delta == 0:   # ease steering back toward 0
                if abs(self.steer) <= self.decay:
                    self.steer = 0.0   # snap to neutral
                elif self.steer > 0:
                    self.steer -= self.decay
                elif self.steer < 0:
                    self.steer += self.decay

            if brake_delta == 0:   # ease brake back toward 0
                if self.brake > 0:
                    self.brake = max(0.0, self.brake - self.decay)

        else: 
            self.throttle = throt_delta
            self.steer = steer_delta
            self.brake = brake_delta
            self.reverse = reverse
            self.hand_brake = hand_brake


        if regulate_speed:
            limit = self.vehicle.get_speed_limit()
            current_v = self.get_velocity(False)

            error = limit - current_v  

            Kp = 0.1
            Ki = 0.05
            Kd = 0.05

            dt = self.world.get_settings().fixed_delta_seconds or 0.05

            if not hasattr(self, "error_sum"):
                self.error_sum = 0.0
            if not hasattr(self, "last_error"):
                self.last_error = error

            self.error_sum += error * dt
            self.error_sum = max(min(self.error_sum, 10.0), -10.0)

            d_error = (error - self.last_error) / dt
            self.last_error = error

            u = Kp * error + Ki * self.error_sum + Kd * d_error

            if u >= 0:
                self.throttle = max(0.0, min(1.0, u))
                self.brake = 0.0
            else:
                self.throttle = 0.0
                self.brake = max(0.0, min(1.0, -u))

        
        if self._autopilot:
            self.throttle = 0
            self.steer = 0
            self.brake = 0
            self.hand_brake = False
            self.reverse = False
        else: 
            if self.throttle > 1.0: self.throttle = 1.0
            if abs(self.steer) > 1: self.steer = 1.0 * (self.steer / abs(self.steer))
            if self.brake > 1: self.brake = 1.0

            if self.throttle < 0: self.throttle = 0
            if self.brake < 0: self.brake = 0

        self.vehicle.apply_control(carla.VehicleControl(throttle = self.throttle, 
                                                        steer = self.steer,
                                                        brake = self.brake,
                                                        reverse = self.reverse,
                                                        hand_brake = self.hand_brake,
                                                        manual_gear_shift = False,
                                                        gear = 0))
    
    def _poll_location(self, hz: float = 10.0, move_thresh_m = 1.0):
        period = 1 / hz
        last_loc = None
        while not self._stop.is_set():
            t0 = time.time()
            try:
                curr_loc = self.vehicle.get_location()
                if last_loc is not None and curr_loc.distance(last_loc) < move_thresh_m:
                    dt = time.time() - t0
                    if dt < period:
                        time.sleep(period - dt)
                    continue
                else:
                    last_loc = curr_loc
                
                with self._lock: self.location = last_loc
            except:
                print_exc()

            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)
        
    def _poll_traffic_light(self, hz: float = 10.0):
        period = 1 / hz
        while not self._stop.is_set():
            t0 = time.time()
            try:
                tl = self.vehicle.get_traffic_light()
                if tl is None:
                    self.tl_state = None
                else:
                    self.tl_state = tl.get_state()
            except:
                print_exc()

            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)
            
            
    @staticmethod
    def format_waypoint(wp: carla.Waypoint):
        if wp is None:
            return None
        tr = wp.transform
        return {
            "road_id": wp.road_id,
            "section_id": wp.section_id,
            "lane_id": wp.lane_id,
            "s": wp.s,  # longitudinal position along road
            "location": {
                "x": round(tr.location.x, 2),
                "y": round(tr.location.y, 2),
                "z": round(tr.location.z, 2),
            },
            "rotation": {
                "yaw": round(tr.rotation.yaw, 1),
                "pitch": round(tr.rotation.pitch, 1),
                "roll": round(tr.rotation.roll, 1),
            }
        }

    
    def _poll_traffic_sign(self, lookahead_m = 60.0, hz: float = 10.0, move_thresh_m: float = 2.0):
        period = 1 / hz
        last_loc = None
        while not self._stop.is_set():
            t0 = time.time()
            try:
                wp  = self.map.get_waypoint(self.vehicle.get_location(), project_to_road = True,
                                            lane_type = carla.LaneType.Driving)


                if last_loc is not None and wp.transform.location.distance(last_loc) < move_thresh_m:
                    dt = time.time() - t0
                    if dt < period:
                        time.sleep(period - dt)
                    continue
                last_loc = wp.transform.location
                
                stops    = wp.get_landmarks_of_type(lookahead_m, carla.LandmarkType.StopSign, stop_at_junction = False)
                yields   = wp.get_landmarks_of_type(lookahead_m, carla.LandmarkType.YieldSign, stop_at_junction = False)

                lms = list(stops) + list(yields)
                
                events = []
                for lm in lms:
                    try:
                        lm_wp = self.map.get_waypoint(
                            lm.transform.location,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving
                        )
                    except Exception:
                        lm_wp = None
                    
                    d = getattr(lm, "distance", float("nan"))
                    if np.isnan(d) and lm_wp is not None:
                        d = wp.transform.location.distance(lm_wp.transform.location)
                    
                    events.append({
                        "type": lm.type,
                        "id": lm.id,
                        "distance": d,
                        "waypoint": lm_wp
                    })
                
                events.sort(key = lambda e: e["distance"])
                with self._lock: self.ts_events = events

                
            except Exception:
                time.sleep(period)
                
            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)
            
    def _poll_waypoint(self, hz: float = 10.0, move_thresh_m: float = 2.0):
        period = 1 / hz
        last_loc = None
        self.waypoint = None
        while not self._stop.is_set():
            t0 = time.time() 
            try:
                wp = self.map.get_waypoint(
                    self.vehicle.get_location(),
                    project_to_road = True,
                    lane_type = carla.LaneType.Driving
                )
                
                if last_loc is not None and wp.transform.location.distance(last_loc) < move_thresh_m:
                    dt = time.time() - t0
                    if dt < period: time.sleep(period - dt)
                    continue
                last_loc = wp.transform.location

                with self._lock: self.waypoint = wp

            except:
                ...

            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)

            
    def _poll_junction(self, lookahead_m = 5.0, hz: float = 10.0, move_thresh_m: float = 2.0):
        period = 1 / hz
        last_loc = None
        while not self._stop.is_set():
            t0 = time.time() 
            try:
                wp = self.map.get_waypoint(
                    self.vehicle.get_location(),
                    project_to_road = True,
                    lane_type = carla.LaneType.Driving
                )
                
                if last_loc is not None and wp.transform.location.distance(last_loc) < move_thresh_m:
                    dt = time.time() - t0
                    if dt < period: time.sleep(period - dt)
                    continue
                last_loc = wp.transform.location
                
                jsx = self.find_upcoming_junctions(
                    lookahead_m = lookahead_m, 
                    step_m = 2.0,
                    fov_half_deg = 30,
                    max_junctions = 10
                ) 
                
                with self._lock:  self.junctions = jsx
                
            except:
                ...

            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)
            
    def find_upcoming_junctions(self,
                            lookahead_m: float = 120.0,
                            step_m: float = 2.0,
                            fov_half_deg: float = 30.0,
                            max_junctions: int = 10):
        """
        March forward along the current lane, collecting distinct junctions ahead.
        Filters by forward FOV (±fov_half_deg) relative to vehicle heading.
        Returns a list sorted by distance.
        """
        ego_loc = self.vehicle.get_location()
        ego_pos2 = _vec2(ego_loc)
        ego_dir2 = _forward2(self.vehicle.get_transform())

        wp = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return []

        total = 0.0
        seen = set()
        found = []

        prev_dir = _forward2(wp.transform)

        if wp.is_junction:
            j = wp.get_junction()
            if j and j.id not in seen:
                center = j.bounding_box.location
                vec_to_center = _vec2(center) - ego_pos2
                ang = _angle_deg(ego_dir2, vec_to_center)
                if ang <= fov_half_deg:
                    d = float(np.linalg.norm(vec_to_center))
                    found.append({
                        "junction_id": j.id,
                        "distance_m": round(d, 2),
                        "angle_deg": round(ang, 1),
                        "center": (round(center.x, 2), round(center.y, 2), round(center.z, 2)),
                        "bbox_extents": (round(j.bounding_box.extent.x, 2), round(j.bounding_box.extent.y, 2), round(j.bounding_box.extent.z, 2)),
                        "entry_waypoint": self.format_waypoint(wp)
                    })
                    seen.add(j.id)

        steps = int(max(1, math.ceil(lookahead_m / max(step_m, 0.2))))
        cur_wp = wp
        for _ in range(steps):
            nxt_wp = _next_along(cur_wp, step_m, prev_dir)
            if nxt_wp is None:
                break
            prev_dir = _forward2(nxt_wp.transform)

            total += step_m
            cur_wp = nxt_wp

            if cur_wp.is_junction:
                j = cur_wp.get_junction()
                if j and j.id not in seen:
                    center = j.bounding_box.location
                    vec_to_center = _vec2(center) - ego_pos2
                    ang = _angle_deg(ego_dir2, vec_to_center)

                    if ang <= fov_half_deg:
                        d = float(np.linalg.norm(vec_to_center))
                        found.append({
                            "junction_id": j.id,
                            "distance_m": round(d, 2),
                            "angle_deg": round(ang, 1),
                            "center": (round(center.x, 2), round(center.y, 2), round(center.z, 2)),
                            "bbox_extents": (round(j.bounding_box.extent.x, 2), round(j.bounding_box.extent.y, 2), round(j.bounding_box.extent.z, 2)),
                            "entry_waypoint": self.format_waypoint(cur_wp)
                        })
                        seen.add(j.id)
                        if len(found) >= max_junctions:
                            break

        found.sort(key=lambda e: e["distance_m"])
        return found

def _vec2(loc: carla.Location):
    return np.array([loc.x, loc.y], dtype=np.float32)

def _forward2(tr: carla.Transform):
    f = tr.get_forward_vector()
    return np.array([f.x, f.y], dtype=np.float32)

def _angle_deg(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    nu = u / (np.linalg.norm(u) + eps)
    nv = v / (np.linalg.norm(v) + eps)
    c = float(np.clip(np.dot(nu, nv), -1.0, 1.0))
    return math.degrees(math.acos(c))

def _pick_straightest(prev_dir: np.ndarray, candidates: list[carla.Waypoint]) -> carla.Waypoint:
    """Choose candidate whose heading deviates least from prev_dir."""
    best = None
    best_ang = 1e9
    for wp in candidates:
        ang = _angle_deg(prev_dir, _forward2(wp.transform))
        if ang < best_ang:
            best, best_ang = wp, ang
    return best

def _next_along(current_wp: carla.Waypoint, step: float, prev_dir: np.ndarray) -> carla.Waypoint | None:
    nxt = current_wp.next(step)
    if not nxt:
        return None
    if len(nxt) == 1:
        return nxt[0]
    return _pick_straightest(prev_dir, nxt)

def wait_for_actor_by_role(world: carla.World, role_name: str, timeout_s: float = 10.0) -> carla.Actor | None:
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        vehicles = world.get_actors().filter('vehicle.*')
        for v in vehicles:
            try:
                if v.attributes.get('role_name', '') == role_name:
                    return v
            except Exception:
                pass
        # advance world in sync or just sleep in async
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            time.sleep(0.05)
    return None