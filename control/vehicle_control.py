import carla
import numpy as np
import threading
import time
import functools

from traceback import print_exc

class Vehicle:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
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
        
        self.hand_brake = False
        self.reverse    = False

        self.decay = 0.08
        

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.tl_thread = threading.Thread(target = self._poll_traffic_light, daemon = True); self.tl_state = None
        self.tl_thread.start()
        self.ts_thread = threading.Thread(target = self._poll_traffic_sign, daemon = True); self.ts_events = []
        self.ts_thread.start()
        self.junction_thread = threading.Thread(target = self._poll_junction, daemon = True)
        self.junction_thread.start()
        
    def stop(self):
        self._stop.set()
        if self.tl_thread.is_alive():
            self.tl_thread.join(timeout = 1.0)
        if self.ts_thread.is_alive():
            self.ts_thread.join(timeout = 1.0)
        if self.junction_thread.is_alive():
            self.junction_thread.join(timeout = 1.0)

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

        return np.sqrt(vel_vec.x ** 2 + vel_vec.y ** 2 + vel_vec.z ** 2) * 3.6
    
    def get_ctrl(self):
        control = self.vehicle.get_control()

        throttle   = control.throttle
        steer      = control.steer
        brake      = control.brake
        reverse    = control.reverse
        handbrake  = control.hand_brake
        manual     = control.manual_gear_shift
        gear       = control.gear

        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "reverse": reverse,
            "handbrake": handbrake,
            "manual": manual,
            "gear": gear,
            "autopilot": self._autopilot
        }
        
    def apply_control(self, throt_delta: float, steer_delta: float, brake_delta: float, reverse: bool, hand_brake: bool, regulate_speed: bool):
        
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

        if regulate_speed and self.get_velocity(False) >= self.vehicle.get_speed_limit() and brake_delta == 0:
            self.throttle = .3
            self.brake = .2
        
        
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
                time.sleep(0.05)

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
                    time.sleep(period); continue
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
                with self._lock:
                    self.ts_events = events

                
            except Exception:
                time.sleep(period)
                
            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)
            
    def _poll_junction(self, lookahead_m = 60.0, hz: float = 10.0):
        period = 1 / hz
        while not self._stop.is_set():
            t0 = time.time() 

            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)