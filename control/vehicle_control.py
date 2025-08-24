import carla
import numpy as np
import threading

class Vehicle:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        self.vehicle = vehicle
        self.world = world
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
            if self.throttle > 0.7: self.throttle = 0.7
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