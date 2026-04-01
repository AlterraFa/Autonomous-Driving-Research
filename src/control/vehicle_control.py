import os, sys
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import carla
import numpy as np

from traceback import print_exc
from scipy.signal import butter, lfilter, lfilter_zi
from src.messages.message_handler import MessageSubscriber, MessageSender
from src.messages.all_messages import (
    Throttle,
    Steer,
    Brake,
    Reverse,
    Handbrake,
    ModelSpeed,
    ModelSteer,
    SpeedLimit
)
from src.others.others import get_nested_config
from config import CONFIG

decay     = CONFIG.vehicle.decay
max_steer = CONFIG.vehicle.physics.max_steer
wheelbase = CONFIG.vehicle.physics.wheelbase
fs        = CONFIG.vehicle.control_filter.fs
x0        = CONFIG.vehicle.control_filter.x0
Kp        = CONFIG.vehicle.velocity_regulator.kp
Ki        = CONFIG.vehicle.velocity_regulator.ki
Kd        = CONFIG.vehicle.velocity_regulator.kd

class OnlineLowPassFilter:
    def __init__(self, cutoff, fs, order=2, x0=0.0):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)
        self.zi = lfilter_zi(self.b, self.a) * x0

    def step(self, x):
        y, self.zi = lfilter(self.b, self.a, [x], zi=self.zi)
        return y[0]


class Vehicle:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World, fps = 70):
        self._init_transmitter()
        
        self.vehicle = vehicle
        self.world = world
        self.map = self.world.get_map()
        self._autopilot = False
        self._model_autopilot = False
        self.set_autopilot(self._autopilot)
        self.set_model_autopilot(self._model_autopilot)

        self.throttle = 0
        self.steer = 0
        self.brake = 0
        self.throt_delta = 0
        self.steer_delta = 0
        self.brake_delta = 0
        
        self.hand_brake     = False
        self.reverse        = False
        self.regulate_speed = False

        self.throttle_filter = OnlineLowPassFilter(fs, fps, x0)
        self.brake_filter    = OnlineLowPassFilter(fs, fps, x0)

        
        self.prev_loc = self.vehicle.get_transform().location

        self.max_steer = max_steer
        self.wheelbase = wheelbase
    
    def _init_transmitter(self):
        """Initialize all message subscribers for vehicle control."""
        self.sub_throttle   = MessageSubscriber(Throttle)
        self.sub_steer      = MessageSubscriber(Steer)
        self.sub_brake      = MessageSubscriber(Brake)
        self.sub_reverse    = MessageSubscriber(Reverse)
        self.sub_handbrake  = MessageSubscriber(Handbrake)
        self.sub_model_speed = MessageSubscriber(ModelSpeed)
        self.sub_model_steer = MessageSubscriber(ModelSteer)
        
        self.send_speed_lim = MessageSender(SpeedLimit)
        
    def set_autopilot(self, enable: bool):
        self.vehicle.set_autopilot(enable)
        self._autopilot = enable
        
    def set_model_autopilot(self, enable: bool):
        self._model_autopilot = enable

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

        self.send_speed_lim.send(self.vehicle.get_speed_limit())
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
            "model_autopilot": self._model_autopilot,
            "regulate_speed": self.regulate_speed
        }
    
    def _regulate_speed_PID(self):
        model_speed = self.sub_model_speed.receive()
        limit = self.vehicle.get_speed_limit() if model_speed is None else model_speed
        current_v = self.get_velocity(False)

        error = limit - current_v  

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
    
    def _keyboard(self):
        throttle   = self.sub_throttle.receive()
        steer      = self.sub_steer.receive()
        brake      = self.sub_brake.receive()
        reverse    = self.sub_reverse.receive()
        hand_brake = self.sub_handbrake.receive()
        if throttle is not None:   self.throttle  += throttle
        if steer is not None:      self.steer     += steer
        if brake is not None:      self.brake     += brake
        if reverse is not None:    self.reverse    = reverse
        if hand_brake is not None: self.hand_brake = hand_brake

        self.throt_delta = throttle
        self.steer_delta = steer
        self.brake_delta = brake

        if throttle == 0:   # ease throttle back toward 0
            if self.throttle > 0:
                self.throttle = max(0.0, self.throttle - 0.03)
            elif self.throttle < 0:
                self.throttle = min(0.0, self.throttle + 0.03)

        if steer == 0:   # ease steering back toward 0
            if abs(self.steer) <= decay:
                self.steer = 0.0   # snap to neutral
            elif self.steer > 0:
                self.steer -= decay
            elif self.steer < 0:
                self.steer += decay

        if brake == 0:   # ease brake back toward 0
            if self.brake > 0:
                self.brake = max(0.0, self.brake - decay)

    def _joystick(self):
        throttle   = self.sub_throttle.receive()
        steer      = self.sub_steer.receive()
        brake      = self.sub_brake.receive()
        reverse    = self.sub_reverse.receive()
        hand_brake = self.sub_handbrake.receive()
        if throttle is not None:   self.throttle   = throttle
        if steer is not None:      self.steer      = steer
        if brake is not None:      self.brake      = brake
        if reverse is not None:    self.reverse    = reverse
        if hand_brake is not None: self.hand_brake = hand_brake
    
    def _clamp_ctrl(self):
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
        
    def apply_control(self, regulate_speed: bool = False, use_joystick: bool = False, using_model = False):
        self.regulate_speed = regulate_speed

        if use_joystick == False:
            self._keyboard()
        else: 
            self._joystick()

        if regulate_speed: self._regulate_speed_PID()
                
        if using_model:
            steer = self.sub_model_steer.receive()
            if steer is not None: self.steer = steer

        self._clamp_ctrl()

        self.vehicle.apply_control(carla.VehicleControl(throttle = self.throttle, 
                                                        steer = self.steer,
                                                        brake = self.brake,
                                                        reverse = self.reverse,
                                                        hand_brake = self.hand_brake,
                                                        manual_gear_shift = False,
                                                        gear = 0))
    