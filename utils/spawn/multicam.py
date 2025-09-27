import carla
import inspect

from utils.spawn.sensor_spawner import SensorSpawn
from utils.messages.logger import Logger
from typing import Optional


class MultiCamera:
    def __init__(self, world, camera_type: "SensorSpawn", camera_yaws = list[float], convert_to: Optional[callable] = None):
        
        self.log = Logger()
                
        self.quantity = len(camera_yaws)
        self.cameras  = {}
        self.main_camera = None
        sig = inspect.signature(camera_type.__init__)
        arg_names = [p for p in sig.parameters if p != "self"]

        for idx in range(self.quantity):
            camera = camera_type(world, convert_to = convert_to) if "convert_to" in arg_names else camera_type(world)
            if camera_yaws[idx] == 0:
                self.log.INFO("Found a camera with yaw = 0. Setting it as main camera")
                self.main_camera = camera
            else:
                self.cameras.update({camera: camera_yaws[idx]})
        
        if self.main_camera is None:
            self.log.DEBUG("Did not find a camera with yaw = 0. Automatically create one")
            self.main_camera = camera_type(world, convert_to = convert_to) if "convert_to" in arg_names else camera_type(world)
        