import carla
import inspect
import numpy as np
import threading
import time

from src.spawn.sensor_spawner import SensorSpawn
from src.messages import Logger

from typing import Optional, Any
from collections.abc import Iterable


class MultiCamera:
    def __init__(self, world, camera_type: "SensorSpawn", quantity, convert_to: Optional[callable] = None):
        """
        Initialize multiple cameras of the same type.
        - world: carla.World
        - camera_type: class (subclass of SensorSpawn)
        - quantity: number of cameras to spawn
        - convert_to: optional conversion function for sensor output
        """
        self.log = Logger()
        self.world = world
        self.quantity = quantity
        self.sub_cameras = {}

        sig = inspect.signature(camera_type.__init__)
        arg_names = [p for p in sig.parameters if p != "self"]

        for _ in range(self.quantity):
            camera = camera_type(world, convert_to=convert_to) if "convert_to" in arg_names else camera_type(world)
            if "camera" not in camera.name:
                self.log.ERROR(
                    f"Sensor [u][bold]{camera.literal_name}[/][/] is not a camera",
                    exit_code=12
                )
            self.sub_cameras.update({camera: None})

        self.log.CUSTOM(
            "SUCCESS",
            f"Initialized {self.quantity} cameras of type [bold][u]{camera.literal_name}[/][/]"
        )
        
        camera_tname = camera.name.split('.')[-1]
        self.name = camera.name.replace("camera", "multicamera").replace(camera_tname, "multi" + camera_tname)
        self.literal_name = f"Multi {camera.literal_name}"

    def set_attribute(self, name: str, value: list | Any):
        """Set attributes for all sub cameras"""
        cameras = list(self.sub_cameras.keys())
        n = len(cameras)

        if isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
            if len(value) != n:
                self.log.ERROR(
                    f"Length of attribute list ({len(value)}) does not match number of cameras ({n})",
                    exit_code=13
                )
            for camera, val in zip(cameras, value):
                camera.set_attribute(name, val)
        else:
            for camera in cameras:
                camera.set_attribute(name, value)

    def spawn(self, attach_to=None, **kwargs):
        """
        Spawn all sub cameras.
        kwargs may include x, y, z, roll, pitch, yaw
        - If a value is iterable, each camera gets the corresponding element.
        - If a value is scalar, it is applied to all cameras.
        """
        cameras = list(self.sub_cameras.keys())
        n = len(cameras)

        expanded_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, Iterable) and not isinstance(val, (bytes, str)):
                if len(val) != n:
                    self.log.ERROR(
                        f"Length of iterable for '{key}' ({len(val)}) does not match number of cameras ({n})",
                        exit_code=14
                    )
                expanded_kwargs[key] = val
            else:
                expanded_kwargs[key] = [val] * n

        for idx, camera in enumerate(cameras):
            per_camera_kwargs = {k: v[idx] for k, v in expanded_kwargs.items()}
            camera.spawn(attach_to, **per_camera_kwargs)
            self.sub_cameras[camera] = per_camera_kwargs

    def extract_data(self, idx: Optional[int] = None):
        """
        Extract latest decoded frames.
        - If idx is None: returns dict {camera: frame}.
        - If idx is given: returns frame for that camera only.
        """
        cameras = list(self.sub_cameras.keys())

        if idx is not None:
            if idx < 0 or idx >= len(cameras):
                raise IndexError(f"Camera index {idx} out of range")
            return cameras[idx].extract_data()

        # all frames
        data_buffer = {}
        for camera in cameras:
            try:
                frame = camera.extract_data()
            except ReferenceError:
                frame = None
            data_buffer[camera] = frame
        return data_buffer

    def destroy(self):
        """Destroy all sub cameras"""
        for camera in self.sub_cameras:
            camera.destroy()
