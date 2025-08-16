import carla
import numpy as np
import queue
import json

from .stubs.sensor__camera__semantic_segmentation_stub import SensorCameraSemanticSegmentationStub
from .stubs.sensor__camera__rgb_stub import SensorCameraRgbStub
from .stubs.sensor__lidar__ray_cast_stub import SensorLidarRayCastStub
from typing import Optional
from enum import IntEnum
from pathlib import Path
from rich import print

class CarlaLabel(IntEnum):
    Unlabeled     = 0
    Road          = 1
    SideWalk      = 2
    Building      = 3
    Wall          = 4
    Fence         = 5
    Pole          = 6
    TrafficLight  = 7
    TrafficSign   = 8
    Vegetation    = 9
    Terrain       = 10
    Sky           = 11
    Pedestrian    = 12
    Rider         = 13
    Car           = 14
    Truck         = 15
    Bus           = 16
    Train         = 17
    Motorcycle    = 18
    Bicycle       = 19
    Static        = 20
    Dynamic       = 21
    Other         = 22
    Water         = 23
    RoadLine      = 24
    Ground        = 25
    Bridge        = 26
    RailTrack     = 27
    GuardRail     = 28
    
class SensorSpawn(object):
    def __init__(self, name, sensor_bp: carla.ActorBlueprint, world: carla.World):
        self.sensor_bp = sensor_bp.find(name)
        self.world = world
        self.name = name
        self.literal_name = self.__literal_name__(self.name)
        
    def spawn(self, attach_to: None, **kwargs):
        loc = carla.Location(
            x=kwargs.get("x", 0.0),
            y=kwargs.get("y", 0.0),
            z=kwargs.get("z", 0.0)
        )

        # Extract rotation
        rot = carla.Rotation(
            roll=kwargs.get("roll", 0.0),
            pitch=kwargs.get("pitch", 0.0),
            yaw=kwargs.get("yaw", 0.0)
        )
        transform = carla.Transform(loc, rot)
        
        
        self.actor = self.world.spawn_actor(self.sensor_bp, transform, attach_to = attach_to)
        self.actor.listen(self.queue.put)
        print(f"[green][INFO][/]: Sensor [bold]{self.literal_name}[/bold] spawned successfully. Listening to it")

    def destroy(self):
        if hasattr(self, "actor"):
            self.actor.stop()
            self.actor.destroy()
    
    @staticmethod
    def __literal_name__(name: str):
        string = name.split(".")[1:][::-1]
        sensor_type = string[0]
        temp = []
        for word in sensor_type.split('_'):
            temp += [word.capitalize()]
        sensor_name = ' '.join(temp + [string[1].capitalize()])
        return sensor_name
        
class LidarRaycast(SensorLidarRayCastStub, SensorSpawn):
    def __init__(self, sensor_bp, world):
        super().__init__()
        SensorSpawn.__init__(self, self.name, sensor_bp, world)
        
        self.sensor_bp = sensor_bp.find(self.name)

class SemanticSegmentation(SensorCameraSemanticSegmentationStub, SensorSpawn):
    
    def __init__(self, sensor_bp: carla.ActorBlueprint, world: carla.World):
        super().__init__()
        SensorSpawn.__init__(self, self.name, sensor_bp, world)

        palette_autopath = Path(__file__).resolve().parent / "palette.json"
        with palette_autopath.open("r")  as f:
            palette = json.load(f)
        self.palette = {int(k): tuple(v) for k, v in palette.items()}
        
        self.sensor_bp = sensor_bp.find(self.name)
        self.world = world
        self.queue = queue.Queue()
        self.num_label = len(list(CarlaLabel))
        self._lut_data = self._build_lut_rgba()
        
    def extract_data(self, layers: Optional[list[CarlaLabel]] = None, alpha = 1):
        if not hasattr(self, "actor"):
            raise ReferenceError(f"Actor {self.__class__.__name__} has not been spawned")
        if layers is not None and not isinstance(layers, list):
            raise TypeError("layers arg must be a list of name of layers")

        data = self.queue.get()
        labels = self.__decode_semantic_labels__(data)

        if layers is None:
            lut = self._lut_data
        else:
            sel = np.zeros(self.num_label, dtype=bool)
            for lbl in layers:
                sel[int(lbl.value)] = True
            lut = self._lut_data.copy()
            lut[~sel] = 0

        overlay = lut[labels]

        a = int(round(max(0.0, min(1.0, float(alpha))) * 255))
        if a < 256:
            rgb = overlay[..., :3].astype(np.uint16)
            overlay[..., :3] = (rgb * a // 255).astype(np.uint8)
            overlay[..., 3] = 255

        return overlay  # RGBA uint8
    
    def _build_lut_rgba(self) -> np.ndarray:
        lut = np.zeros((self.num_label, 4), dtype=np.uint8)
        for k, (r, g, b) in self.palette.items():
            lut[k] = (r, g, b, 255)  # RGBA
        return lut
            

    def __decode_semantic_labels__(self, carla_image: carla.Image) -> np.ndarray:
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8).reshape(carla_image.height, carla_image.width, 4)
        labels = array[:, :, 2].copy()  # BGRA -> take R channel
        return labels


class RGB(SensorCameraRgbStub, SensorSpawn):
    def __init__(self, sensor_bp: carla.ActorBlueprint, world: carla.World):
        super().__init__()
        SensorSpawn.__init__(self, self.name, sensor_bp, world)

        self.sensor_bp = sensor_bp.find(self.name)
        self.world = world
        self.queue = queue.Queue()
    
    def extract_data(self):
        if not hasattr(self, "actor"):
            raise ReferenceError(f"Actor {self.__class__.__name__} has not been spawned")
        
        data = self.queue.get()
        frame = self.__decode_frame__(data)
        return frame
    
    @staticmethod
    def __decode_frame__(carla_image: carla.Image) -> np.ndarray:
        array = np.frombuffer(carla_image.raw_data, dtype = np.uint8).reshape(carla_image.height, carla_image.width, 4)
        return array