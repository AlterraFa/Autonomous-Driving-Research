import carla
import numpy as np
import queue

from .stubs.sensor__camera__semantic_segmentation_stub import SensorCameraSemanticSegmentationStub
from .stubs.sensor__camera__rgb_stub import SensorCameraRgbStub
from typing import Optional
from enum import IntEnum

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
    def __init__(self, sensor_bp: carla.ActorBlueprint, world: carla.World):
        self.bp = sensor_bp
        self.world = world
        
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

    def destroy(self):
        if hasattr(self, "actor"):
            self.actor.stop()
            self.actor.destroy()
        
class SemanticSegmentation(SensorSpawn, SensorCameraSemanticSegmentationStub):
    
    palette = {
        0:  (0,   0,   0),      # Unlabeled
        1:  (128, 64,  128),    # Road
        2:  (244, 35,  232),    # Sidewalk
        3:  (70,  70,  70),     # Building
        4:  (102, 102, 156),    # Wall
        5:  (190, 153, 153),    # Fence
        6:  (153, 153, 153),    # Pole
        7:  (250, 170, 30),     # TrafficLight
        8:  (220, 220, 0),      # TrafficSign
        9:  (107, 142, 35),     # Vegetation
        10: (152, 251, 152),    # Terrain
        11: (70,  130, 180),    # Sky
        12: (220, 20,  60),     # Pedestrian
        13: (255, 0,   0),      # Rider
        14: (0,   0,   142),    # Car
        15: (0,   0,   70),     # Truck
        16: (0,   60,  100),    # Bus
        17: (0,   80,  100),    # Train
        18: (0,   0,   230),    # Motorcycle
        19: (119, 11,  32),     # Bicycle
        20: (110, 190, 160),    # Static
        21: (170, 120, 50),     # Dynamic
        22: (55,  90,  80),     # Other
        23: (45,  60,  150),    # Water
        24: (157, 234, 50),     # RoadLine
        25: (81,  0,   81),     # Ground
        26: (150, 100, 100),    # Bridge
        27: (230, 150, 140),    # RailTrack
        28: (180, 165, 180),    # GuardRail
    }
    
    def __init__(self, sensor_bp: carla.ActorBlueprint, world: carla.World):
        super().__init__(sensor_bp, world)
        self.sensor_bp = sensor_bp.find("sensor.camera.semantic_segmentation")
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


class RGB(SensorSpawn, SensorCameraRgbStub):
    def __init__(self, sensor_bp: carla.ActorBlueprint, world: carla.World):
        super().__init__(sensor_bp, world)
        self.sensor_bp = sensor_bp.find("sensor.camera.rgb")
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