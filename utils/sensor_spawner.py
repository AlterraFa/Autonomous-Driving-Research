import carla
import numpy as np
import queue
import json
import threading

from .stubs.sensor__camera__semantic_segmentation_stub import SensorCameraSemanticSegmentationStub
from .stubs.sensor__camera__rgb_stub import SensorCameraRgbStub
from .stubs.sensor__lidar__ray_cast_stub import SensorLidarRayCastStub
from .stubs.sensor__other__gnss_stub import SensorOtherGnssStub
from .lidar_visualization import LIDARVisualizer
from config.enum import CarlaLabel
from typing import Optional
from pathlib import Path
from rich import print
from collections.abc import Mapping
from dataclasses import dataclass
    
class SensorSpawn(object):
    def __init__(self, name, world: carla.World):
        self.name         = name
        self.sensor_bp    = world.get_blueprint_library().find(self.name)
        self.world        = world
        self.literal_name = self.__literal_name__(self.name)
        self.actor        = None
        
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
        self.actor.listen(self.callback.put)
        print(f"[green][INFO][/]: Sensor [bold]{self.literal_name}[/bold] spawned successfully. Listening to it")

    def destroy(self):
        if hasattr(self, "actor"):
            self.actor.stop()
            self.actor.destroy()
            print(f"[green][INFO][/]: Sensor [bold]{self.literal_name}[/bold] stopped and destroyed")
    
    @staticmethod
    def __literal_name__(name: str):
        string = name.split(".")[1:][::-1]
        sensor_type = string[0]
        temp = []
        for word in sensor_type.split('_'):
            temp += [word.capitalize()]
        sensor_name = ' '.join(temp + [string[1].capitalize()])
        return sensor_name
        
class LidarRaycast(SensorLidarRayCastStub, SensorSpawn, LIDARVisualizer):
    def __init__(self, world, vis_range = 50, vis_window = (800, 600)):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)
        LIDARVisualizer.__init__(self, vis_range, vis_window)
        
        self.sensor_bp = world.get_blueprint_library().find(self.name)
        self.callback = queue.Queue()
        self.pc = np.zeros((1, 3))
    
    def extract_data(self):
        if not hasattr(self, "actor"):
            raise ReferenceError(f"Actor {self.__class__.__name__} has not been spawned")

        data         = self.callback.get()
        decoded_data = self.__decode_data__(data)
        self.pcd     = decoded_data[:, :-1]
        self.intense = decoded_data[:, -1]
        return self.pc, self.intense
    
    def __decode_data__(self, carla_pointcloud: carla.Image) -> np.ndarray:
        return np.frombuffer(carla_pointcloud.raw_data, dtype = np.float32).reshape((-1, 4))
    
    def visualize(self) -> None:
        self.display(self.pcd, self.intense)

class SemanticSegmentation(SensorCameraSemanticSegmentationStub, SensorSpawn):
    
    def __init__(self, world: carla.World):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)

        palette_autopath = Path(__file__).resolve().parent / "../config/palette.json"
        with palette_autopath.open("r")  as f:
            palette = json.load(f)
        self.palette = {int(k): tuple(v) for k, v in palette.items()}
        
        self.sensor_bp = world.get_blueprint_library().find(self.name)
        self.callback = queue.Queue()
        self.num_label = len(list(CarlaLabel))
        self._lut_data = self._build_lut_rgba()
        
    def extract_data(self, layers: Optional[list[CarlaLabel]] = None, alpha = 1):
        if not hasattr(self, "actor"):
            raise ReferenceError(f"Actor {self.__class__.__name__} has not been spawned")
        if layers is not None and not isinstance(layers, list):
            raise TypeError("layers arg must be a list of name of layers")

        data = self.callback.get()
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

        return overlay, labels  # RGBA uint8
    
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
    def __init__(self, world: carla.World):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)

        self.sensor_bp = world.get_blueprint_library().find(self.name)
        self.callback = queue.Queue()
    
    def extract_data(self):
        if not hasattr(self, "actor"):
            raise ReferenceError(f"Actor {self.__class__.__name__} has not been spawned")
        
        data = self.callback.get()
        frame = self.__decode_frame__(data)
        return frame
    
    @staticmethod
    def __decode_frame__(carla_image: carla.Image) -> np.ndarray:
        return np.frombuffer(carla_image.raw_data, dtype = np.uint8).reshape(carla_image.height, carla_image.width, 4)

        
class GNSS(SensorOtherGnssStub, SensorSpawn):
    
    @dataclass
    class Geodetic:
        lat: float
        lon: float
        alt: float

    @dataclass
    class ECEF:
        x: float
        y: float
        z: float

    @dataclass
    class ENU:
        east: float
        north: float
        up: float
        
    class CustomCallback:
        def __init__(self):
            self._latest = None
            self._lock = threading.Lock()
            self._have_sample = threading.Event()
            
        def put(self, data: carla.GnssMeasurement):
            with self._lock:
                self._latest = data
                self._have_sample.set()
            
        def get(self, wait: bool = True, timeout: float = 0.5):
            if wait and not self._have_sample.wait(timeout):
                return None
            with self._lock:
                data = self._latest
            if data is None:
                return None
            
            return {
                "lat": float(data.latitude),
                "lon": float(data.longitude),
                "alt": float(data.altitude),
            }
            
    class GNSSData(Mapping):
        def __init__(self,
                     geodetic: Optional['GNSS.Geodetic']=None,
                     ecef: Optional['GNSS.ECEF']=None,
                     enu: Optional['GNSS.ENU']=None):
            self.Geodetic = geodetic
            self.ECEF = ecef
            self.ENU = enu
            self._data = {"Geodetic": geodetic, "ECEF": ecef, "ENU": enu}
            
        def __getitem__(self, k): return self._data[k]
        def __iter__(self): return iter(self._data)
        def __len__(self): return len(self._data)

        # --- Pretty print ---
        def __repr__(self):
            if self.Geodetic is None:
                return "<GNSSData: no data>"
            parts = [
                f"Geodetic(lat={self.Geodetic.lat:.6f}, "
                f"lon={self.Geodetic.lon:.6f}, "
                f"alt={self.Geodetic.alt:.2f})"
            ]
            if self.ECEF is not None:
                parts.append(
                    f"ECEF(x={self.ECEF.x:.2f}, y={self.ECEF.y:.2f}, z={self.ECEF.z:.2f})"
                )
            if self.ENU is not None:
                parts.append(
                    f"ENU(east={self.ENU.east:.2f}, north={self.ENU.north:.2f}, up={self.ENU.up:.2f})"
                )
            return " | ".join(parts)
    
    def __init__(self, world: carla.World, origin: tuple = (42.0, 2.0, 2.036)):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)
        
        self.sensor_bp = world.get_blueprint_library().find(self.name)
        self.callback = self.CustomCallback()
        
        
        self.a = 6378137
        self.b = 6356752.3142
        self.f = (self.a - self.b) / self.a
        self.e_sq = self.f * (2 - self.f)
        self.origin = origin

        self._geodetic: Optional[GNSS.Geodetic] = None
        self._ecef: Optional[GNSS.ECEF] = None
        self._enu: Optional[GNSS.ENU] = None

    
    def extract_data(self, return_ecf = False, return_enu = False) -> GNSSData:
        gdict = self.callback.get()
        if gdict is None:
            self._geodetic = self._ecef = self._enu = None
            return GNSS.GNSSData()

        # NOTE: refer to the inner classes via the class, not self.*
        self._geodetic = GNSS.Geodetic(**gdict)
        self._ecef = GNSS.ECEF(**self.geodetic_to_ecef(**gdict)) if return_ecf else None
        self._enu  = GNSS.ENU(**self.geodetic_to_enu(**gdict))   if return_enu  else None

        return GNSS.GNSSData(self._geodetic, self._ecef, self._enu)
    
    
    @property
    def geodetic(self): return self._geodetic

    @property
    def ecef(self): return self._ecef

    @property
    def enu(self): return self._enu


    def geodetic_to_ecef(self, lat, lon, alt):
        lamb = np.radians(lat)
        phi = np.radians(lon)
        s = np.sin(lamb)
        N = self.a / np.sqrt(1 - self.e_sq * s * s)

        sin_lambda = np.sin(lamb)
        cos_lambda = np.cos(lamb)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        x = (alt + N) * cos_lambda * cos_phi
        y = (alt + N) * cos_lambda * sin_phi
        z = (alt + (1 - self.e_sq) * N) * sin_lambda

        return {
            'x': float(x),
            'y': float(y),
            'z': float(z)
        }
    
    def ecef_to_enu(self, x, y, z):
        lamb = np.radians(self.origin[0])
        phi = np.radians(self.origin[1])
        s = np.sin(lamb)
        N = self.a / np.sqrt(1 - self.e_sq * s * s)

        sin_lambda = np.sin(lamb)
        cos_lambda = np.cos(lamb)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        x0 = (self.origin[2] + N) * cos_lambda * cos_phi
        y0 = (self.origin[2] + N) * cos_lambda * sin_phi
        z0 = (self.origin[2] + (1 - self.e_sq) * N) * sin_lambda

        xd = x - x0
        yd = y - y0
        zd = z - z0

        xEast = -sin_phi * xd + cos_phi * yd
        yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
        zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

        return {
            'east': float(xEast),
            'north': float(yNorth),
            'up': float(zUp)
        }

    def geodetic_to_enu(self, lat, lon, alt):
        ecf = self.geodetic_to_ecef(lat, lon, alt)
        
        return self.ecef_to_enu(**ecf)