import carla
import numpy as np
import json
import cv2
import time

from utils.others.lidar_visualization import LIDARVisualizer
from utils.spawn.callback import Extractor
from utils.messages.logger import Logger

from config.enum import CarlaLabel
from typing import Optional
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass

np.printoptions(5)
    
class SensorSpawn(object):
    
    __slot__ = ['log', 'name', 'sensor_bp', 'world', 'literal_name', 'actor']
    log = Logger()

    def __init__(self, name, world: carla.World):
        self.log          = Logger()
        self.name         = name
        self.sensor_bp    = world.get_blueprint_library().find(self.name)
        self.world        = world
        self.literal_name = self.__literal_name__(self.name)
        self.actor        = None
        
    def spawn(self, attach_to = None, **kwargs):
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
        self.log.INFO(f"Sensor [blue][bold]{self.literal_name}[/bold][/] spawned successfully. Listening to it")
    
    def change_view(self, **kwargs):
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
        
        self.actor.set_transform(transform)

    def destroy(self):
        if hasattr(self, "actor"):
            self.actor.stop()
            self.actor.destroy()
            self.log.INFO(f"Sensor [blue][bold]{self.literal_name}[/bold][/] stopped and destroyed")
    
    @staticmethod
    def __literal_name__(name: str):
        string = name.split(".")[1:][::-1]
        sensor_type = string[0]
        temp = []
        for word in sensor_type.split('_'):
            temp += [word.capitalize()]
        sensor_name = ' '.join(temp + [string[1].capitalize()])
        return sensor_name

    @classmethod
    def test_sensor(cls, game_viewer, sensors_metadata, max_retries = 3):
        for attempt in range(max_retries):
            try:
                game_viewer.init_sensor(sensors_metadata)
                cls.log.CUSTOM("SUCCESS", "All sensor initialized successfully")
                return True
            except RuntimeError as e:
                if "parent actor not found" in str(e):
                    if attempt < max_retries - 1:
                        cls.log.WARNING(f"Sensor init failed (attempt {attempt + 1}/{max_retries}): {e}")
                        wait_ticks = 20 + (attempt * 10)
                        cls.log.WARNING(f"Retrying after {wait_ticks} ticks...")
                        for _ in range(wait_ticks):
                            if game_viewer.virt_world.world.get_settings().synchronous_mode:
                                game_viewer.virt_world.world.tick()
                            else:
                                time.sleep(0.1)
                    else:
                        cls.log.ERROR(f"Sensor init failed after {max_retries} attempts: {e}")
                        return False
                else:
                    cls.log.ERROR(f"Sensor init error: {e}")
                    return False
        return False
    
        
from utils.stubs.sensor__lidar__ray_cast_stub import SensorLidarRayCastStub
class LidarRaycast(SensorLidarRayCastStub, SensorSpawn, LIDARVisualizer):
    def __init__(self, world, vis_range = 50, vis_window = (800, 600)):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)
        LIDARVisualizer.__init__(self, vis_range, vis_window)
        
        self.sensor_bp = world.get_blueprint_library().find(self.name)
        self.callback = Extractor()
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
    
    @property
    def visualize(self) -> None:
        self.display(self.pcd, self.intense)

from utils.stubs.sensor__camera__semantic_segmentation_stub import SensorCameraSemanticSegmentationStub
class SemanticSegmentation(SensorCameraSemanticSegmentationStub, SensorSpawn):
    
    class SemanticData(np.ndarray):
        __array_priority__ = 1000

        def __new__(cls, arr, **meta):
            obj = np.asarray(arr, dtype=np.uint32).view(cls)
            obj.meta = meta
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.meta = getattr(obj, "meta", {})

        def to_image(self, layers: list[CarlaLabel] = None) -> np.ndarray:
            if layers is not None and not isinstance(layers, list):
                raise TypeError("layers arg must be a list of name of layers")
            
            _lut_data = self.meta['lut']
            num_label = self.meta['num_label']
            
            if layers is None:
                lut = _lut_data
            else:
                sel = np.zeros(num_label, dtype=bool)
                for lbl in layers:
                    sel[int(lbl.value)] = True
                lut = _lut_data.copy()
                lut[~sel] = 0

            overlay = lut[self]

            return overlay


    
    def __init__(self, world: carla.World, filter_labels = None, binarize = False, convert_to: SemanticData = None):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)

        palette_autopath = Path(__file__).resolve().parent / "../../config/palette.json"
        with palette_autopath.open("r")  as f:
            palette = json.load(f)
        self.palette = {int(k): tuple(v) for k, v in palette.items()}
        
        self.callback = Extractor()
        self.num_label = len(list(CarlaLabel))
        self._lut_data = self._build_lut_rgba()
        
        self.filter_labels = filter_labels
        self.binarize = binarize
        self.convert_to = convert_to
        
    def extract_data(self):
        if not hasattr(self, "actor"):
            raise ReferenceError(f"Actor {self.__class__.__name__} has not been spawned")

        data = self.callback.get()
        labels = self.__decode_semantic_labels__(data)

        if self.filter_labels is not None:
            if not isinstance(self.filter_labels, (list, tuple, np.ndarray, set)):
                 raise TypeError("filter_labels must be an iterable of allowed label IDs.")
            
            allowed_labels = set(self.filter_labels)
            keep_mask = np.isin(labels, list(allowed_labels))
            
            filtered_labels = labels.copy() 
            filtered_labels[~keep_mask] = 0
            
            labels = filtered_labels
            
        if self.binarize:
            labels = self._binarize(labels)
            return labels
        
        sem_data = SemanticSegmentation.SemanticData(labels, lut = self._lut_data, num_label = self.num_label)
        if self.convert_to:
            return self.convert_to(sem_data)
        else: return sem_data
    
    def _build_lut_rgba(self) -> np.ndarray:
        lut = np.zeros((self.num_label, 4), dtype=np.uint8)
        for k, (r, g, b) in self.palette.items():
            lut[k] = (r, g, b, 255)  # RGBA
        return lut
    
    def _binarize(self, labels: np.ndarray) -> np.ndarray:
        if labels.ndim > 2:
            raise ValueError("Binarization should be performed on the single-channel ID array, not the color image.")
            
        return np.where(labels != 0, 255, 0).astype(np.uint8)
            

    def __decode_semantic_labels__(self, carla_image: carla.Image) -> np.ndarray:
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8).reshape(carla_image.height, carla_image.width, 4)
        labels = array[:, :, 2].copy()  # BGRA -> take R channel
        return labels


from utils.stubs.sensor__camera__rgb_stub import SensorCameraRgbStub
class RGB(SensorCameraRgbStub, SensorSpawn):
    def __init__(self, world: carla.World):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)

        self.callback = Extractor()
    
    def extract_data(self):
        if not hasattr(self, "actor"):
            raise ReferenceError(f"Actor {self.__class__.__name__} has not been spawned")
        
        data = self.callback.get()
        frame = self.__decode_frame__(data)
        return frame
    
    @staticmethod
    def __decode_frame__(carla_image: carla.Image) -> np.ndarray:
        return np.frombuffer(carla_image.raw_data, dtype = np.uint8).reshape(carla_image.height, carla_image.width, 4)

        
from utils.stubs.sensor__other__gnss_stub import SensorOtherGnssStub
class GNSS(SensorOtherGnssStub, SensorSpawn):
    """GNSS IS VERY INACURATE COMPARE TO vehicle.get_location()"""
    @dataclass
    class Geodetic:
        lat: float
        lon: float
        alt: float
        def to_numpy(self): return np.array([self.lat, self.lon, self.alt])

    @dataclass
    class ECEF:
        x: float
        y: float
        z: float
        def to_numpy(self): return np.array([self.x, self.y, self.z])

    @dataclass
    class ENU:
        east: float
        north: float
        up: float
        def to_numpy(self): return np.array([self.east, self.north, self.up])
            
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
                f"Geodetic(lat={self.Geodetic.lat: .6f}, "
                f"lon={self.Geodetic.lon: .6f}, "
                f"alt={self.Geodetic.alt: .2f})"
            ]
            if self.ECEF is not None:
                parts.append(
                    f"ECEF(x={self.ECEF.x: .2f}, y={self.ECEF.y: .2f}, z={self.ECEF.z: .2f})"
                )
            if self.ENU is not None:
                parts.append(
                    f"ENU(east={self.ENU.east: .2f}, north={self.ENU.north: .2f}, up={self.ENU.up: .2f})"
                )
            return " | ".join(parts)
    
    def __init__(self, world: carla.World, freq_hz=10, mu_ms=120, sigma_ms=10):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world, )
        from collections import deque

        self.callback = Extractor({
            "lat": "latitude",
            "lon": "longitude",
            "alt": "altitude"
        })

        # WGS84 Constants (using more precise inverse flattening)
        self.a = 6378137.0
        self.f = 1 / 298.257223563 
        self.e_sq = self.f * (2 - self.f)

        # Initialize origin with 0s, then calibrate
        self.origin = (0.0, 0.0, 0.0)
        self.alt_offset = 0.0
        self._calibrate_origin(world)

        self._geodetic: Optional[GNSS.Geodetic] = None
        self._ecef: Optional[GNSS.ECEF] = None
        self._enu: Optional[GNSS.ENU] = None
        
        self.world = world
        self.sample_interval = 1.0 / freq_hz
        self.next_sample_time = 0.0

        # -- Latency Drift (Ornstein-Uhlenbeck)
        self.mu = mu_ms / 1000.0
        self.sigma = sigma_ms / 1000.0
        self.theta = 0.05
        self.current_latency = self.mu

        self.delivery_queue = []
        self.last_delivered_data = None
        
    def _calibrate_origin(self, world: carla.World):
        """
        Queries the CARLA map metadata to find the exact, noiseless 
        Geodetic coordinates for the world origin (0,0,0).
        """
        carla_map = world.get_map()
        
        # transform_to_geolocation converts CARLA coordinates to Lat/Lon/Alt
        # We query (0,0,0) to find the perfect reference point.
        origin_geo = carla_map.transform_to_geolocation(carla.Location(x=0, y=0, z=0))

        # self.origin[0]=Lat, [1]=Lon, [2]=Alt
        self.origin = (origin_geo.latitude, origin_geo.longitude, origin_geo.altitude)
        
        # This ensures that when CARLA Z = 0, our ENU 'Up' also = 0.
        self.alt_offset = origin_geo.altitude

    def geodetic_to_ecef(self, lat, lon, alt):
        # Subtract the map's origin altitude so that CARLA Z=0 is our baseline
        alt_corrected = alt - self.alt_offset

        lamb = np.radians(lat)
        phi = np.radians(lon)
        s = np.sin(lamb)
        N = self.a / np.sqrt(1 - self.e_sq * s * s)

        cos_lambda = np.cos(lamb)
        sin_lambda = np.sin(lamb)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x = (N + alt_corrected) * cos_lambda * cos_phi
        y = (N + alt_corrected) * cos_lambda * sin_phi
        z = (N * (1 - self.e_sq) + alt_corrected) * sin_lambda

        return {"x": x, "y": y, "z": z}
    
    def ecef_to_enu(self, x, y, z):
        # Coordinates of the origin in ECEF (using alt=0 because of alt_corrected)
        lamb = np.radians(self.origin[0])
        phi = np.radians(self.origin[1])
        s = np.sin(lamb)
        N = self.a / np.sqrt(1 - self.e_sq * s * s)

        cos_lambda = np.cos(lamb)
        sin_lambda = np.sin(lamb)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Reference ECEF at origin (0,0,0)
        # Note: We use 0 for altitude here because geodetic_to_ecef already 
        # normalized the incoming altitude relative to this origin's altitude.
        x0 = N * cos_lambda * cos_phi
        y0 = N * cos_lambda * sin_phi
        z0 = N * (1 - self.e_sq) * sin_lambda

        xd, yd, zd = x - x0, y - y0, z - z0

        # Standard ECEF to ENU Rotation Matrix
        xEast = -sin_phi * xd + cos_phi * yd
        yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
        zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

        return {
            'east': float(xEast),
            'north': float(-yNorth), # Correcting for CARLA's Left-Handed Y-axis
            'up': float(zUp)
        }

    def geodetic_to_enu(self, lat, lon, alt):
        ecf = self.geodetic_to_ecef(lat, lon, alt)
        
        return self.ecef_to_enu(**ecf)
    
    def _update_latency_drift(self, dt):
        """Changes the latency value slowly over time, not instantly."""
        noise = self.sigma * np.sqrt(dt) * np.random.normal()
        drift = self.theta * (self.mu - self.current_latency) * dt
        self.current_latency += drift + noise
        self.current_latency = np.clip(self.current_latency, 0.05, 0.25)
    
    
    def extract_data(self) -> GNSSData:
        snapshot = self.world.get_snapshot()
        t = snapshot.timestamp.elapsed_seconds
        dt = snapshot.timestamp.delta_seconds
        
        self._update_latency_drift(dt)

        if t >= self.next_sample_time:
            gdict = self.callback.get()
            if gdict is  not None:

                # NOTE: refer to the inner classes via the class, not self.*
                self._geodetic = GNSS.Geodetic(**gdict)
                self._ecef = GNSS.ECEF(**self.geodetic_to_ecef(**gdict))
                self._enu  = GNSS.ENU(**self.geodetic_to_enu(**gdict))
                
                raw_data = GNSS.GNSSData(self._geodetic, self._ecef, self._enu)

                delivery_time = t + self.current_latency

                if self.delivery_queue and delivery_time < self.delivery_queue[-1][0]:
                    delivery_time = self.delivery_queue[-1][0] + 0.001
                self.delivery_queue.append((delivery_time, raw_data))
                self.next_sample_time = t + self.sample_interval
                
        ready_packets = [data for arrival_t, data in self.delivery_queue if arrival_t <= t]
        self.delivery_queue = [(arrival_t, data) for arrival_t, data in self.delivery_queue if arrival_t > t]    
        
        if ready_packets:
            self.last_delivered_data = ready_packets[-1]
            
            self._geodetic = self.last_delivered_data.Geodetic
            self._ecef = self.last_delivered_data.ECEF
            self._enu = self.last_delivered_data.ENU

        if self.last_delivered_data is not None:
            return self.last_delivered_data
        
        gdict = self.callback.get()
        geodetic = GNSS.Geodetic(**gdict)
        ecef = GNSS.ECEF(**self.geodetic_to_ecef(**gdict))
        enu  = GNSS.ENU(**self.geodetic_to_enu(**gdict))
        return GNSS.GNSSData(geodetic, ecef, enu)
         
    
    
    @property
    def geodetic(self): return self._geodetic

    @property
    def ecef(self): return self._ecef

    @property
    def enu(self): return self._enu

    
from utils.stubs.sensor__other__imu_stub import SensorOtherImuStub
class IMU(SensorOtherImuStub, SensorSpawn):
    
    class IMUData(Mapping):
        def __init__(self, 
                    accel: np.ndarray = None,
                    gyro: np.ndarray = None,
                    comp: float = None):
           self.Acceleration = accel
           self.Gyroscope    = gyro
           self.Compass      = comp
           
           self._data = {"Acceleration": self.Acceleration, "Gyroscope": self.Gyroscope, "Compass": self.Compass}
           
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return 3
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __repr__(self):
            if self.Acceleration is None:
                return "<GNSSData: no data>"
            parts = [
                f"Acceleration(x={self.Acceleration[0]: .4f}, "
                f"y={self.Acceleration[1]: .4f}, "
                f"z={self.Acceleration[2]: .4f})"
            ]
            if self.Gyroscope is not None:
                parts.append(
                    f"Gyroscope(x={self.Gyroscope[0]: .4f}, y={self.Gyroscope[1]: .4f}, z={self.Gyroscope[2]: .4f})"
                )
            if self.Compass is not None:
                parts.append(
                    f"Compass={np.degrees(self.Compass): .4f}"
                )
            return " | ".join(parts)

    def __init__(self, world: carla.World):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)
        
        self.callback = Extractor({
            "accel": "accelerometer", 
            "gyro": "gyroscope", 
            "comp": "compass"
        }) 
        
    def extract_data(self):
        data = self.callback.get()
        if data['accel'] is not None:
            # convert carla.Vector3D to np.ndarray
            data["accel"] = np.array([data['accel'].x, data['accel'].y, data['accel'].z])
            data["gyro"] = np.array([data['gyro'].x, data['gyro'].y, data['gyro'].z])
            return IMU.IMUData(accel = data['accel'], gyro = data['gyro'], comp = data['comp']) 
        
        return IMU.IMUData(*[None] * 3)
    
from utils.stubs.sensor__camera__depth_stub import SensorCameraDepthStub
class Depth(SensorCameraDepthStub, SensorSpawn):
    class DepthMap(np.ndarray):
        __array_priority__ = 1000

        def __new__(cls, arr, **meta):
            obj = np.asarray(arr, dtype=np.float32).view(cls)
            obj.meta = meta
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.meta = getattr(obj, "meta", {})

        def to_windowed(self, max_depth: float = 80.0, invert: bool = True) -> np.ndarray:
            """
            Linear windowed depth → grayscale.
            Args:
                depth_meter: depth in meters.
                max_depth: distances ≥ max_depth map to the darkest end.
                invert: if True, nearer → brighter.
            Returns:
                uint8 grayscale image [0,255].
            """
            d = np.clip(self / max_depth, 0.0, 1.0)
            if invert:
                d = 1.0 - d
            return (d * 255.0).astype(np.uint8)

        def to_log(self, max_depth: float = 80.0, log_scale: float = 100.0,
                    invert: bool = True, epsilon: float = 1e-6) -> np.ndarray:
            """Convert depth array to gray scaled depth image using log scale

            Args:
                depth_meter (np.ndarray): depth data in meters
                epsilon (float, optional): prevent division by zero. Defaults to 1e-6.
                scale (float, optional): scale intensity of pixels. Defaults to 100.0.
                invert (bool, optional): invert to brighter when nearer. Defaults to False.

            Returns:
                np.ndarray (np.uint8): depth image 
            """
            d = np.clip(self, 0.0, max_depth)
            x = np.log1p(d / (log_scale + epsilon))
            x /= np.log1p(max_depth / (log_scale + epsilon)) + epsilon
            x = np.clip(x, 0.0, 1.0)
            if invert:
                x = 1.0 - x
            return (x * 255.0).astype(np.uint8)

        def to_disparity(self, min_depth: float = 1.0, max_depth: float = 80.0,
                        epsilon: float = 1e-6) -> np.ndarray:
            """
            Convert depth in meters to a disparity-like grayscale image for visualization.

            Disparity emphasizes nearer objects by mapping inverse depth to intensity.
            Nearer objects appear brighter, farther objects darker.

            Args:
                depth_meter (np.ndarray): Depth map in meters (from CARLA decoded depth).
                min_depth (float, optional): Minimum depth to consider (in meters). 
                                            Depths smaller than this are clamped. Defaults to 1.0.
                max_depth (float, optional): Maximum depth to consider (in meters). 
                                            Depths larger than this are clamped. Defaults to 80.0.
                epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

            Returns:
                np.ndarray: Grayscale disparity image (uint8), shape = depth_meter.shape,
                            values in range [0, 255], where closer = brighter.
            """
            d = np.clip(self, min_depth, max_depth)
            disp = (1.0 / (d + epsilon) - 1.0 / max_depth) / \
                (1.0 / (min_depth + epsilon) - 1.0 / max_depth)
            disp = np.clip(disp, 0.0, 1.0)
            return (disp * 255.0).astype(np.uint8)

    def __init__(self, world: carla.World, convert_to: DepthMap = None):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)
        
        self.callback = Extractor()
        self.convert = convert_to
        
    def extract_data(self):
        data = self.callback.get()
        image = np.frombuffer(data.raw_data, dtype = np.uint8).reshape(data.height, data.width, -1)

        depth_meter = self.__to_meter__(image)
        depth_map = Depth.DepthMap(depth_meter)
        return depth_map if self.convert is None else self.convert(depth_map)
        

    @staticmethod
    def __to_meter__(depth_map: np.ndarray) -> np.ndarray:
        r = depth_map[:, :, 2].astype(np.float32)
        g = depth_map[:, :, 1].astype(np.float32)
        b = depth_map[:, :, 0].astype(np.float32)
        
        depth_norm = (r + g * 256.0 + b * 65536.0) / (256 ** 3 - 1)
        return depth_norm * 1000.0
    
from utils.stubs.sensor__camera__instance_segmentation_stub import SensorCameraInstanceSegmentationStub
class InstanceSegmentation(SensorCameraInstanceSegmentationStub, SensorSpawn):
    def __init__(self, world: carla.World, sat = 0.65, convert_to = None):
        super().__init__()
        SensorSpawn.__init__(self, self.name, world)
        
        self.callback = Extractor()
        
        self.sat = sat
        self.cache = {}
        self.convert_to = convert_to
        
    def extract_data(self):
        data = self.callback.get()
        image = np.frombuffer(data.raw_data, dtype = np.uint8).reshape(data.height, data.width, -1)
        
        b = image[:, :, 0].astype(np.uint32)
        g = image[:, :, 1].astype(np.uint32)
        r = image[:, :, 2].astype(np.uint32)
        ids = r + (g << 8) + (b << 16)          # 24-bit instance id

        if self.convert_to is not None:
            return self.convert_to(ids, sat=self.sat)

        return ids.astype(np.int32) 
    
    @staticmethod
    def instance_edge(instance_id: np.ndarray, color=(255,255,255)) -> np.ndarray:
        edges = (cv2.Canny((instance_id % 256).astype(np.uint8), 0, 1) > 0)  # quick edge proxy
        edgeImg = np.zeros((*instance_id.shape, 3), np.uint8)
        edgeImg[edges] = color
        return edgeImg
    
    @staticmethod
    def colorized(instanceId: np.ndarray, sat: float = 0.65) -> np.ndarray:
        h, w = instanceId.shape
        ids, inv = np.unique(instanceId, return_inverse=True)

        phi = 0.61803398875
        h_ = (ids * phi) % 1.0
        s_ = np.full_like(h_, float(np.clip(sat, 0, 1)))
        v_ = np.ones_like(h_)

        i = np.floor(h_ * 6).astype(int) % 6
        f = h_ * 6 - np.floor(h_ * 6)

        p = v_ * (1 - s_)
        q = v_ * (1 - f * s_)
        t = v_ * (1 - (1 - f) * s_)

        r = np.empty_like(h_); g = np.empty_like(h_); b = np.empty_like(h_)
        m0 = (i == 0); r[m0], g[m0], b[m0] = v_[m0], t[m0], p[m0]
        m1 = (i == 1); r[m1], g[m1], b[m1] = q[m1], v_[m1], p[m1]
        m2 = (i == 2); r[m2], g[m2], b[m2] = p[m2], v_[m2], t[m2]
        m3 = (i == 3); r[m3], g[m3], b[m3] = p[m3], q[m3], v_[m3]
        m4 = (i == 4); r[m4], g[m4], b[m4] = t[m4], p[m4], v_[m4]
        m5 = (i == 5); r[m5], g[m5], b[m5] = v_[m5], p[m5], q[m5]

        bgr = (np.stack([b, g, r], axis=1) * 255.0).astype(np.uint8)
        bgr[ids <= 0] = (0, 0, 0)

        out = bgr[inv].reshape(h, w, 3)
        return out
