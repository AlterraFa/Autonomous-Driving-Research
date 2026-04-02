"""
Sensor Manager for CARLA - Handles extraction and organization of sensor data.
"""

from typing import Dict, Optional, Any, List
import numpy as np
import config.enum as enum_defs
from src.messages.logger import Logger


class SensorManager:
    """
    Manages sensor data extraction from CARLA sensors.
    
    Organizes sensors by type and provides clean APIs for:
    - Extracting data from specific sensors
    - Extracting data from all sensors of a type
    - Batch data extraction with error handling
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the SensorManager.
        
        :param logger: Optional Logger instance for debug output
        """
        self.logger = logger or Logger()
        self.sensors_list: Dict[str, Any] = {}  # All sensors by name
        self.view_list: Dict[str, Any] = {}
        self.sensors_by_type: Dict[str, Dict[str, Any]] = {}  # Organized by type
        self.active_camera: Optional[str] = None
        self.camera_keys: List[str] = []
        self.camera_views: List[str] = []
        self.active_view: Optional[str] = None

    def _refresh_camera_views(self) -> None:
        """Rebuild cached camera view names from the live enum."""
        self.camera_views = list(enum_defs.CameraView.__members__.keys())

    def add_view(self, view_name: str, transform: Dict[str, float], activate: bool = False, overwrite: bool = False) -> bool:
        """
        Register a custom runtime camera view.

        :param view_name: Unique name for this view
        :param transform: Camera transform keys (x, y, z, roll, pitch, yaw)
        :param activate: If True, set this view as active immediately
        :param overwrite: If True, allow replacing an existing custom view
        :return: True if view was added/updated successfully
        """
        if not isinstance(transform, dict):
            self.logger.WARNING("View transform must be a dictionary")
            return False

        required_keys = {"x", "y", "z", "roll", "pitch", "yaw"}
        if not required_keys.issubset(transform.keys()):
            self.logger.WARNING(f"View '{view_name}' missing required keys: {required_keys}")
            return False

        if view_name in self.view_list and not overwrite:
            self.logger.WARNING(f"View '{view_name}' already exists. Use overwrite=True to replace it")
            return False

        self.view_list[view_name] = {k: float(transform[k]) for k in required_keys}
        if activate:
            self.active_view = view_name
        self._refresh_camera_views()
        return True

    def update_view(self, view_name: str, transform: Dict[str, float]) -> bool:
        """
        Update transform attributes for an existing runtime view.

        :param view_name: Existing runtime view name
        :param transform: Camera transform keys (x, y, z, roll, pitch, yaw)
        :return: True if updated successfully
        """
        if view_name not in self.view_list:
            return False

        required_keys = {"x", "y", "z", "roll", "pitch", "yaw"}
        if not isinstance(transform, dict) or not required_keys.issubset(transform.keys()):
            return False

        self.view_list[view_name] = {k: float(transform[k]) for k in required_keys}
        self._refresh_camera_views()
        return True

    def get_view_transform(self, view_name: str) -> Optional[Dict[str, float]]:
        """
        Get transform for a view from enum or runtime custom views.

        :param view_name: View name to resolve
        :return: Transform dict or None
        """
        if view_name in enum_defs.CameraView.__members__:
            return getattr(enum_defs.CameraView, view_name).value
        return self.view_list.get(view_name)

    @staticmethod
    def _pose6_to_view_dict(pose6: np.ndarray) -> Dict[str, float]:
        """Convert [x, y, z, roll, pitch, yaw] into camera transform dict."""
        return {
            "x": float(pose6[0]),
            "y": float(pose6[1]),
            "z": float(pose6[2]),
            "roll": float(pose6[3]),
            "pitch": float(pose6[4]),
            "yaw": float(pose6[5]),
        }

    def register_view_matrix(self, prefix: str, views: Any) -> List[str]:
        """
        Register/update runtime views from an (N, 6) matrix.

        Each row maps to: [x, y, z, roll, pitch, yaw].
        Existing views with the same prefix are overwritten each call.

        :param prefix: Prefix used to name views (e.g., "LOCAL_WP")
        :param views: (N, 6) array-like or single (6,) row
        :return: List of updated view names in order
        """
        if views is None:
            return []

        if hasattr(views, "to_numpy"):
            views = views.to_numpy()

        views_arr = np.asarray(views)
        if views_arr.ndim == 1 and views_arr.shape[0] == 6:
            views_arr = views_arr.reshape(1, 6)

        if views_arr.ndim != 2 or views_arr.shape[1] != 6:
            return []

        updated_names: List[str] = []
        new_members: Dict[str, Dict[str, float]] = {}
        for idx, pose in enumerate(views_arr):
            view_name = f"{prefix}_{idx}"
            transform = self._pose6_to_view_dict(pose)
            if not self.update_view(view_name, transform):
                self.add_view(view_name, transform, overwrite=True)
            updated_names.append(view_name)
            new_members[view_name] = transform

        if new_members:
            enum_defs.extend_view(new_members)

        # Remove stale views if waypoint count shrank.
        stale_names = [
            name for name in self.view_list.keys()
            if name.startswith(f"{prefix}_") and name not in updated_names
        ]
        for name in stale_names:
            self.view_list.pop(name, None)

        # Keep cached names in sync even if user doesn't press view-change controls.
        self._refresh_camera_views()

        return updated_names

    def register_sensor(self, sensor_name: str, sensor_object: Any, sensor_type: str) -> None:
        """
        Register a sensor in the manager.
        
        :param sensor_name: Short name for the sensor (e.g., 'rgb', 'depth')
        :param sensor_object: The sensor object
        :param sensor_type: Type of sensor (e.g., 'camera', 'depth', 'semantic_segmentation')
        """
        self.sensors_list[sensor_name] = sensor_object
        
        if sensor_type not in self.sensors_by_type:
            self.sensors_by_type[sensor_type] = {}
        self.sensors_by_type[sensor_type][sensor_name] = sensor_object
        
        # Track cameras separately for easy access
        if sensor_type == 'camera':
            self.camera_keys.append(sensor_name)
            if self.active_camera is None:
                self.active_camera = sensor_name

    def get_sensor(self, sensor_name: str) -> Optional[Any]:
        """
        Get a sensor object by name.
        
        :param sensor_name: Name of the sensor
        :return: Sensor object or None
        """
        return self.sensors_list.get(sensor_name)

    def get_sensor_data(self, sensor_name: str) -> Optional[Any]:
        """
        Safely extract data from a specific sensor.
        
        :param sensor_name: Name of the sensor
        :return: Sensor data or None if extraction fails
        """
        try:
            if sensor_name in self.sensors_list:
                return self.sensors_list[sensor_name].extract_data()
        except Exception as e:
            self.logger.ERROR(f"Failed to extract data from '{sensor_name}': {e}")
        return None

    def get_sensors_by_type(self, sensor_type: str) -> Dict[str, Any]:
        """
        Get all sensors of a specific type.
        
        :param sensor_type: Type of sensor ('camera', 'depth', 'semantic_segmentation', etc.)
        :return: Dictionary of {sensor_name: sensor_object}
        """
        return self.sensors_by_type.get(sensor_type, {})

    def get_sensor_data_by_type(self, sensor_type: str) -> Dict[str, Any]:
        """
        Extract data from all sensors of a specific type.
        
        :param sensor_type: Type of sensor
        :return: Dictionary of {sensor_name: sensor_data} with only successful extractions
        """
        results = {}
        sensors = self.sensors_by_type.get(sensor_type, {})
        
        for sensor_name, sensor_obj in sensors.items():
            data = self.get_sensor_data(sensor_name)
            if data is not None:
                results[sensor_name] = data
        
        return results

    def get_all_sensor_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract data from all sensors organized by type.
        
        :return: Nested dictionary with structure: {sensor_type: {sensor_name: sensor_data}}
        """
        all_data = {}
        
        for sensor_type in self.sensors_by_type.keys():
            all_data[sensor_type] = self.get_sensor_data_by_type(sensor_type)
        
        return all_data

    def get_active_camera_data(self) -> Optional[Any]:
        """
        Extract data from the currently active camera.
        
        :return: Camera frame or None
        """
        if self.active_camera:
            return self.get_sensor_data(self.active_camera)
        return None

    def set_active_camera(self, camera_name: str) -> bool:
        """
        Set the active camera.
        
        :param camera_name: Name of camera to activate
        :return: True if successful, False otherwise
        """
        if camera_name in self.camera_keys:
            self.active_camera = camera_name
            self.logger.DEBUG(f"Switched to camera - [bold]{self.choosen_sensor.literal_name}[/]")
            
            return True
        self.logger.WARNING(f"Camera '{camera_name}' not found")
        return False

    def switch_camera(self, step: int = 1) -> Optional[str]:
        """
        Cycle to next/previous camera.
        
        :param step: Direction and amount (+1 for next, -1 for previous)
        :return: Name of new active camera or None
        """
        if not self.camera_keys:
            return None
        
        current_idx = self.camera_keys.index(self.active_camera) if self.active_camera else 0
        new_idx = (current_idx + step) % len(self.camera_keys)
        self.active_camera = self.camera_keys[new_idx]
        
        return self.active_camera

    def switch_view(self, step: int = 1) -> Optional[str]:
        """
        Cycle to next/previous camera view from CameraView enum.

        :param step: Direction and amount (+1 for next, -1 for previous, 0 keeps current)
        :return: Name of active view or None
        """
        self._refresh_camera_views()
        if not self.camera_views:
            return None

        if self.active_view not in self.camera_views:
            self.active_view = "FIRST_PERSON" if "FIRST_PERSON" in self.camera_views else self.camera_views[0]

        if step == 0:
            return self.active_view

        current_idx = self.camera_views.index(self.active_view)
        new_idx = (current_idx + step) % len(self.camera_views)
        self.active_view = self.camera_views[new_idx]

        return self.active_view

    def get_available_sensor_types(self) -> List[str]:
        """
        Get list of all available sensor types.
        
        :return: List of sensor types
        """
        return list(self.sensors_by_type.keys())

    def get_available_sensors(self) -> Dict[str, List[str]]:
        """
        Get all available sensors organized by type.
        
        :return: Dictionary with {sensor_type: [sensor_names]}
        """
        return {
            sensor_type: list(sensors.keys())
            for sensor_type, sensors in self.sensors_by_type.items()
        }

    def print_sensor_summary(self) -> None:
        """Print a summary of all registered sensors."""
        print("\n" + "="*50)
        print("SENSOR SUMMARY")
        print("="*50)
        
        for sensor_type, sensors in self.sensors_by_type.items():
            print(f"\n{sensor_type.upper()}:")
            for sensor_name in sensors.keys():
                marker = " [ACTIVE]" if sensor_name == self.active_camera else ""
                print(f"  • {sensor_name}{marker}")
        
        print("\n" + "="*50)
