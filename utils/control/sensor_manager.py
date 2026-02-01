"""
Sensor Manager for CARLA - Handles extraction and organization of sensor data.
"""

from typing import Dict, Optional, Any, List
from utils.messages.logger import Logger


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
        self.sensors_by_type: Dict[str, Dict[str, Any]] = {}  # Organized by type
        self.active_camera: Optional[str] = None
        self.camera_keys: List[str] = []

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
            self.logger.DEBUG(f"Failed to extract data from '{sensor_name}': {e}")
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
