import os, sys
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import ast
import carla
import numpy as np
import configparser
from utils.messages.logger import Logger
from typing import Literal

config = configparser.ConfigParser()
config.read(parent + "/../config/config.ini")
excluded_junctions = ast.literal_eval(config.get("TrafficManager", "excluded_junctions"))

class World:
    def __init__(self, client: carla.Client, tm_port: int, delta = 0.05):
        self.client = client
        self.world = client.get_world()
        self.tm = client.get_trafficmanager(tm_port)
        self.tm_port = tm_port
        self.map = self.world.get_map()

        self.sync = False; self.delta = delta
        self.timeout = 1.0; self.disable_render = False
        self.settings: carla.WorldSettings = self.world.get_settings()

        self.log = Logger()   # <-- attach logger to this class
        
    def switch_map(self, name: str):
        self.client.load_world(name)
        
    def apply_settings(self):
        self.client.set_timeout(self.timeout)
        self.settings.synchronous_mode = self.sync
        self.settings.fixed_delta_seconds = self.delta if self.sync != 0 else None
        self.settings.no_rendering_mode = self.disable_render
        self.world.apply_settings(self.settings)
        self.tm.set_synchronous_mode(self.sync)
        
        self.log.DEBUG(
            f"Applied settings:\n"
            f"    synchronous_mode={self.settings.synchronous_mode}\n"
            f"    fixed_delta_seconds={self.settings.fixed_delta_seconds}\n"
            f"    no_rendering_mode={self.settings.no_rendering_mode}\n"
            f"    timeout={self.timeout}\n"
            f"    tm_port={self.tm.get_port()}"
        )

    def factory_reset(self):
        self.log.DEBUG("Reseting world to factory")
        self.sync = False
        self.settings.synchronous_mode = self.sync
        self.settings.fixed_delta_seconds = self.delta if self.sync else None
        try:
            self.tm.set_synchronous_mode(self.sync)
            self.world.apply_settings(self.settings)
        except Exception as e:
            self.log.ERROR(f"Failed to reset world -> {e}", e)
        self.log.CUSTOM("SUCCESS", "World reset to [bold]factory default[/]")

    def draw_waypoints(self, waypoints, duration: float = 1, color: tuple = (0, 255, 0), size = 0.1):
        for point in waypoints:
            point_loc = carla.Location(x = float(point[0]), y = float(point[1]), z = float(point[2]))
            self.world.debug.draw_point(point_loc, size = size, color = carla.Color(*color), life_time = duration)
    
    def draw_single_waypoint(self, waypoint, duration: float = 1, color: tuple = (0, 255, 0), size = .18):
        point_loc = carla.Location(x = float(waypoint[0]), y = float(waypoint[1]), z = float(waypoint[2]))
        self.world.debug.draw_point(point_loc, size = size, color = carla.Color(*color), life_time = duration)

    def get_waypoint_junction(self, location: np.ndarray):
        wp = self.map.get_waypoint(carla.Location(*location))
        if wp.is_junction:
            junction = wp.get_junction()
            if junction.id not in excluded_junctions:  # Not a 2 way junction
                return True, junction
            return False, None
        return False, None

    def get_segments_from_points(self, seg_type: Literal["junction", "road"], locations: np.ndarray):
        """
        Returns a dictionary of junction_id -> list of waypoints (locations) inside that junction.
        
        Parameters
        ----------
        locations : np.ndarray, shape (N,3)
            List of points to check.
        excluded_junctions : set, optional
            Junction IDs to ignore.

        Returns
        -------
        junction_dict : dict
            Keys are junction IDs, values are lists of np.ndarray locations inside that junction.
        """
        global excluded_junctions
        if excluded_junctions is None:
            excluded_junctions = set()

        if seg_type == "junction":
            junctions = []
            
            for loc in locations:
                wp = self.map.get_waypoint(carla.Location(*loc))
                if wp.is_junction:
                    junction = wp.get_junction()
                    # if junction.id in excluded_junctions:
                    #     continue

                    junctions.append(junction)
            
            return junctions
        
        elif seg_type == "road":
            road_wps = []
            for loc in locations:
                wp = self.map.get_waypoint(carla.Location(*loc), project_to_road=True)
                if not wp.is_junction:
                    road_wps.append(wp)
            return road_wps
        else:
            raise ValueError(f"Invalid seg_type '{seg_type}', must be 'junction' or 'road'.")