import os, sys
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import ast
import carla
import numpy as np
from rich import print
import configparser

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
        
    def switch_map(self, name: str):
        self.client.load_world(name)
        
    def apply_settings(self):
        self.client.set_timeout(self.timeout)
        self.settings.synchronous_mode = self.sync
        self.settings.fixed_delta_seconds = self.delta if self.sync != 0 else None
        self.settings.no_rendering_mode = self.disable_render
        self.world.apply_settings(self.settings)
        self.tm.set_synchronous_mode(self.sync)

    def factory_reset(self):
        print(f"[[yellow]WARNING[/] [purple]({self.__class__.__name__})[/]]: Reseting world to factory")
        self.sync = False
        self.settings.synchronous_mode = self.sync
        self.settings.fixed_delta_seconds = self.delta if self.sync else None
        try:
            self.tm.set_synchronous_mode(self.sync)
            self.world.apply_settings(self.settings)
        except Exception as e:
            print(f"[[red]ERROR[/] [purple]({self.__class__.__name__})[/]]: Failed to reset world -> {e}")
        print(f"[[green]INFO[/] [purple]({self.__class__.__name__})[/]]: World reset to async")

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
            if junction.id not in excluded_junctions: # Not a 2 way junction
                return True, junction
            return False, None
        return False, None