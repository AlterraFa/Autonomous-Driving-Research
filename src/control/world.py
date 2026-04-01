import os, sys
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import carla
import numpy as np
from src.messages.logger import Logger
from typing import Literal
from functools import lru_cache
from config import CONFIG

excluded_junctions = CONFIG.traffic_manager.excluded_junctions

class World:
    def __init__(self, client: carla.Client, tm_port: int, delta=CONFIG.world.fixed_delta_seconds):
        self.client = client
        self.world = client.get_world()
        self.tm = client.get_trafficmanager(tm_port)
        self.tm_port = tm_port
        self.map = self.world.get_map()

        self.sync = False; self.delta = delta
        self.disable_render = False
        self.settings: carla.WorldSettings = self.world.get_settings()

        self.log = Logger()   # <-- attach logger to this class
        
        # Cache for waypoint junction lookups to avoid repeated CARLA queries
        self._junction_cache = {}
        self._cache_max_size = CONFIG.world.junction_cache_max_size
        
    def refresh(self):
        """Update all internal CARLA handles after a map change."""
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.settings = self.world.get_settings()
        
        self._junction_cache.clear()
        self.log.INFO(f"World Internal handles refreshed for: {self.map.name}")
        
    def switch_map(self, name: str):
        self.client.load_world(name)
        
    def apply_settings(self):
        self.settings.synchronous_mode = self.sync
        self.settings.fixed_delta_seconds = self.delta if self.sync != 0 else None
        self.settings.no_rendering_mode = self.disable_render
        self.world.apply_settings(self.settings)
        self.tm.set_synchronous_mode(self.sync)
        
        self.log.INFO(
            f"Applied settings:\n"
            f"    synchronous_mode={self.settings.synchronous_mode}\n"
            f"    fixed_delta_seconds={self.settings.fixed_delta_seconds}\n"
            f"    no_rendering_mode={self.settings.no_rendering_mode}\n"
            f"    tm_port={self.tm.get_port()}"
        )

    def factory_reset(self):
        self.log.WARNING("Reseting world to factory")
        self.sync = False
        self.settings.synchronous_mode = self.sync
        self.settings.fixed_delta_seconds = self.delta if self.sync else None
        self.settings.no_rendering_mode = self.disable_render
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
        # Use tuple of location as cache key (rounded to avoid floating point precision issues)
        cache_key = tuple(np.round(location, decimals=CONFIG.world.waypoint_rounding_decimals))
        
        if cache_key in self._junction_cache:
            return self._junction_cache[cache_key]
        
        # Clear cache if it gets too large to prevent unbounded growth
        if len(self._junction_cache) > self._cache_max_size:
            self._junction_cache.clear()
        
        wp = self.map.get_waypoint(carla.Location(*location))
        if wp.is_junction:
            junction = wp.get_junction()
            if junction.id not in excluded_junctions:  # Not a 2 way junction
                result = (True, junction)
            else:
                result = (False, None)
        else:
            result = (False, None)
        
        self._junction_cache[cache_key] = result
        return result

    def get_multi_junctions(self, waypoints: np.ndarray):
        junctions_metadata = []; cached_id = []
        for wp in waypoints:
            # Use tuple of location as cache key (rounded to avoid floating point precision issues)
            cache_key = tuple(np.round(wp, decimals=CONFIG.world.waypoint_rounding_decimals))
            
            if cache_key in self._junction_cache:
                is_junction, junction = self._junction_cache[cache_key]
                if is_junction and junction.id not in cached_id:
                    junctions_metadata.append(junction)
                    cached_id.append(junction.id)
            else:
                # Clear cache if it gets too large to prevent unbounded growth
                if len(self._junction_cache) > self._cache_max_size:
                    self._junction_cache.clear()
                
                loc = carla.Location(*wp)
                carla_wp = self.map.get_waypoint(loc)
                if carla_wp.is_junction:
                    junction = carla_wp.get_junction()
                    if junction.id not in excluded_junctions and junction.id not in cached_id:
                        junctions_metadata.append(junction)
                        cached_id.append(junction.id)
                        self._junction_cache[cache_key] = (True, junction)
                    else:
                        self._junction_cache[cache_key] = (False, None)
                else:
                    self._junction_cache[cache_key] = (False, None)
        
        return junctions_metadata

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
                x, y, z = loc
                wp = self.map.get_waypoint(carla.Location(float(x), float(y), float(z)))
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