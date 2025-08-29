import carla
from rich import print

class World:
    def __init__(self, client: carla.Client, tm_port: int, delta = 0.05):
        self.client = client
        self.world = client.get_world()
        self.tm = client.get_trafficmanager(tm_port)
        self.tm_port = tm_port

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
        print("[yellow][WARNING][/]: Reseting world to factory")
        self.sync = False
        self.settings.synchronous_mode = self.sync
        self.settings.fixed_delta_seconds = self.delta if self.sync else None
        try:
            self.tm.set_synchronous_mode(self.sync)
            self.world.apply_settings(self.settings)
        except Exception as e:
            print(f"[red][ERROR][/]: Failed to reset world -> {e}")
        print("[green][INFO]: World reset to async[/]")

    def draw_waypoints(self, waypoints, duration: float = 1):
        for point in waypoints:
            point_loc = carla.Location(x = float(point[0]), y = float(point[1]), z = float(point[2]))
            self.world.debug.draw_point(point_loc, size = 0.2, color = carla.Color(0, 255, 0), life_time = duration)
    
    def draw_single_waypoint(self, waypoint, duration: float = 1):
        point_loc = carla.Location(x = float(waypoint[0]), y = float(waypoint[1]), z = float(waypoint[2]))
        self.world.debug.draw_point(point_loc, size = 0.2, color = carla.Color(0, 255, 0), life_time = duration)