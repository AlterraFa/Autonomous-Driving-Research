import pygame
import numpy as np
import time


from utils.control.world import World
from utils.control.vehicle_control import Vehicle
from utils.control.path import PathHandler, TurnClassify
from utils.sensor_spawner import *
from utils.data_collector import TrajectoryBuffer, CarlaDatasetCollector
from config.enum import JoyControl, JOYBINDS, KEYBINDS

from typing import Optional
from enum import Enum
from rich import print
from typing import Union, Dict
from traceback import print_exc


class CameraView(Enum):
    FIRST_PERSON = {
        "x": 0.0, "y": 0.0, "z": 2,    # position
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0
    }
    THIRD_PERSON = {
        "x": -6.0, "y": 0.0, "z": 3.0,   # behind & above
        "roll": 0.0, "pitch": -10.0, "yaw": 0.0
    }

class CarlaViewer:
    def __init__(self, world: World, vehicle: Vehicle, width: int, height: int, sync: bool = False, fps: int = 70):
        self.virt_world = world
        self.world = world.world
        self.width = width
        self.height = height
        self.sync = sync
        self.fps = fps

        self.virt_vehicle = vehicle
        self.vehicle      = vehicle.vehicle

        self.vehicle_name = vehicle.literal_name()
        self.world_name   = self.world.get_map().name.split("/")[-1]

        self.display: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.running = False
        self.rgb_sensor = None  
        self.sensors_list: Dict[str, Union[RGB, Depth, SemanticSegmentation, GNSS, IMU, LidarRaycast]] = {}
        self.camera_keys = []

        
        self.controller = Controller()
        self.hud = HUD("jetbrainsmononerdfontpropo", fontSize = 12)
        
    def init_sensor(self, sensors: list):
        """Lazy initialize sensors"""
        for sensor in sensors:
            name = sensor.name.split(".")[-1]
            if sensor.name.split(".")[1] == 'camera':
                sensor.set_attribute("image_size_x", self.width)
                sensor.set_attribute("image_size_y", self.height)
                self.camera_keys.append(name)
            sensor.spawn(attach_to = self.vehicle, **CameraView.FIRST_PERSON.value)
            self.sensors_list.update({name: sensor})
            
        if self.camera_keys:
            self.active_cam_idx = 0
            self.choosen_sensor = self.sensors_list[self.camera_keys[self.active_cam_idx]]
            
            print(f"[blue][INFO][/]: Defaulting to {self.choosen_sensor.literal_name}")

    def switch_camera(self, step=1):
        """Switch between camera sensors"""
        if not self.camera_keys:
            return

        self.active_cam_idx = (self.active_cam_idx + step) % len(self.camera_keys)

        cam_name = self.camera_keys[self.active_cam_idx]
        self.choosen_sensor = self.sensors_list[cam_name]

        print(f"[blue][INFO][/] Switched to camera: {self.choosen_sensor.literal_name}")
            
    def add_sensor(self, sensor):
        """Add individual sensor"""
    
            

    def init_win(self, title: str = "CARLA Camera") -> None:
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()


    @staticmethod
    def to_surface(frame: np.ndarray) -> pygame.Surface:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        frame = np.ascontiguousarray(frame)

        # RGBA Processing
        if frame.ndim == 3 and frame.shape[2] == 4:
            h, w, _ = frame.shape
            surf = pygame.image.frombuffer(frame.data, (w, h), "BGRA")
            return surf
        # RGB Processing
        if frame.ndim == 3 and frame.shape[2] == 3:
            h, w, _ = frame.shape
            surf = pygame.image.frombuffer(frame.data, (w, h), "RGB")
            return surf

        # Grayscaled Processing
        if frame.ndim == 2:
            rgb = np.repeat(frame[:, :, None], 3, axis=2)
            h, w, _ = rgb.shape
            return pygame.image.frombuffer(rgb.data, (w, h), "RGB")

        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    def draw_frame(self, frame: np.ndarray) -> None:
        surface = self.to_surface(frame)
        self.display.blit(surface, (0, 0))

    def step_world(self) -> None:
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
            
    def change_view_all(self, view_name: str):
        for camera_name in self.camera_keys:
            self.sensors_list[camera_name].change_view(**getattr(CameraView, view_name).value)
    
    def draw_hud(self, filter_ctrl = False):
        snapshot = self.world.get_snapshot()
        current_platform_time = snapshot.timestamp.platform_timestamp  # server wall clock
        if self.last_platform_time is not None:
            dt_real = current_platform_time - self.last_platform_time
            self.server_fps = 1.0 / dt_real if dt_real > 0 else 0
        self.last_platform_time = current_platform_time

        accel = self.sensors_list['imu'].extract_data().Acceleration
        try: heading = self.sensors_list['imu'].extract_data().Compass * 180 / np.pi
        except: heading = "N/A"
        try: accel = self.sensors_list['imu'].extract_data().Acceleration
        except: accel = "N/A"
        try: gyro = self.sensors_list['imu'].extract_data().Gyroscope
        except: gyro = "N/A"
        try: enu = self.sensors_list['gnss'].extract_data(return_ecf = True, return_enu = True).ENU
        except: enu = "N/A"
        try: geo = self.sensors_list['gnss'].extract_data(return_ecf = True, return_enu = True).Geodetic
        except: geo = "N/A"
        
        self.heading = np.radians(heading)
        self.enu     = enu
        
        client_runtime = time.time() - self.client_start
        server_runtime = snapshot.timestamp.platform_timestamp - self.server_start 
        
        # In carla replay, it does not record velocity => manual calc is needed
        self.velocity = self.virt_vehicle.get_velocity(False)
        if self.velocity < 1e-1:
            curr     = self.vehicle.get_transform().location
            distance = curr.distance(self.prev_loc)
            self.prev_loc = curr
            
            dt = self.world.get_snapshot().timestamp.delta_seconds
            self.velocity = (distance / dt) * 3.6 # Scaled by some factor (close to 3.6 (conversion from m/s to km/h))
            
        self.hud.update_measurement(server_fps = self.server_fps, client_fps = self.clock.get_fps(), 
                                    vehicle_name = self.vehicle_name, world_name = self.world_name,
                                    velocity = self.velocity, heading = heading,
                                    accel = accel, gyro = gyro, enu = enu, geo = geo, 
                                    client_runtime = client_runtime, 
                                    server_runtime = server_runtime)
        
        self.hud.update_control(**self.virt_vehicle.get_ctrl(filter_ctrl), regulate_speed = self.controller.regulate_speed)
        
    @staticmethod
    def to_location(loc):
        if isinstance(loc, carla.Waypoint):
            return loc.transform.location
        if isinstance(loc, carla.Transform):
            return loc.location
        if isinstance(loc, carla.Location):
            return loc
        return None

    @profile
    def run(self, 
            save_logging: str = None, 
            use_temporal_wp: bool = False, 
            data_collect_dir: str = None, 
            replay_logging: str = None, 
            debug = False) -> None:
        if self.display is None:
            self.init_win()

        self.virt_vehicle.set_autopilot(self.controller.autopilot) # First init for autopilot
        self.prev_loc = self.vehicle.get_transform().location

        
        actuation_filter = False
        if save_logging != None:
            trajectory_buff = TrajectoryBuffer(min_dt_s = .2)
        elif replay_logging != None:
            actuation_filter = True
            waypoints_storage = np.load(replay_logging[0])
            path_handling  = PathHandler(waypoints_storage); 
            if data_collect_dir is not None:
                data_collector = CarlaDatasetCollector(save_dir = data_collect_dir, save_interval = 20)
            # For debugging
            # path_handling.position_idx = 60
            # path_handling.position_idx = 250
            # path_handling.position_idx = 300
            # path_handling.position_idx = 760
            # path_handling.position_idx = 1150
            # path_handling.position_idx = 2260
            # path_handling.position_idx = 3980

            scout_points = [i for i in range(-18, 33, 2)]
            if not use_temporal_wp:
                offset          = [3, 5, 7, 8, 9]
            else:
                offset          = [.2, .4, .6, .8, 1.0]
            turn_classifier = TurnClassify(world = self.world, threshold_deg = 15)

        
        try:
            self.last_platform_time = None; 
            self.server_fps = self.fps; 
            self.client_start = time.time()
            self.server_start = self.world.get_snapshot().timestamp.platform_timestamp
            while self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.draw_hud(actuation_filter)

                frame = self.choosen_sensor.extract_data()
                if frame is not None:
                    self.draw_frame(frame)
                    self.hud.draw_measurement(self.display)
                    self.hud.draw_controls(self.display)
                    
                    
                if self.controller.view_changed:
                    self.change_view_all(self.controller.view_name)
                if self.controller.camera_changed:
                    self.switch_camera(self.controller.camera_step)
                self.virt_vehicle.set_autopilot(self.controller.autopilot)
                if self.controller.autopilot == False:
                    self.virt_vehicle.apply_control(self.controller.throt_ctrl, 
                                                    self.controller.steer_ctrl, 
                                                    self.controller.brake_ctrl,
                                                    self.controller.reverse,
                                                    self.controller.hand_brake,
                                                    self.controller.regulate_speed,
                                                    self.controller.has_joystick)
                
                if save_logging != None: 
                    trajectory_buff.add_if_needed(self.vehicle.get_location())
                elif replay_logging != None:
                    position  = self.enu.to_numpy()
                    ego_waypoints, global_waypoints = path_handling.waypoints(position, offset, self.heading, return_global = True, use_time = use_temporal_wp)
                    _, global_scout                 = path_handling.waypoints(position, scout_points, self.heading, return_global = True)
                    if debug:
                        for waypoint in global_waypoints:
                            self.virt_world.draw_single_waypoint(waypoint, 1.5 * (1 / self.server_fps))
                        for waypoint in global_scout:
                            self.virt_world.draw_single_waypoint(waypoint, 1.5 * (1 / self.server_fps), color = (255, 0, 0), size = 0.1)
                    
                    
                    is_at_junction, junction = self.virt_world.get_waypoint_junction(global_scout[14])
                    not_exit_junction, _     = self.virt_world.get_waypoint_junction(global_scout[10])
                    is_exit_junction         = not not_exit_junction
                    turn_signal    = turn_classifier.turning_type(is_at_junction, junction, 
                                                                  is_exit_junction, 
                                                                  global_scout)
                    self.hud.update_logging(turn = turn_signal)
                    self.hud.draw_logging(self.display)
                    
                    if data_collect_dir is not None:
                        data_collector.maybe_save(
                            frame, ego_waypoints, 
                            {
                                'steer': self.hud.ctrl['steer'],
                                'throttle': self.hud.ctrl['throttle'],
                                'brake': self.hud.ctrl['brake'],
                                'velocity': self.velocity
                            }, turn_signal
                        )
                
                pygame.display.flip()
                if self.clock:
                    self.clock.tick(self.fps)



        except KeyboardInterrupt:
            print("[yellow][INFO]: Viewer interrupted by user[/]")
            self.controller.running = False
        except Exception as e:
            print(f"[red][ERROR]: Viewer error: {e}[/]")
            print_exc()
            self.controller.running = False
        finally:
            self.virt_vehicle.stop()
            self.close()
            if save_logging != None:
                trajectory_buff.save(save_logging + "/trajectory")

    def close(self) -> None:
        
        print("[blue][INFO]: Closing CarlaViewer...[/]")

        for name, sensor in list(self.sensors_list.items()):
            sensor.destroy()
        self.virt_world.factory_reset()

        try:
            if pygame.get_init():
                pygame.quit()
                print("[green][INFO]: Pygame closed successfully[/]")
        except Exception as e:
            print(f"[red][ERROR]: Pygame quit failed: {e}[/]")
        
class Controller:
    def __init__(self):
        pygame.joystick.init()

        self.has_joystick = pygame.joystick.get_count() > 0
        if self.has_joystick:
            print("[blue][INFO][/]: Joystick detected, prioritized using it")
            joystick = pygame.joystick.Joystick(0)
            self.joystick = joystick
            self.joystick.init()
            print("[blue][INFO][/]: Joystick name:", joystick.get_name())
            print("[blue][INFO][/]: Number of axes:", joystick.get_numaxes())
            print("[blue][INFO][/]: Number of buttons:", joystick.get_numbuttons())
            print("[blue][INFO][/]: Number of hats:", joystick.get_numhats())
        else:
            print("[yellow][WARNING][/]: No joystick detected. Falling back to keyboard input")

        self.deadzone_stick = 0.12
        self.deadzone_trigger = 0.05
        self.steer_curve = 3  # 1.0 = linear, >1 smoother center
        
        pygame.key.set_repeat()  # no auto-repeat by default
        self.running = True
        
        
        self.view_name = "FIRST_PERSON"; self.view_changed = False
        self.camera_step = 1; self.camera_changed = False
        self.prev_keys_view = pygame.key.get_pressed()
        
        self.autopilot = False
        self.throt_ctrl = 0; self.steer_ctrl = 0; self.brake_ctrl = 0
        self.reverse = False
        self.hand_brake = False
        self.regulate_speed = False
        
    def _apply_deadzone(self, x: float, dz: float) -> float:
        if abs(x) < dz:
            return 0.0
        s = (abs(x) - dz) / (1.0 - dz)
        return s if x > 0 else -s

    def _curve(self, x: float) -> float:
        return (abs(x) ** self.steer_curve) * (1 if x >= 0 else -1)

    def _trigger_01(self, v: float) -> float:
        return max(0.0, min(1.0, (v + 1.0) * 0.5))

    def process_events(self, server_time: float):
        """Process keyboard + window events.
        Returns False if the program should quit."""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_k] or keys[pygame.K_ESCAPE]:
            self.running = False

        self.process_view(events)
        self.process_ctrl(events, server_time)
        return self.running
    
    def process_view(self, events):
        self.view_changed = False; self.camera_changed = False
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.view_name = "FIRST_PERSON"
                    self.view_changed = True
                    print(f"[green][INFO][/]: View toggled → [i]{'First Person'}[/i]")

                elif event.key == pygame.K_DOWN:
                    self.view_name = "THIRD_PERSON"
                    self.view_changed = True
                    print(f"[green][INFO][/]: View toggled → [i]{'Third Person'}[/i]")

                elif event.key == pygame.K_RIGHT:
                    self.camera_changed = True
                    self.camera_step = 1

                elif event.key == pygame.K_LEFT:
                    self.camera_changed = True
                    self.camera_step = -1
                    
            # joystick hat → mirror your view/camera controls
            if self.has_joystick and event.type == pygame.JOYHATMOTION:
                hx, hy = event.value
                if hy == 1:
                    self.view_name = "FIRST_PERSON"; self.view_changed = True
                    print(f"[green][INFO][/]: View toggled → [i]{'First Person'}[/i]")
                elif hy == -1:
                    self.view_name = "THIRD_PERSON"; self.view_changed = True
                    print(f"[green][INFO][/]: View toggled → [i]{'Third Person'}[/i]")
                if hx != 0:
                    self.camera_changed = True
                    self.camera_step = 1 if hx > 0 else -1
    
    def toggle_autopilot(self):
        self.autopilot = not self.autopilot
        print(f"[yellow][WARNING][/]: Autopilot toggled → "
            f"[i][{'green' if self.autopilot else 'red'}]"
            f"{'Engaged' if self.autopilot else 'Disengaged'}[/i][/]")

    def toggle_reverse(self):
        self.reverse = not self.reverse
        print(f"[blue][INFO][/]: Reverse {'ON' if self.reverse else 'OFF'}")

    def toggle_hand_brake(self):
        self.hand_brake = not self.hand_brake

    def toggle_regulate_speed(self):
        self.regulate_speed = not self.regulate_speed

    def process_ctrl(self, events, server_time):
        self.throt_ctrl = 0
        self.steer_ctrl = 0
        self.brake_ctrl = 0

        for event in events:
            # Handle quit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                return False

            # Keyboard toggles
            if event.type == pygame.KEYDOWN and event.key in KEYBINDS:
                getattr(self, KEYBINDS[event.key])()

            # Joystick toggles
            if self.has_joystick and event.type == pygame.JOYBUTTONDOWN and event.button in JOYBINDS:
                getattr(self, JOYBINDS[event.button])()

        # Keyboard continuous controls
        keys = pygame.key.get_pressed()
        steer_inc = 5e-4 * server_time * 1000
        self.throt_ctrl = 0.01 if keys[pygame.K_w] else 0
        self.brake_ctrl = 0.2 if keys[pygame.K_s] else 0
        self.steer_ctrl = (-steer_inc if keys[pygame.K_a] else
                            steer_inc if keys[pygame.K_d] else 0)

        # Joystick continuous controls
        if self.has_joystick:
            left_x = self._apply_deadzone(self.joystick.get_axis(JoyControl.JoyStick.LX), self.deadzone_stick)
            lt = self._trigger_01(self.joystick.get_axis(JoyControl.JoyStick.LT))
            rt = self._trigger_01(self.joystick.get_axis(JoyControl.JoyStick.RT))

            lt = 0.0 if lt < self.deadzone_trigger else lt
            rt = 0.0 if rt < self.deadzone_trigger else rt

            self.steer_ctrl = self._curve(left_x) * 0.5
            self.throt_ctrl = rt
            self.brake_ctrl = lt
        
class HUD:
    def __init__(self, fontName="Arial", fontSize=24):
        pygame.font.init()
        self.font = pygame.font.SysFont(fontName, fontSize, bold = True)
        self.height = 20
        
        self.measurement = {
            "server_fps": 0,
            "client_fps": 0,
            "vehicle_name": "",
            "world_name": "",
            "velocity": 0,
            "heading": 0,
            "accel": "N/A",
            "gyro": "N/A",
            "enu": "N/A",
            "geo": "N/A",
            "client_runtime": 0,
            "server_runtime": 0,
        }
        
        self.ctrl = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "reverse": False,
            "handbrake": False,
            "manual": False,
            "gear": 0,
            'autopilot': False,
            'regulate_speed': False
        }
    
        self.logging = {
            "turn": -1
        }
        
    def update_measurement(self, **kwargs):
        for key in self.measurement:
            if key in kwargs:
                self.measurement[key] = kwargs[key]

    def update_control(self, **kwargs):
        for key in self.ctrl:
            if key in kwargs:
                self.ctrl[key] = kwargs[key]
                
    def update_logging(self, **kwargs):
        for key in self.logging:
            if key in kwargs:
                self.logging[key] = kwargs[key]
        
    @staticmethod
    def heading_to_cardinal(deg: float) -> str:
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((deg + 22.5) % 360 // 45)
        return directions[idx] if isinstance(deg, Union[float, int]) else ""

    def _time_to_str(self, t: float) -> str:
        """Convert seconds float into HH:MM:SS.mmm string."""
        hours   = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        millis  = int((t % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _render_line(self, surface, label: str, value: str, line_idx: int, x: int = 10, y: int = 10, spacing: int = 15):
        """Helper for rendering a single line with label and value."""
        text = self.font.render(f"{label:<{spacing}}{value:>{self.max_string}}", True, (255, 255, 255))
        surface.blit(text, (x, y + self.height * line_idx))

    def draw_measurement(self, surface: pygame.Surface):
        # Transparent overlay
        overlay = pygame.Surface((310, surface.get_height()), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 100), overlay.get_rect())
        surface.blit(overlay, (0, 0))

        # Shortcuts
        m = self.measurement

        # Format values
        accel = m["accel"]
        gyro  = m["gyro"]
        geo   = m["geo"]
        enu   = m["enu"]

        accel_str = accel if isinstance(accel, str) else f"( {accel[0]: 6.2f}, {accel[1]: 6.2f}, {accel[2]: 6.2f} )"
        gyro_str  = gyro  if isinstance(gyro, str)  else f"( {gyro[0]: 6.2f}, {gyro[1]: 6.2f}, {gyro[2]: 6.2f} )"
        geo_str   = geo   if isinstance(geo, str)   else f"( {geo.lat: 6.6f}, {geo.lon: 6.6f} )"

        client_time_str = self._time_to_str(m["client_runtime"])
        server_time_str = self._time_to_str(m["server_runtime"])

        if isinstance(enu, str):
            h_str, loc_str = "N/A", "N/A"
        else:
            h_str   = f"{enu.up: 6.2f} m"
            loc_str = f"( {enu.east: 6.2f}, {enu.north: 6.2f} )"

        # Collect all strings (label, value, line index)
        value_lines = [
            ("Server side:",   f"{int(m['server_fps'])} FPS", 0),
            ("Client side:",   f"{int(m['client_fps'])} FPS", 1),
            ("Client runtime:", f"{client_time_str} s",       2),
            ("Server runtime:", f"{server_time_str} s",       3),
            ("Vehicle name:",   m["vehicle_name"],            5),
            ("World name:",     m["world_name"],              6),
            ("Velocity:",       f"{m['velocity']:.2f} (km/h)",8),
            ("Heading:",        f"{m['heading']:.1f}° {self.heading_to_cardinal(m['heading'])}", 9),
            ("Acceleration:",   accel_str,                   10),
            ("Gyroscope:",      gyro_str,                    11),
            ("Location:",       loc_str,                     12),
            ("Geodetic:",       geo_str,                     13),
            ("Height:",         h_str,                       14),
        ]


        # For alignment
        self.max_string = max(max(len(v) for _, v, _ in value_lines), 15)

        # Render all lines
        for label, value, idx in value_lines:
            self._render_line(surface, label, value, idx)
        
    def draw_controls(self, surface, x=10, y=330):
        line_h = 20
        bar_w, bar_h = 150, 10   # bar size
        bar_x = x + 100          # where bars start (shifted right of labels)

        white = (255, 255, 255)
        green = (0, 200, 0)
        red   = (200, 0, 0)

        # ---- Throttle bar ----
        surface.blit(self.font.render("Throttle:", True, white), (x, y))
        pygame.draw.rect(surface, white, (bar_x, y+5, bar_w, bar_h), 1)  # outline
        pygame.draw.rect(surface, green, (bar_x, y+5, min(int(bar_w * self.ctrl['throttle']), bar_w), bar_h))

        # ---- Steer bar (centered at 0) ----
        surface.blit(self.font.render("Steer:", True, white), (x, y+line_h))
        pygame.draw.rect(surface, white, (bar_x, y+line_h+5, bar_w, bar_h), 1)  # outline
        steer_val = self.ctrl['steer']  # -1 .. +1
        if steer_val >= 0:
            # fill right side
            pygame.draw.rect(surface, green, (bar_x + bar_w//2, y+line_h+5,
                                            min(int((bar_w//2) * steer_val), (bar_w//2)), bar_h))
        else:
            # fill left side
            pygame.draw.rect(surface, green, (bar_x + bar_w//2 + int((bar_w//2)*steer_val), y+line_h+5,
                                            max(int(-(bar_w//2) * steer_val), -(bar_w//2)), bar_h))

        # ---- Brake bar ----
        surface.blit(self.font.render("Brake:", True, white), (x, y+2*line_h))
        pygame.draw.rect(surface, white, (bar_x, y+2*line_h+5, bar_w, bar_h), 1)  # outline
        pygame.draw.rect(surface, red, (bar_x, y+2*line_h+5, min(int(bar_w * self.ctrl['brake']), bar_w), bar_h))

        # ---- Others as text ----
        spacing = 33
        surface.blit(self.font.render(f"{'Throttle:':<{spacing}} {'■' if self.ctrl['reverse'] else '□'}", True, white), (x, y+3*line_h))
        surface.blit(self.font.render(f"{'Hand brake:':<{spacing}} {'■' if self.ctrl['handbrake'] else '□'}", True, white), (x, y+4*line_h))
        surface.blit(self.font.render(f"{'Manual:':<{spacing}} {'■' if self.ctrl['manual'] else '□'}", True, white), (x, y+5*line_h))
        surface.blit(self.font.render(f"{'Gear:':<{spacing}} {self.ctrl['gear']}", True, white), (x, y+6*line_h))
        surface.blit(self.font.render(f"{'Autopilot:':<{spacing}} {'■' if self.ctrl['autopilot'] else '□'}", True, (255,255,255)), (x, y+7*line_h))
        surface.blit(self.font.render(f"{'Regulate speed:':<{spacing}} {'■' if self.ctrl['regulate_speed'] else '□'}", True, (255,255,255)), (x, y+8*line_h))

    def draw_logging(self, surface, x = 10, y = 510):
        line_h = 20
        
        if self.logging['turn'] == -1:
            direction_str = "Keep lane"
        elif self.logging['turn'] == 0:
            direction_str = "Go straight"
        elif self.logging['turn'] == 1:
            direction_str = "Turn left"
        elif self.logging['turn'] == 2:
            direction_str = "Turn right"
        else:
            direction_str = "N/A"

        spacing = 15

        directionText = self.font.render(f"{'Turn signal:':<{spacing}}{f'{direction_str}':>{self.max_string}}", True, (255, 255, 255))
        surface.blit(directionText, (x, y + 1 * line_h)) 
