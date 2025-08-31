import pygame
import numpy as np
import time


from control.world import World
from control.vehicle_control import Vehicle
from control.path import PathHandler, TurnClassify
from utils.sensor_spawner import *
from utils.buffer import TrajectoryBuffer
from config.enum import JoyControl

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
    
    def draw_hud(self):
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
        
        runtime = time.time() - getattr(self, "start_time", time.time())
        
        self.hud.update_measurement(server_fps = self.server_fps, client_fps = self.fps, 
                        vehicle_name = self.vehicle_name, world_name = self.world_name,
                        velocity = self.virt_vehicle.get_velocity(False), heading = heading,
                        accel = accel, gyro = gyro, enu = enu, geo = geo, runtime = runtime)
        
        self.hud.update_control(**self.virt_vehicle.get_ctrl(), regulate_speed = self.controller.regulate_speed)
        
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
    def run(self, save_logging: str = None, record_type: str = "location", replay_logging: str = None, debug = False) -> None:
        if self.display is None:
            self.init_win()

        self.last_platform_time = None; self.server_fps = self.fps; self.start_time = time.time()
        self.virt_vehicle.set_autopilot(self.controller.autopilot) # First init for autopilot

        if save_logging != None:
            trajectory_buff = TrajectoryBuffer(dist_thresh_m = 1.0)
        elif replay_logging != None:
            waypoints_storage = np.load(replay_logging[0])
            path_handling = PathHandler(waypoints_storage); 
            path_handling.position_idx = 60
            # path_handling.position_idx = 790
            offset = [3, 5, 7, 9, 11]
            scout = [12, 14, 16, 18, 20]
            turn_classifier = TurnClassify(4.8)
        
        try:
            while self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.draw_hud()

                frame = self.choosen_sensor.extract_data()
                if frame is not None:
                    self.draw_frame(frame)
                    self.hud.draw(self.display)
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
                    trajectory_buff.add_if_needed(self.to_location(getattr(self.virt_vehicle, record_type)))
                elif replay_logging != None:
                    position  = self.enu.to_numpy()
                    ego_waypoints, global_waypoints = path_handling.waypoints(position, offset, self.heading, return_global = True)
                    for waypoint in global_waypoints:
                        self.virt_world.draw_single_waypoint(waypoint, 1.5 * (1 / self.server_fps))
                    ego_scout, global_scout = path_handling.waypoints(position, scout, self.heading, return_global = True)
                    # for waypoint in global_scout:
                    #     self.virt_world.draw_single_waypoint(waypoint, 1.5 * (1 / self.server_fps), color = (255, 0, 0))
                    
                    
                    is_at_junction = self.virt_world.get_waypoint_junction(global_waypoints[-1])
                    exit_junction  = not self.virt_world.get_waypoint_junction(global_waypoints[2])
                    turn_signal    = turn_classifier.turning_type(is_at_junction, exit_junction, np.r_[ego_waypoints, ego_scout])
                    print("Go straight" if turn_signal == 0 else "Turn left" if turn_signal == 1 else "Turn right" if turn_signal == 2 else None, path_handling.position_idx)
                
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


    def process_ctrl(self, events, server_time):
        self.throt_ctrl = 0; self.steer_ctrl = 0; self.brake_ctrl = 0
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False

                elif event.key == pygame.K_BACKQUOTE:   # the ` key under ESC
                    self.autopilot = not self.autopilot
                    print(f"[yellow][WARNING][/]: Autopilot toggled → [i][{'green'if self.autopilot else 'red'}]{'Engaged' if self.autopilot else 'Disengaged'}[/i][/]")

                elif event.key == pygame.K_q:
                    self.reverse = not self.reverse
                    
                elif event.key == pygame.K_SPACE:
                    self.hand_brake = not self.hand_brake
                
                elif event.key == pygame.K_f:
                    self.regulate_speed = not self.regulate_speed
                    
                
        keys = pygame.key.get_pressed()
        steer_inc = 5e-4 * server_time * 1000
        if keys[pygame.K_w]:
            self.throt_ctrl = 0.01
        if keys[pygame.K_s]:
            self.brake_ctrl = 0.2
        if keys[pygame.K_a]:
            self.steer_ctrl = -steer_inc
        if keys[pygame.K_d]:
            self.steer_ctrl = steer_inc

        if self.has_joystick:
            for event in events:
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == JoyControl.JoyKey.A:
                        self.autopilot = not self.autopilot
                        print(f"[yellow][WARNING][/]: Autopilot toggled → [i][{'green' if self.autopilot else 'red'}]{'Engaged' if self.autopilot else 'Disengaged'}[/i][/]")
                    elif event.button == JoyControl.JoyKey.B:
                        self.reverse = not self.reverse
                    elif event.button == JoyControl.JoyKey.LB:
                        self.hand_brake = not self.hand_brake
                    elif event.button == JoyControl.JoyKey.RB:
                        self.regulate_speed = not self.regulate_speed
                    
            left_x = self._apply_deadzone(self.joystick.get_axis(JoyControl.JoyStick.LX), self.deadzone_stick)
            # left_y = self._apply_deadzone(self.joystick.get_axis(1), self.deadzone_stick)

            lt = self._trigger_01(self.joystick.get_axis(JoyControl.JoyStick.LT))
            rt = self._trigger_01(self.joystick.get_axis(JoyControl.JoyStick.RT))
            lt = 0.0 if lt < self.deadzone_trigger else lt
            rt = 0.0 if rt < self.deadzone_trigger else rt

            self.steer_ctrl = self._curve(left_x) * .5

            self.throt_ctrl = rt
            self.brake_ctrl = lt
        
class HUD:
    def __init__(self, fontName="Arial", fontSize=24):
        pygame.font.init()
        self.font = pygame.font.SysFont(fontName, fontSize, bold = True)
        self.server_fps = 0
        self.client_fps = 0
        self.vehicle_name = ""
        self.world_name = ""
        self.velocity = 0
        self.heading = 0
        self.accel = 'N/A'
        self.gyro = 'N/A'
        self.enu = 'N/A'
        self.geo = 'N/A'
        self.runtime = 0
        
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
        
    def update_measurement(self, **kwargs):
        self.client_fps = kwargs.get("client_fps", self.client_fps)
        self.server_fps = kwargs.get("server_fps", self.server_fps)
        self.vehicle_name = kwargs.get("vehicle_name", self.vehicle_name)
        self.world_name = kwargs.get("world_name", self.world_name)
        self.velocity = kwargs.get("velocity", self.velocity)
        self.heading = kwargs.get("heading", self.heading)
        self.accel = kwargs.get("accel", self.accel)
        self.gyro  = kwargs.get("gyro", self.gyro)
        self.enu   = kwargs.get('enu', self.enu)
        self.geo   = kwargs.get('geo', self.geo)
        self.runtime = kwargs.get("runtime", self.runtime)

    def update_control(self, **kwargs):
        for key in self.ctrl:
            if key in kwargs:
                self.ctrl[key] = kwargs[key]

        
    @staticmethod
    def heading_to_cardinal(deg: float) -> str:
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((deg + 22.5) % 360 // 45)
        return directions[idx] if isinstance(deg, Union[float, int]) else ""

    def draw(self, surface: pygame.Surface):

        overlay = pygame.Surface((310, surface.get_height()), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 100), overlay.get_rect())
        surface.blit(overlay, (0, 0))
        
        if isinstance(self.accel, str): accel_str = self.accel
        else: accel_str = f"( {self.accel[0]: 6.2f}, {self.accel[1]: 6.2f}, {self.accel[2]: 6.2f} )"
        if isinstance(self.gyro, str): gyro_str = self.gyro
        else: gyro_str = f"( {self.gyro[0]: 6.2f}, {self.gyro[1]: 6.2f}, {self.gyro[2]: 6.2f} )"
        if isinstance(self.geo, str): geo_str = self.geo
        else: geo_str = f"( {self.geo.lat: 6.6f}, {self.geo.lon: 6.6f} )"
        
        runtime = self.runtime
        hours   = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        millis  = int((runtime % 1) * 1000)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
        
        if isinstance(self.enu, str):
            h_str = "N/A"
            loc_str = "N/A"
        else:
            h_str = f"{self.enu.up: 6.2f} m" 
            loc_str = f"( {self.enu.east: 6.2f}, {self.enu.north: 6.2f} )"


        value_strings = [
            f"{int(self.server_fps)} FPS",
            f"{int(self.client_fps)} FPS",
            self.vehicle_name,
            self.world_name,
            f'{self.velocity:.2f} (km/h)',
            f'{self.heading}',
            accel_str,
            loc_str,
            h_str,
            geo_str,
            time_str
        ]
        max_string = max(max(len(s) for s in value_strings), 15)
        spacing = 15
        
        serverfpsText = self.font.render(f"{'Server side:':<{spacing}}{f'{int(self.server_fps)} FPS':>{max_string}}", True, (255, 255, 255))
        surface.blit(serverfpsText, (10, 10)) 

        clientfpsText = self.font.render(f"{'Client side:':<{spacing}}{f'{int(self.client_fps)} FPS':>{max_string}}", True, (255, 255, 255))
        surface.blit(clientfpsText, (10, 30)) 

        runtimeText = self.font.render(f"{'Runtime:':<{spacing}}{f'{time_str} s':>{max_string}}", True, (255, 255, 255))
        surface.blit(runtimeText, (10, 50)) 

        vehicleText = self.font.render(f"{'Vehicle name:':<{spacing}}{self.vehicle_name:>{max_string}}", True, (255, 255, 255))
        surface.blit(vehicleText, (10, 90)) 

        worldText = self.font.render(f"{'World name:':<{spacing}}{self.world_name:>{max_string}}", True, (255, 255, 255))
        surface.blit(worldText, (10, 110)) 

        velocityText = self.font.render(f"{'Velocity:':<{spacing}}{f'{self.velocity:.2f} (km/h)':>{max_string}}", True, (255, 255, 255))
        surface.blit(velocityText, (10, 150)) 

        headingText = self.font.render(f"{'Heading:':<{spacing}}{f'{self.heading:.1f}° {self.heading_to_cardinal(self.heading)}':>{max_string}}", True, (255, 255, 255))
        surface.blit(headingText, (10, 170)) 

        accelText = self.font.render(f"{'Acceleration:':<{spacing}}{f'{accel_str}':>{max_string}}", True, (255, 255, 255))
        surface.blit(accelText, (10, 190)) 
        
        gyroText = self.font.render(f"{'Gyroscope:':<{spacing}}{f'{gyro_str}':>{max_string}}", True, (255, 255, 255))
        surface.blit(gyroText, (10, 210)) 

        locText = self.font.render(f"{'Location:':<{spacing}}{f'{loc_str}':>{max_string}}", True, (255, 255, 255))
        surface.blit(locText, (10, 230)) 

        geoText = self.font.render(f"{'Geodetic:':<{spacing}}{f'{geo_str}':>{max_string}}", True, (255, 255, 255))
        surface.blit(geoText, (10, 250)) 

        heightText = self.font.render(f"{'Height:':<{spacing}}{f'{h_str}':>{max_string}}", True, (255, 255, 255))
        surface.blit(heightText, (10, 270)) 
        
    def draw_controls(self, surface, x=10, y=310):
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