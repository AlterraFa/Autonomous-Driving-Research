import pygame
import numpy as np
import time
import torch
import concurrent.futures
import cv2


from utils.control.world import World
from utils.control.vehicle_control import Vehicle
from utils.math.path import ReplayHandler
from utils.spawn.sensor_spawner import *
from utils.others.data_processor import TrajectoryBuffer
from model.inference import AsyncInference
from config.enum import (
    JoyControl, 
    CameraView,
    JOYBINDS, 
    KEYBINDS, 
)
from utils.messages.message_handler import (
    MessagingSenders,
    MessagingSubscribers 
)
from utils.messages.logger import Logger

from typing import Optional
from typing import Union, Dict
from traceback import print_exc
        

def generate_controller_doc(keybinds: dict, joybinds: dict) -> str:
    """
    Generate documentation string for Controller based on KEYBINDS and JOYBINDS.
    """
    import pygame

    # Map pygame key constants to readable names
    key_names = {k: pygame.key.name(k).upper() for k in keybinds.keys()}

    doc = []
    doc.append("Welcome to CARLA Manual Control (Custom Controller).\n")
    doc.append("Controls can be provided via Keyboard or Joystick.")
    doc.append("Joystick (if detected) is prioritized automatically.")
    doc.append("If no joystick is found, keyboard input is used as fallback.\n")

    doc.append("----------------------------------------")
    doc.append("Keyboard Controls")
    doc.append("----------------------------------------")
    for k, func in keybinds.items():
        doc.append(f"    {key_names[k]:<12} : {func.replace('toggle_', '').replace('_', ' ')}")

    # Always include quit keys
    doc.append(f"    {'K/ESC':<12} : quit program")

    doc.append("\n----------------------------------------")
    doc.append("Joystick Controls")
    doc.append("----------------------------------------")
    doc.append("Axes:")
    doc.append("    Left Stick X : steer left / right (with deadzone and curve applied)")
    doc.append("    RT (Right Trigger) : throttle")
    doc.append("    LT (Left Trigger)  : brake\n")

    doc.append("Hats (D-pad):")
    doc.append("    Up           : switch to First Person view")
    doc.append("    Down         : switch to Third Person view")
    doc.append("    Left / Right : step camera left / right\n")

    doc.append("Buttons:")
    for btn, func in joybinds.items():
        doc.append(f"    Button {btn:<3} : {func.replace('toggle_', '').replace('_', ' ')}")

    doc.append("\n----------------------------------------")
    doc.append("Modes")
    doc.append("----------------------------------------")
    doc.append("    • Autopilot and Model Autopilot are mutually exclusive.")
    doc.append("      Enabling one disables the other automatically.\n")
    doc.append("    • Regulate Speed mode maintains a constant velocity until disabled.\n")

    doc.append("----------------------------------------")
    doc.append("Notes")
    doc.append("----------------------------------------")
    doc.append("    • Deadzones:")
    doc.append("        Stick deadzone   = 0.12")
    doc.append("        Trigger deadzone = 0.05\n")
    doc.append("    • Steering curve applied for smoother control:")
    doc.append("        steer_curve = 3\n")
    doc.append("    • Keyboard inputs are only active when Model Autopilot is OFF.")
    doc.append("      In Model Autopilot mode, keys [W/A/D] send turn signal messages.")

    return "\n".join(doc)

Logger().INFO(generate_controller_doc(KEYBINDS, JOYBINDS))


class CarlaViewer(MessagingSenders, MessagingSubscribers):
    def __init__(self, world: World, vehicle: Vehicle, width: int, height: int, sync: bool = False, fps: int = 70):
        self.log = Logger() 
        MessagingSenders.__init__(self)
        MessagingSubscribers.__init__(self)

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

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future   = None
        
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
            
            self.log.INFO(f"Defaulting to {self.choosen_sensor.literal_name}")


    def switch_camera(self, step=1):
        """Switch between camera sensors"""
        if not self.camera_keys:
            return

        self.active_cam_idx = (self.active_cam_idx + step) % len(self.camera_keys)

        cam_name = self.camera_keys[self.active_cam_idx]
        self.choosen_sensor = self.sensors_list[cam_name]

        self.log.DEBUG(f"Switched to camera - [bold]{self.choosen_sensor.literal_name}[/]")

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
        surface = self.to_surface(frame.copy())
        self.display.blit(surface, (0, 0))

    def step_world(self) -> None:
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
            
    def change_view_all(self, view_name: str):
        for camera_name in self.camera_keys:
            self.sensors_list[camera_name].change_view(**getattr(CameraView, view_name).value)
    
    def data_bus(self, filter_ctrl=False):
        snapshot = self.world.get_snapshot()
        current_platform_time = snapshot.timestamp.platform_timestamp  # server wall clock
        if self.last_platform_time is not None:
            dt_real = current_platform_time - self.last_platform_time
            self.server_fps = 1.0 / dt_real if dt_real > 0 else 0
        self.last_platform_time = current_platform_time

        # Extract sensor data with safe fallback
        try:
            heading = self.sensors_list['imu'].extract_data().Compass * 180 / np.pi
        except:
            heading = "N/A"
        try:
            accel = self.sensors_list['imu'].extract_data().Acceleration
        except:
            accel = "N/A"
        try:
            gyro = self.sensors_list['imu'].extract_data().Gyroscope
        except:
            gyro = "N/A"
        try:
            enu = self.sensors_list['gnss'].extract_data(return_ecf=True, return_enu=True).ENU
        except:
            enu = "N/A"
        try:
            geo = self.sensors_list['gnss'].extract_data(return_ecf=True, return_enu=True).Geodetic
        except:
            geo = "N/A"

        self.heading = np.radians(heading) if isinstance(heading, (int, float)) else heading
        self.enu = enu

        client_runtime = time.time() - self.client_start
        server_runtime = snapshot.timestamp.elapsed_seconds - self.server_start

        # Velocity fallback
        self.velocity = self.virt_vehicle.get_velocity(False)

        #  Publish to subscribers
        self.send_server_fps.send(self.server_fps)
        self.send_client_fps.send(self.clock.get_fps())
        self.send_vehicle_name.send(self.vehicle_name)
        self.send_world_name.send(self.world_name)
        self.send_velocity.send(self.velocity)
        self.send_heading.send(np.degrees(self.heading))
        self.send_accel.send(accel)
        self.send_gyro.send(gyro)
        self.send_enu.send(enu.to_numpy())
        self.send_geo.send(geo.to_numpy())
        self.send_client_runtime.send(client_runtime)
        self.send_server_runtime.send(server_runtime)
        vehicle_loc = self.vehicle.get_location()
        self.send_location.send(np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z]))

        self.ctrl = self.virt_vehicle.get_ctrl(filter_ctrl)
        self.send_model_autopilot_logging.send(self.ctrl['model_autopilot'])
        self.send_autopilot_logging.send(self.ctrl['autopilot'])
        self.send_regulate_speed_logging.send(self.ctrl['regulate_speed'])
        self.send_throttle_logging.send(self.ctrl['throttle'])
        self.send_steer_logging.send(self.ctrl['steer'])
        self.send_brake_logging.send(self.ctrl['brake'])
        self.send_reverse_logging.send(self.ctrl['reverse'])
        self.send_handbrake_logging.send(self.ctrl['handbrake'])
        self.send_manual_logging.send(self.ctrl['manual'])
        self.send_gear_logging.send(self.ctrl['gear'])
        
    def run(self, 
            model = None,
            save_logging: str = None, 
            use_temporal_wp: bool = False, 
            data_collect_dir: str = None, 
            replay_logging: str = None, 
            debug = False) -> None:
        if self.display is None:
            self.init_win()

        self.virt_vehicle.set_autopilot(self.controller.autopilot) # First init for autopilot
        self.prev_loc = self.vehicle.get_transform().location

        
        logger    = TrajectoryBuffer(save_logging, min_dt_s = .2) if save_logging else None
        replayer  = ReplayHandler(replay_logging[0], self.virt_world, data_collect_dir, use_temporal_wp, debug) if replay_logging else None
        inference = AsyncInference(model) if model is not None else None

        H, W, _    = 720, 1280, 3
        x_top_left = 70; x_top_right = W - x_top_left
        x_bot_left = 20; x_bot_right = W - x_bot_left
        y_hor      = 390; y_bot         = 720
        src_points = np.float32([[x_top_left, y_hor],
                                [x_top_right, y_hor],
                                [x_bot_right, y_bot],
                                [x_bot_left, y_bot]])
        width = 270; height = 150
        dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        frame_id = 0
        try:
            self.last_platform_time = None; 
            self.server_fps = self.fps; 
            self.client_start = time.time()
            self.server_start = self.world.get_snapshot().timestamp.elapsed_seconds
            while self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.data_bus(replay_logging != None)

                frame = self.choosen_sensor.extract_data()
                try:
                    H, W, _ = frame.shape
                except:
                    H, W = frame.shape


                if frame is not None:
                    self.draw_frame(frame.copy())
                    self.hud.draw_measurement(self.display)
                    self.hud.draw_controls(self.display)
                    self.hud.draw_logging(self.display)
                    
                    
                if self.controller.view_changed:
                    self.change_view_all(self.controller.view_name)
                if self.controller.camera_changed:
                    self.switch_camera(self.controller.camera_step)
                self.virt_vehicle.set_autopilot(self.controller.autopilot)
                self.virt_vehicle.set_model_autopilot(self.controller.model_autopilot)
                if self.controller.autopilot == False:
                    self.virt_vehicle.apply_control(self.controller.regulate_speed,
                                                    self.controller.has_joystick, 
                                                    self.controller.model_autopilot)
                    
                if logger: 
                    logger.update(self.sub_location.receive())
                if replayer: 
                    replayer.step(frame)
                if model and self.controller.model_autopilot:
                    
                    
                    if frame_id % 5 == 0:
                        inp = cv2.warpPerspective(frame[:, :, :3], M, (width, height))
                        turn_signal = self.sub_turn_signal.receive()
                        inference.put(inp, turn_signal)
                        steer = inference.get()
                        if steer is not None:
                            self.send_model_steer.send(float(steer))
                        
                        preview_h, preview_w, _ = inp.shape
                        inp_surface = self.to_surface(inp[:, :, ::-1])
                        self.display.blit(inp_surface, (self.width - preview_w - 10, self.height - preview_h - 10))
                    frame_id += 1
                    
                    # local_wp = infer(model, inp)[0]
                    # local_wp[:, 1] = -local_wp[:, 1]
                    # global_wp = self.virt_vehicle.global_transform(local_wp, np.radians(self.sub_heading.receive()))
                    # # self.virt_world.draw_waypoints(global_wp, duration = 1 * (1 / self.server_fps))
                

                if replay_logging is not None and replay_logging[1] <= self.sub_server_runtime.receive():
                    self.log.INFO("Reached replay limit. Goodbye.")
                    break

                pygame.display.flip()
                if self.clock:
                    self.clock.tick(self.fps)



        except KeyboardInterrupt:
            self.log.WARNING("Viewer interrupted by user")
            self.controller.running = False
        except Exception as e:
            self.log.ERROR("Viewer error", full_traceback = e)
            self.controller.running = False
        finally:
            self.virt_vehicle.stop()
            self.close()
            if logger:
                logger.finalize()

    def close(self) -> None:
        
        self.log.WARNING("Closing CarlaViewer...")

        for name, sensor in list(self.sensors_list.items()):
            sensor.destroy()
        self.virt_world.factory_reset()

        try:
            if pygame.get_init():
                pygame.quit()
                self.log.CUSTOM("SUCCESS", "Pygame closed successfully!")
        except Exception as e:
            self.log.ERROR("Pygame quit failed", full_traceback = e)
        
class Controller(MessagingSenders):
    def __init__(self):
        self.log = Logger()
        MessagingSenders.__init__(self)
        pygame.joystick.init()

        self.has_joystick = pygame.joystick.get_count() > 0
        if self.has_joystick:
            self.log.INFO("Joystick detected, prioritized using it")
            joystick = pygame.joystick.Joystick(0)
            self.joystick = joystick
            self.joystick.init()
            self.log.DEBUG(f"Joystick name: {joystick.get_name()}")
            self.log.DEBUG(f"Number of axes: {joystick.get_numaxes()}")
            self.log.DEBUG(f"Number of buttons: {joystick.get_numbuttons()}")
            self.log.DEBUG(f"Number of hats: {joystick.get_numhats()}")
        else:
            self.log.WARNING("No joystick detected. Falling back to keyboard input")

        self.deadzone_stick = 0.12
        self.deadzone_trigger = 0.05
        self.steer_curve = 3  # 1.0 = linear, >1 smoother center
        
        pygame.key.set_repeat()  # no auto-repeat by default
        self.running = True
        
        
        self.view_name = "FIRST_PERSON"; self.view_changed = False
        self.camera_step = 1; self.camera_changed = False
        self.prev_keys_view = pygame.key.get_pressed()
        
        self.autopilot = False; self.model_autopilot = False
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
                    self.log.DEBUG(f"View toggled → [i]{'First Person'}[/i]")

                elif event.key == pygame.K_DOWN:
                    self.view_name = "THIRD_PERSON"
                    self.view_changed = True
                    self.log.DEBUG(f"View toggled → [i]{'Third Person'}[/i]")

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
                    self.log.DEBUG(f"View toggled → [i]{'First Person'}[/i]")
                elif hy == -1:
                    self.view_name = "THIRD_PERSON"; self.view_changed = True
                    self.log.DEBUG(f"View toggled → [i]{'Third Person'}[/i]")
                if hx != 0:
                    self.camera_changed = True
                    self.camera_step = 1 if hx > 0 else -1
    
    def toggle_autopilot(self):
        self.autopilot = not self.autopilot
        self.log.WARNING(
            f"Autopilot toggled → "
            f"[i][{'green' if self.autopilot else 'red'}]"
            f"{'Engaged' if self.autopilot else 'Disengaged'}[/i][/]"
        )

        if self.autopilot == True:
            self.model_autopilot = False

    def toggle_model_autopilot(self):
        self.model_autopilot = not self.model_autopilot
        self.log.WARNING(
            f"Model inference toggled → "
            f"[i][{'green' if self.model_autopilot else 'red'}]"
            f"{'Engaged' if self.model_autopilot else 'Disengaged'}[/i][/]"
        )
        
        if self.model_autopilot == True:
            self.autopilot = False

    def toggle_reverse(self):
        self.reverse = not self.reverse
        self.log.INFO(f"Reverse [bold][i]{'ON' if self.reverse else 'OFF'}[/][/]")

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
        if not self.model_autopilot:
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

        if self.model_autopilot:
            if keys[pygame.K_w]:
                self.send_turn_signal.send(0)
            elif keys[pygame.K_a]:
                self.send_turn_signal.send(1)
            elif keys[pygame.K_d]:
                self.send_turn_signal.send(2)
            else:
                self.send_turn_signal.send(-1)
            
        self.send_throttle.send(self.throt_ctrl)
        self.send_steer.send(self.steer_ctrl)
        self.send_brake.send(self.brake_ctrl)
        self.send_reverse.send(self.reverse)
        self.send_handbrake.send(self.hand_brake)
        self.send_regulate_speed.send(self.regulate_speed)

class HUD(MessagingSubscribers):
    def __init__(self, fontName="Arial", fontSize=24):
        super().__init__()  # init all subscribers
        pygame.font.init()
        self.font = pygame.font.SysFont(fontName, fontSize, bold=True)
        self.height = 20

    @staticmethod
    def heading_to_cardinal(deg: float) -> str:
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        if not isinstance(deg, (int, float)):
            return ""
        idx = int((deg + 22.5) % 360 // 45)
        return directions[idx]

    def _time_to_str(self, t: float) -> str:
        hours   = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        millis  = int((t % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _render_line(self, surface, label: str, value: str, line_idx: int, x: int = 10, y: int = 10, spacing: int = 15):
        text = self.font.render(f"{label:<{spacing}}{value:>{self.max_string}}", True, (255, 255, 255))
        surface.blit(text, (x, y + self.height * line_idx))

    def _read(self, sub, default="N/A"):
        """Helper to read latest subscriber value or fallback."""
        val = sub.receive()
        return val if val is not None else default

    def draw_measurement(self, surface: pygame.Surface):
        # Transparent overlay
        overlay = pygame.Surface((310, surface.get_height()), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 100), overlay.get_rect())
        surface.blit(overlay, (0, 0))

        # Read values directly from subscribers
        server_fps = self._read(self.sub_server_fps, 0)
        client_fps = self._read(self.sub_client_fps, 0)
        vehicle_name = self._read(self.sub_vehicle_name, "")
        world_name   = self._read(self.sub_world_name, "")
        velocity     = self._read(self.sub_velocity, 0.0)
        heading      = self._read(self.sub_heading, 0.0)
        accel        = self._read(self.sub_accel, "N/A")
        gyro         = self._read(self.sub_gyro, "N/A")
        enu          = self._read(self.sub_enu, "N/A")
        geo          = self._read(self.sub_geo, "N/A")
        client_runtime = self._read(self.sub_client_runtime, 0.0)
        server_runtime = self._read(self.sub_server_runtime, 0.0)

        accel_str = accel if isinstance(accel, str) else f"( {accel[0]: 6.2f}, {accel[1]: 6.2f}, {accel[2]: 6.2f} )"
        gyro_str  = gyro  if isinstance(gyro, str)  else f"( {gyro[0]: 6.2f}, {gyro[1]: 6.2f}, {gyro[2]: 6.2f} )"
        geo_str   = geo   if isinstance(geo, str)   else f"( {geo[0]: 6.6f}, {geo[1]: 6.6f} )"

        client_time_str = self._time_to_str(client_runtime)
        server_time_str = self._time_to_str(server_runtime)

        if isinstance(enu, str):
            h_str, loc_str = "N/A", "N/A"
        else:
            h_str   = f"{enu[2]: 6.2f} m"
            loc_str = f"( {enu[0]: 6.2f}, {enu[1]: 6.2f} )"

        # Collect lines
        value_lines = [
            ("Server side:",   f"{int(server_fps)} FPS", 0),
            ("Client side:",   f"{int(client_fps)} FPS", 1),
            ("Client runtime:", f"{client_time_str} s", 2),
            ("Server runtime:", f"{server_time_str} s", 3),
            ("Vehicle name:",   vehicle_name, 5),
            ("World name:",     world_name,   6),
            ("Velocity:",       f"{velocity:.2f} (km/h)", 8),
            ("Heading:",        f"{heading:.1f}° {self.heading_to_cardinal(heading)}", 9),
            ("Acceleration:",   accel_str, 10),
            ("Gyroscope:",      gyro_str, 11),
            ("Location:",       loc_str, 12),
            ("Geodetic:",       geo_str, 13),
            ("Height:",         h_str, 14),
        ]

        # Alignment
        self.max_string = max(max(len(v) for _, v, _ in value_lines), 15)

        for label, value, idx in value_lines:
            self._render_line(surface, label, value, idx)

    def draw_controls(self, surface, x=10, y=330):
        line_h = 20
        bar_w, bar_h = 150, 10
        bar_x = x + 100

        white = (255, 255, 255)
        green = (0, 200, 0)
        red   = (200, 0, 0)

        # Read controls directly
        throttle = self._read(self.sub_throttle_logging, 0.0)
        steer    = self._read(self.sub_steer_logging, 0.0)
        brake    = self._read(self.sub_brake_logging, 0.0)
        reverse  = self._read(self.sub_reverse_logging, False)
        handbrake= self._read(self.sub_handbrake_logging, False)
        manual   = self._read(self.sub_manual_logging, False)
        gear     = self._read(self.sub_gear_logging, 0)
        autopilot= self._read(self.sub_autopilot_logging, False)
        model_autopilot = self._read(self.sub_model_autopilot_logging, False)
        regulate = self._read(self.sub_regulate_speed_logging, False)

        # Bars
        surface.blit(self.font.render("Throttle:", True, white), (x, y))
        pygame.draw.rect(surface, white, (bar_x, y+5, bar_w, bar_h), 1)
        pygame.draw.rect(surface, green, (bar_x, y+5, int(bar_w * min(throttle,1.0)), bar_h))

        surface.blit(self.font.render("Steer:", True, white), (x, y+line_h))
        pygame.draw.rect(surface, white, (bar_x, y+line_h+5, bar_w, bar_h), 1)
        if steer >= 0:
            pygame.draw.rect(surface, green, (bar_x + bar_w//2, y+line_h+5,
                                              int((bar_w//2) * min(steer,1.0)), bar_h))
        else:
            pygame.draw.rect(surface, green, (bar_x + bar_w//2 + int((bar_w//2)*steer), y+line_h+5,
                                              int(-(bar_w//2) * steer), bar_h))

        surface.blit(self.font.render("Brake:", True, white), (x, y+2*line_h))
        pygame.draw.rect(surface, white, (bar_x, y+2*line_h+5, bar_w, bar_h), 1)
        pygame.draw.rect(surface, red, (bar_x, y+2*line_h+5, int(bar_w * min(brake,1.0)), bar_h))

        # Others as text
        spacing = 33
        surface.blit(self.font.render(f"{'Throttle:':<{spacing}} {'■' if reverse else '□'}", True, white), (x, y+3*line_h))
        surface.blit(self.font.render(f"{'Hand brake:':<{spacing}} {'■' if handbrake else '□'}", True, white), (x, y+4*line_h))
        surface.blit(self.font.render(f"{'Manual:':<{spacing}} {'■' if manual else '□'}", True, white), (x, y+5*line_h))
        surface.blit(self.font.render(f"{'Gear:':<{spacing}} {gear}", True, white), (x, y+6*line_h))
        surface.blit(self.font.render(f"{'Autopilot:':<{spacing}} {'■' if autopilot else '□'}", True, white), (x, y+7*line_h))
        surface.blit(self.font.render(f"{'Model autopilot:':<{spacing}} {'■' if model_autopilot else '□'}", True, white), (x, y+8*line_h))
        surface.blit(self.font.render(f"{'Regulate speed:':<{spacing}} {'■' if regulate else '□'}", True, white), (x, y+9*line_h))

    def draw_logging(self, surface, x=10, y=510):
        turn = self._read(self.sub_turn_signal, -1)
        if turn == -1:
            direction_str = "Keep lane"
        elif turn == 0:
            direction_str = "Go straight"
        elif turn == 1:
            direction_str = "Turn left"
        elif turn == 2:
            direction_str = "Turn right"
        else:
            direction_str = "N/A"

        line_h = 20
        spacing = 15
        text = self.font.render(f"{'Turn signal:':<{spacing}}{direction_str:>{self.max_string}}", True, (255, 255, 255))
        surface.blit(text, (x, y + 1 * line_h))
