import pygame
import numpy as np
import time
import cv2
import gc


from utils.control.world import World
from utils.control.controller import Controller
from utils.control.displayer import HUD
from utils.control.vehicle_control import Vehicle

from utils.math.path import ReplayHandler, OptimizePath
from utils.math.world_map import Map
from utils.math.pid import lateral_control
from utils.math.coordinate_transform import local_2_global

from utils.spawn.sensor_spawner import *
from utils.others.data_processor import TrajectoryBuffer
from model.inference import AsyncInference
from config.enum import (
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
from tqdm.auto import tqdm
        

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
    override_render_map = False


    def __init__(self, world: World, vehicle: Vehicle, width: int, height: int, headless = False, sync: bool = False, fps: int = 70):
        self.log = Logger() 
        MessagingSenders.__init__(self)
        MessagingSubscribers.__init__(self)

        if headless:
            self.log.WARNING("HEADLESS MODE [bold][red][u]ENABLED[/][/][/]")
        self.headless = headless
        self.virt_world = world
        self.world = world.world
        self.width = width
        self.height = height
        self.sync = sync
        self.fps = fps if not self.headless else 100
        self.last_platform_time = None; 
        self.server_fps = self.fps; 
        self.client_start = time.time()
        self.server_start = self.world.get_snapshot().timestamp.elapsed_seconds

        self.avg_server_fps = 0.0
        self.alpha = 0.05  # smoothing factor, adjust as needed

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
        self.hud = HUD("jetbrainsmononerdfontpropo", fontSize = 12, height = self.height)
        
        self.map_processor  = Map(self.virt_world, (6, 4), map_offset = (100, 100), range_ = (50, 50), resize_to = (200, 200), scale = 3)
        self.path_optimizer = OptimizePath(self.virt_world, step = 2.0) 
    
        
    def init_sensor(self, sensors_metadata: dict):
        """Lazy initialize sensors"""
        for sensor, transform in sensors_metadata.items():
            # If sensor is a camera, set default image size
            if sensor.name.split(".")[1] == 'camera':
                sensor.set_attribute("image_size_x", self.width)
                sensor.set_attribute("image_size_y", self.height)
                self.camera_keys.append(sensor.name.split(".")[-1])
                if transform is None:
                    transform = CameraView.FIRST_PERSON.value  # default camera transform
            else:
                if transform is None:
                    transform = {}  # default for non-camera sensors

            # Spawn sensor with transform
            sensor.spawn(attach_to=self.vehicle, **transform)
            self.sensors_list[sensor.name.split(".")[-1]] = sensor

        # Set first camera as active
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
        if self.headless:
            # # headless mode → fake clock for FPS timing
            class DummyClock:
                def __init__(self, fps) : self._fps = fps
                def tick(self, fps=None): time.sleep(1.0 / (fps or self._fps))
                def get_fps(self)       : return self._fps
            self.clock = DummyClock(self.fps)
            return

        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()


    def to_surface(self, frame: np.ndarray) -> pygame.Surface:
        if self.headless: return
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

    def draw_frame(self, frame: np.ndarray, position = (0, 0)) -> None:
        if self.headless: return
        surface = self.to_surface(frame)
        self.display.blit(surface, position)

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
        
    def draw_border(self, frame, border_thicc: int, border_color: tuple):
        frame = cv2.copyMakeBorder(
            frame,
            top=border_thicc, bottom=border_thicc,
            left=border_thicc, right=border_thicc,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )
        return frame
        
    # WARNING: REMEMBER TO CHECK COLOR CHANNEL
    def run(self, 
            model_path = None,
            save_logging: str = None, 
            use_temporal_wp: bool = False, 
            data_collect_dir: str = None, 
            replay_logging: str = None, 
            debug = False) -> None:
        if self.display is None:
            self.init_win()

        try:
            self.virt_vehicle.set_autopilot(self.controller.autopilot) # First init for autopilot
            
            # ================== INITIALIZING CLASSES =====================
            midlane_waypoints = None
            if replay_logging is not None:
                self.log.INFO("REPLAYING PATH FOR MAP")
                trajectories = np.load(replay_logging[0])
                midlane_waypoints = self.map_processor.precompute_waypoints(trajectories)
            else:
                self.log.INFO("CREATING RANDOM PATH FOR MAP")
                self.prev_loc = self.vehicle.get_location()
                distant_nodes = self.path_optimizer.find_distant_nodes(np.array([self.prev_loc.x, self.prev_loc.y]), 200)
                if distant_nodes:
                    rand_node = np.random.randint(0, len(distant_nodes))
                    fartest_id, farthes_distance, farthest_pos = distant_nodes[rand_node]

                nodes, path_coor = self.path_optimizer.plan_path(
                    np.array([self.prev_loc.x, self.prev_loc.y]), 
                    np.array(list(farthest_pos))
                )
                path_coor = np.hstack([path_coor, np.zeros((path_coor.shape[0], 1))])
                self.map_processor.precompute_waypoints(path_coor)

            logger    = TrajectoryBuffer(save_logging, min_dt_s = .2) if save_logging else None
            replayer  = ReplayHandler(replay_logging[0], self.virt_world, data_collect_dir, use_temporal_wp, midlane_waypoints, debug) if replay_logging else None
            inference = AsyncInference(model_path, device = 'cuda', batch_output = False) if model_path is not None else None
            pbar      = tqdm(total = replay_logging[1], unit = 'server second', desc = "Play duration") if replay_logging else None
            # ================== INITIALIZING CLASSES =====================

            H, W, _    = 720, 1280, 3
            x_top_left = 250; x_top_right = W - x_top_left
            x_bot_left = 20; x_bot_right = W - x_bot_left
            y_hor      = 300; y_bot         = 680
            frame_id = 0
            while True if self.headless else self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.data_bus(replay_logging != None)
                
                frame = self.choosen_sensor.extract_data()
                try:
                    H, W, _ = frame.shape
                except:
                    H, W = frame.shape


                if frame is not None:
                    self.draw_frame(frame)
                    if not self.headless:
                        self.hud.draw_measurement(self.display)
                        self.hud.draw_controls(self.display)
                        self.hud.draw_logging(self.display)


                    location = self.sub_location.receive()
                    heading  = self.sub_heading.receive()
                    unrouted_map, routed_map = self.map_processor.retrieve_map(
                        coordinate = location, 
                        heading = heading, 
                        display = self.controller.toggle_map or replayer is not None or self.override_render_map
                    )
                    if self.controller.toggle_map:
                        submap_h, submap_w, _ = routed_map.shape
                        self.draw_frame(routed_map, (self.width - submap_w - 10, 0 + 10))
                    
                    if "multi" in "".join(list(self.sensors_list.keys())):
                        self.log.INFO("Multi camera sensor setup detected. Displaying it", once = True)
                        sensor_tname = self.choosen_sensor.name.split(".")[-1]
                        multi_tname  = "multi" + sensor_tname
                        
                        if multi_tname in "".join(list(self.sensors_list.keys())):
                            data = self.sensors_list[multi_tname].extract_data()
                            multi_images_list = []
                            for images in data.values():
                                multi_images_list += [images]
                            multi_images = np.hstack(multi_images_list)
                            multi_images_border = self.draw_border(multi_images, 3, (255, 100, 200, 255))                
                            multi_h, multi_w, _ = multi_images_border.shape
                            self.draw_frame(multi_images_border, (self.width - multi_w - 10, self.height - multi_h - 10))

                    
                    
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
                    
                if logger: # In recording mode
                    logger.update(self.sub_location.receive())
                if replayer: # in replaying mode
                    multi_images_list = list(data.values())   # e.g. [img1, img2, img3]
                    multi_keys = [f"I{i+1}" for i in range(len(multi_images_list))]

                    frame_cutout = frame[y_hor: y_bot, x_top_left: x_top_right]
                    frame_cutout = cv2.resize(frame_cutout, (multi_images_list[0].shape[1], multi_images_list[0].shape[0]))
                    image_kwargs = (
                        {"I0": frame_cutout} | 
                        {k: v for k, v in zip(multi_keys, multi_images_list)} | 
                        {"MU": cv2.resize(unrouted_map, (50, 50)), "MR": cv2.resize(routed_map, (50, 50))}
                    )

                    replayer.step(**image_kwargs)


                if model_path and self.controller.model_autopilot: # in inference mode
                    
                    if frame_id % 1 == 0:
                        # inp = inference.pytorch.preprocessor(**{"I0": frame})
                        inp = inference.pytorch.preprocessor(I0 = frame, I1 = multi_images_list[0], I2 = multi_images_list[1], MU = unrouted_map, MR = routed_map)
                        inference.put(inp, None)
                        output = inference.get()
                        if output is not None:
                            if not isinstance(output, tuple): # the model has no extra information
                                if isinstance(output, float): # if output is just steering
                                    self.send_model_steer.send(float(output))
                                else: # output is waypoints
                                    ...
                            else: # with extra info
                                output, *extra = output
                                if len(output.shape) == 1:
                                    self.send_model_steer.send(float(output[0]))
                                else:
                                    norm_steer = lateral_control(output, Ld = 5, wheelbase = self.virt_vehicle.wheelbase, max_steer = self.virt_vehicle.max_steer)
                                    self.send_model_steer.send(norm_steer)

                                    wp_canvas = draw_waypoints_canvas(output, canvas_size = (100, 100), scale = 5.0)
                                    wp_canvas_h, wp_canvas_w, _ = wp_canvas.shape
                                    self.draw_frame(wp_canvas, (10, self.height - wp_canvas_h - 10))
                        
                        # preview_h, preview_w, _ = inp.shape
                        # inp_surface = self.to_surface(inp[:, :, ::-1])
                        # self.display.blit(inp_surface, (self.width - preview_w - 10, self.height - preview_h - 10))
                    frame_id += 1
                    
                    # local_wp = infer(model, inp)[0]
                    # local_wp[:, 1] = -local_wp[:, 1]
                    # global_wp = self.virt_vehicle.global_transform(local_wp, np.radians(self.sub_heading.receive()))
                    # # self.virt_world.draw_waypoints(global_wp, duration = 1 * (1 / self.server_fps))

                

                if replay_logging is not None and replay_logging[1] <= self.sub_server_runtime.receive():
                    self.log.INFO("Reached replay limit. Goodbye.")
                    break
                else:
                    if pbar is not None: # Not in data collect mode
                        elapsed = round(self.sub_server_runtime.receive(), 1)                    
                        pbar.n  = min(elapsed, replay_logging[1])
                        pbar.refresh()


                if not self.headless:
                    pygame.display.flip()
                    if self.clock:
                        # if self.server_fps > 0:
                        #     if self.avg_server_fps == 0:
                        #         self.avg_server_fps = self.server_fps
                        #     else:
                        #         self.avg_server_fps = (1 - self.alpha) * self.avg_server_fps + self.alpha * self.server_fps
                        # frame_time = self.clock.tick(0)
                        # target = min(self.fps, int(self.avg_server_fps)) if self.avg_server_fps > 0 else self.fps
                        # if target > 0:
                        #     target_dt = 1000.0 / target
                        #     sleep_time = max(0, target_dt - frame_time)
                        #     if sleep_time > 0:
                        #         time.sleep(sleep_time / 1000.0)
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
            if inference is not None:
                inference.stop()

    def close(self) -> None:
        
        print()
        self.log.WARNING("Closing CarlaViewer...")

        for name, sensor in list(self.sensors_list.items()):
            sensor.destroy()
        self.virt_world.factory_reset()
        self.world.wait_for_tick()

        try:
            if pygame.get_init():
                pygame.quit()
                self.log.CUSTOM("SUCCESS", "Pygame closed successfully!")
        except Exception as e:
            self.log.ERROR("Pygame quit failed", full_traceback = e)
            
        gc.collect()

def draw_waypoints_canvas(waypoints: np.ndarray, canvas_size=(200, 200), scale=5.0, color=(0, 255, 0)):
    """
    Draw waypoints on a blank canvas.
    
    Args:
        waypoints: Array of waypoints in vehicle-local coordinates (N x 2)
        canvas_size: (width, height) of the canvas
        scale: Scaling factor for waypoint coordinates
        color: Color of waypoint dots (BGR format)
    
    Returns:
        np.ndarray: Canvas image with waypoints drawn
    """
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    
    # Canvas center (vehicle position)
    center_x, center_y = canvas_size[0] // 2, canvas_size[1] // 2
    
    # Draw vehicle position as a red dot
    cv2.circle(canvas, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Draw waypoints
    for wp in waypoints:
        # Convert local coordinates to canvas coordinates
        x = int(center_x + wp[0] * scale)
        y = int(center_y - wp[1] * scale)  # Flip Y axis (canvas Y increases downward)
        
        # Check if point is within canvas bounds
        if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
            cv2.circle(canvas, (x, y), 2, color, -1)
    
    # Draw connecting lines between waypoints
    for i in range(len(waypoints) - 1):
        x1 = int(center_x + waypoints[i][0] * scale)
        y1 = int(center_y - waypoints[i][1] * scale)
        x2 = int(center_x + waypoints[i+1][0] * scale)
        y2 = int(center_y - waypoints[i+1][1] * scale)
        
        # Check bounds for both points
        if (0 <= x1 < canvas_size[0] and 0 <= y1 < canvas_size[1] and
            0 <= x2 < canvas_size[0] and 0 <= y2 < canvas_size[1]):
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
    
    return canvas