import pygame
import numpy as np
import time
import cv2
import gc


from utils.control.world import World
from utils.control.controller import Controller
from utils.render.hud import HUD, draw_border, overlay_waypoints_on_map, overlay_gmm_on_map
from utils.control.vehicle_control import Vehicle

from utils.math.path import ReplayHandler, OptimizePath
from utils.math.world_map import Map
from utils.control.pid import lateral_control
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
from collections import deque
        


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

        self.init_win()

        self.running = False
        self.rgb_sensor = None  
        self.sensors_list: Dict[str, Union[RGB, Depth, SemanticSegmentation, GNSS, IMU, LidarRaycast]] = {}
        self.camera_keys = []
        
        self.controller = Controller()
        self.hud = HUD(display = self.display, fontName = "jetbrainsmononerdfontpropo", fontSize = 12, height = self.height, headless = headless)
        
        self.map_processor  = Map(self.virt_world, (4, 3), map_offset = (100, 100), range_ = (50, 50), resize_to = (200, 200), scale = 3)
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
        
        
    # WARNING: REMEMBER TO CHECK COLOR CHANNEL
    def run(self, 
            model_path = None,
            save_logging: str = None, 
            use_temporal_wp: bool = False, 
            data_collect_dir: str = None, 
            replay_logging: str = None, 
            debug = False) -> None:

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
                prev_loc = np.array([self.vehicle.get_location().x, self.vehicle.get_location().y])
                extended_path = None
                for _ in range(5):
                    distant_nodes = self.path_optimizer.find_distant_nodes(prev_loc, np.random.randint(20, 200), max_distance = 200)
                    if distant_nodes:
                        rand_node = np.random.randint(0, len(distant_nodes))
                        fartest_id, farthes_distance, farthest_pos = distant_nodes[rand_node]

                    farthest_pos = np.array(list(farthest_pos))
                    nodes, path_coor = self.path_optimizer.plan_path(
                        prev_loc, 
                        farthest_pos
                    )
                    path_coor = np.hstack([path_coor, np.zeros((path_coor.shape[0], 1))])
                    if extended_path is None:
                        extended_path = path_coor
                    else:
                        extended_path = np.vstack([extended_path, path_coor])
                    prev_loc = farthest_pos
                self.map_processor.precompute_waypoints(extended_path)

            logger    = TrajectoryBuffer(save_logging, min_dt_s = .4) if save_logging else None
            replayer  = ReplayHandler(replay_logging[0], self.virt_world, data_collect_dir, use_temporal_wp, midlane_waypoints, debug) if replay_logging else None
            inference = AsyncInference(model_path, device = 'cuda', batch_output = False) if model_path is not None else None
            pbar      = tqdm(total = replay_logging[1], unit = 'server second', desc = "Play duration") if replay_logging else None
            gps_buffer = deque(maxlen = 50)
            # ================== INITIALIZING CLASSES =====================

            frame_id = 0
            while True if self.headless else self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.data_bus(replay_logging != None)
                
                frame = self.choosen_sensor.extract_data()
                try:
                    H, W, _ = frame.shape
                except:
                    H, W = frame.shape



                location = self.sub_location.receive()
                heading  = self.sub_heading.receive()
                gps_buffer.append(location) # THIS DELAY ALSO CONFIRMS THAT THE MODEL IS TOO DEPENDANT ON ROUTED MAP
                choose_delay = np.random.randint(0, min(10, len(gps_buffer)))
                unrouted_map, old_map = self.map_processor.retrieve_map(
                    # coordinate = location,
                    coordinate = gps_buffer[choose_delay] + (np.random.randn(3) - .5) * .1,  # Introduce some noise to the GPS map 
                    heading = heading, 
                    display = self.controller.toggle_map or replayer is not None or self.override_render_map
                )
                if frame_id % 1 == 0:
                    if old_map is not None:
                        routed_map = old_map
                elif old_map is None:
                    routed_map = np.zeros([*self.map_processor.resize_to, 3])

                if "multi" in "".join(list(self.sensors_list.keys())):
                    self.log.INFO("Multi camera sensor setup detected. Displaying it", once = True)
                    sensor_tname = self.choosen_sensor.name.split(".")[-1]
                    multi_tname  = "multi" + sensor_tname
                    
                    if multi_tname in "".join(list(self.sensors_list.keys())):
                        data = self.sensors_list[multi_tname].extract_data()
                        multi_images_list = []
                        for images in data.values():
                            multi_images_list += [images]

                    
                    
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
                    vehicle_transform = self.vehicle.get_transform()
                    vehicle_location = vehicle_transform.location
                    vehicle_rotation = vehicle_transform.rotation

                    # Convert yaw from degrees to radians for math functions
                    yaw_rad = np.radians(vehicle_rotation.yaw)

                    # Distance from the center to the front of the car (adjust as per your vehicle)
                    front_offset = 1.5  # meters

                    # Calculate offset in x and y directions
                    offset_x = front_offset * np.cos(yaw_rad)
                    offset_y = front_offset * np.sin(yaw_rad)

                    # Calculate front location coordinates
                    front_location = np.array([
                        vehicle_location.x + offset_x,
                        vehicle_location.y + offset_y,
                        vehicle_location.z  # same height as center
                    ])
                    logger.update(front_location)
                if replayer: # in replaying mode

                    # =========================== VENL ==========================
                    # multi_images_list = list(data.values())   # e.g. [img1, img2, img3]
                    # multi_keys = [f"I{i+1}" for i in range(len(multi_images_list))]
                    # H, W, _    = frame.shape
                    # x_top_left = 250; x_top_right = W - x_top_left
                    # y_hor      = 390; y_bot         = 680
                    # frame_cutout = frame[y_hor: y_bot, x_top_left: x_top_right]
                    # frame_cutout = cv2.resize(frame_cutout, (multi_images_list[0].shape[1], multi_images_list[0].shape[0]))
                    # image_kwargs = (
                    #     {"I0": frame_cutout} | 
                    #     {k: v for k, v in zip(multi_keys, multi_images_list)} | 
                    #     {"MU": cv2.resize(unrouted_map, (50, 50)), "MR": cv2.resize(routed_map, (50, 50))}
                    # )
                    
                    # =========================== Single VENL ==========================
                    H, W, _    = frame.shape
                    x_top_left = 150; x_top_right = W - x_top_left
                    y_hor      = 370; y_bot       = 720
                    frame_cutout = frame[y_hor: y_bot, x_top_left: x_top_right]
                    frame_cutout = cv2.resize(frame_cutout, (400, 160))
                    image_kwargs = (
                        {"I0": frame_cutout} | 
                        {"MU": cv2.resize(unrouted_map, (50, 50)), "MR": cv2.resize(routed_map, (50, 50))}
                    )

                    replayer.step(**image_kwargs)
                

                if model_path and self.controller.model_autopilot: # in inference mode
                    
                    if frame_id % 1 == 0:
                         
                        # CONCLUSION: MODEL IS HEAVILY RELYING ON ROUTED MAP, THE DATA IS TOO CLEAN
                        frame_inp    = frame
                        # I1_inp       = multi_images_list[0]
                        # I2_inp       = multi_images_list[1]
                        unrouted_inp = unrouted_map
                        routed_inp   = routed_map

                        # frame_inp    = np.zeros_like(frame)
                        # I1_inp       = np.zeros_like(multi_images_list[0])
                        # I2_inp       = np.zeros_like(multi_images_list[1])
                        # unrouted_inp = np.zeros_like(unrouted_map)
                        # routed_inp   = np.zeros_like(routed_map)

                        # inp = inference.pytorch.preprocessor(**{"I0": frame}) # PilotNet preprocessor
                        # inp = inference.pytorch.preprocessor(I0 = frame_inp, I1 = I1_inp, I2 = I2_inp, MU = unrouted_inp, MR = routed_inp) # VENL preprocessor
                        inp = inference.pytorch.preprocessor(I0 = frame_inp, MU = unrouted_inp, MR = routed_inp) # SingleVENL preprocessor
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
                                    norm_steer = lateral_control(output, Ld = 8, wheelbase = self.virt_vehicle.wheelbase, max_steer = self.virt_vehicle.max_steer)
                                    self.send_model_steer.send(norm_steer)

                        
                
                # ======================== RENDER ==========================
                if not self.headless:
                    self.hud.draw_frame(frame)
                    self.hud.draw_measurement()
                    self.hud.draw_controls()
                    self.hud.draw_logging()

                    if "multi_images_list" in locals(): # If there exist multi camera
                        multi_images = np.hstack([draw_border(image, 3, (255, 100, 200, 255)) for image in multi_images_list])
                        multi_h, multi_w, _ = multi_images.shape
                        self.hud.draw_frame(multi_images, (self.width - multi_w - 10, self.height - multi_h - 10))


                    if "output" in locals():
                        if output is not None and self.controller.model_autopilot:
                            try:
                                if 'extra' in locals() and len(extra) == 3:
                                    weights, muys, sigmas = extra
                                    routed_map = overlay_gmm_on_map(
                                        map_img = routed_map,
                                        weights = weights.copy(), 
                                        mu      = muys.copy(),
                                        sigma   = sigmas.copy(),
                                        scale       = self.map_processor.final_scale,   # (px/m_fwd, px/m_lat)
                                        alpha       = 2.5,
                                        n_std       = 1.0,
                                        swap_axes   = True,         # match your [lat, fwd] convention
                                        flip_lat    = False,
                                        origin      = "center"
                                    )
                            finally: 
                                ...

                            routed_map = overlay_waypoints_on_map(
                                map_img = routed_map,
                                waypoints = output.copy(), 
                                scale = self.map_processor.final_scale, 
                                meters_span = self.map_processor.original_range,
                                color=(0, 255, 0),
                                thickness=2,
                                swap_axes=True,  # set True if your waypoints are [lat, fwd]
                                flip_lat=False,    # set True if lateral axis is mirrored
                                origin = "center"
                            )
                            ...
                            # wp_canvas = draw_waypoints_canvas(output, canvas_size = (100, 100), scale = self.map_processor.scale)
                            # wp_canvas_h, wp_canvas_w, _ = wp_canvas.shape
                            # self.hud.draw_frame(wp_canvas, (10, self.height - wp_canvas_h - 10))

                    if self.controller.toggle_map and "routed_map" in locals():
                        submap_h, submap_w, _ = routed_map.shape
                        if old_map is None:
                            routed_map = draw_border(routed_map, border_thicc = 3, border_color = (255, 255, 200, 100))
                        self.hud.draw_frame(routed_map, (self.width - submap_w - 10, 0 + 10))
                # ======================== RENDER ==========================

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
                        self.clock.tick(self.fps)

                frame_id += 1


        except KeyboardInterrupt:
            self.log.WARNING("Viewer interrupted by user")
            self.controller.running = False
        except Exception as e:
            self.log.ERROR("Viewer error", full_traceback = e)
            self.controller.running = False
        finally:
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