import os, sys
import toml
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import pygame
import numpy as np
import time
import gc


from utils.control.world import World
from utils.control.controller import Controller
from utils.control.sensor_manager import SensorManager
from utils.render.hud import HUD, draw_border, overlay_waypoints_on_map, overlay_gmm_on_map
from utils.control.vehicle_control import Vehicle
from utils.control.pid import lateral_control, longitudinal_speed

from utils.spawn.sensor_spawner import *
from config.enum import (
    CameraView,
    JOYBINDS, 
    KEYBINDS, 
)
from utils.messages.message_handler import (
    MessageSubscriber,
    MessageSender
)
from utils.messages.all_messages import (
    PolylinesCmd,
    ServerFps,
    ClientFps,
    VehicleName,
    WorldName,
    Velocity,
    Heading,
    Accel,
    Gyro,
    Enu,
    Geo,
    ClientRuntime,
    ServerRuntime,
    Location,
    TurnSignal,
    ModelSteer,
    ModelSpeed,
    ModelAutopilot,
    ThrottleLog,
    SteerLog,
    BrakeLog,
    ReverseLog,
    HandbrakeLog,
    ManualLog,
    GearLog,
    AutopilotLog,
    RegulateSpeedLog,
)
from utils.messages.logger import Logger

from typing import Optional
from typing import Union, Dict
from tqdm.auto import tqdm
from collections import deque

from model.inference import AsyncInference
from utils.others.data_processor import TrajectoryBuffer
from utils.math.world_map import Map
from utils.math.path import ReplayHandler, OptimizePath
from utils.math.path import PathHandler

        
conf = toml.load(os.path.join(parent, "../config/config.toml"))


path_conf = conf.get("RandPath", {})
MIN_DISTANT_NODE = path_conf.get("min_distant_node", 20)
MAX_DISTANT_NODE = path_conf.get("max_distant_node", 500)
PATH_ITER        = path_conf.get("path_iter", 5)

ui_conf = conf.get("UI", {})
FONT_SIZE = ui_conf.get("font_size", 12)
FONT_NAME = ui_conf.get("font_name", "arial")

offset_conf = conf.get("Offsets", {})
front_vehicle_offset = conf.get("front_vehicle_offset", 1.5)


class CarlaViewer(object):
    override_render_map = False

    def __init__(self, world: World, vehicle: Vehicle, width: int, height: int, headless = False, sync: bool = False, fps: int = 70, duration: tuple = None):
        self.log = Logger() 

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
        self.duration = duration

        self.avg_server_fps = 0.0

        self.virt_vehicle = vehicle
        self.vehicle      = vehicle.vehicle

        self.vehicle_name = vehicle.literal_name()
        self.world_name   = self.world.get_map().name.split("/")[-1]

        self.display: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.world_clock: Optional[pygame.time.Clock] = None

        if self.headless and not pygame.get_init():
            pygame.init()

        self.init_win()

        self.running = False
        self.rgb_sensor = None  
        self.sensor_manager = SensorManager(logger=self.log)
        
        self.controller = Controller()
        self.hud = HUD(display = self.display, fontName = "jetbrainsmononerdfontpropo", fontSize = FONT_SIZE, height = self.height, headless = headless)
        
        
        self.traj_logger   : TrajectoryBuffer = None
        self.replayer      : ReplayHandler    = None
        self.inference     : AsyncInference   = None
        self.pbar          : tqdm             = None
        self.map_processor : Map              = None
        self.path_optimizer: OptimizePath     = None 
        
        # Initialize all message senders and subscribers
        self._init_transmittor()
        
    def _init_transmittor(self):
        """Initialize all message senders and subscribers eagerly"""
        # Senders for telemetry data
        self.send_server_fps = MessageSender(ServerFps)
        self.send_client_fps = MessageSender(ClientFps)
        self.send_vehicle_name = MessageSender(VehicleName)
        self.send_world_name = MessageSender(WorldName)
        self.send_velocity = MessageSender(Velocity)
        self.send_heading = MessageSender(Heading)
        self.send_accel = MessageSender(Accel)
        self.send_gyro = MessageSender(Gyro)
        self.send_enu = MessageSender(Enu)
        self.send_geo = MessageSender(Geo)
        self.send_client_runtime = MessageSender(ClientRuntime)
        self.send_server_runtime = MessageSender(ServerRuntime)
        self.send_location = MessageSender(Location)
        
        # Senders for control logging
        self.send_model_autopilot_logging = MessageSender(ModelAutopilot)
        self.send_autopilot_logging = MessageSender(AutopilotLog)
        self.send_regulate_speed_logging = MessageSender(RegulateSpeedLog)
        self.send_throttle_logging = MessageSender(ThrottleLog)
        self.send_steer_logging = MessageSender(SteerLog)
        self.send_brake_logging = MessageSender(BrakeLog)
        self.send_reverse_logging = MessageSender(ReverseLog)
        self.send_handbrake_logging = MessageSender(HandbrakeLog)
        self.send_manual_logging = MessageSender(ManualLog)
        self.send_gear_logging = MessageSender(GearLog)
        
        # Senders for model predictions
        self.send_model_steer = MessageSender(ModelSteer)
        self.send_model_speed = MessageSender(ModelSpeed)
        
        # Subscribers
        self.sub_location = MessageSubscriber(Location)
        self.sub_heading = MessageSubscriber(Heading)
        self.sub_turn_signal = MessageSubscriber(TurnSignal)
        self.sub_server_runtime = MessageSubscriber(ServerRuntime)
        self.sub_polylines = MessageSubscriber(PolylinesCmd)

    def attach_plugins(self, **plugins):
        for name, value in plugins.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                self.log.ERROR(f"Not able to use plugin {name}. It is not a predefined attribute")
        
    def init_sensor(self, sensors_metadata: dict):
        """Lazy initialize sensors"""
        for sensor, transform in sensors_metadata.items():
            sensor_name = sensor.name.split(".")[-1]
            sensor_type = sensor.name.split(".")[1]
            
            # If sensor is a camera, set default image size
            if sensor_type == 'camera':
                sensor.set_attribute("image_size_x", self.width)
                sensor.set_attribute("image_size_y", self.height)
                if transform is None:
                    transform = CameraView.FIRST_PERSON.value  # default camera transform
            else:
                if transform is None:
                    transform = {}  # default for non-camera sensors

            # Spawn sensor with transform
            sensor.spawn(attach_to=self.vehicle, **transform)
            
            # Register with sensor manager
            self.sensor_manager.register_sensor(sensor_name, sensor, sensor_type)

        # Log sensor summary and set initial choosen_sensor
        if self.sensor_manager.camera_keys:
            self.log.INFO(f"Defaulting to {self.sensor_manager.active_camera}")
            self.choosen_sensor = self.sensor_manager.get_sensor(self.sensor_manager.active_camera)

    def switch_camera(self, step=1):
        """Switch between camera sensors"""
        new_camera = self.sensor_manager.switch_camera(step)
        if new_camera:
            sensor = self.sensor_manager.get_sensor(new_camera)
            if sensor:
                self.choosen_sensor = sensor

    # ============ BACKWARD COMPATIBILITY PROPERTIES ============
    @property
    def sensors_list(self):
        """Backward compatibility: access sensors_list"""
        return self.sensor_manager.sensors_list

    @property
    def camera_keys(self):
        """Backward compatibility: access camera_keys"""
        return self.sensor_manager.camera_keys

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
        self.world_clock = pygame.time.Clock()

    def step_world(self) -> None:
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
            
    def change_view_all(self, view_name: str):
        for camera_name in self.camera_keys:
            self.sensors_list[camera_name].change_view(**getattr(CameraView, view_name).value)
    
    # WARNING: REMEMBER TO CHECK COLOR CHANNEL
    def run(self) -> None:

        try:
            self.virt_vehicle.set_autopilot(self.controller.autopilot) # First init for autopilot
            
            if self.replayer is None:
                random_path = self.generate_randpath()
                midlane_wp = self.map_processor.precompute_waypoints(random_path)

            frame_id = 0
            while True if self.headless else self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.data_bus(self.replayer != None)
                
                if not self.headless:
                    frame = self.choosen_sensor.extract_data()


                unrouted_map, old_map = self.map_processor.retrieve_map(
                    display = self.controller.toggle_map or self.replayer is not None or self.override_render_map
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
                    
                if self.traj_logger: # In recording mode
                    vehicle_transform = self.vehicle.get_transform()
                    vehicle_location = vehicle_transform.location
                    vehicle_rotation = vehicle_transform.rotation

                    yaw_rad = np.radians(vehicle_rotation.yaw)

                    front_offset = front_vehicle_offset  # meters
                    offset_x = front_offset * np.cos(yaw_rad)
                    offset_y = front_offset * np.sin(yaw_rad)

                    front_location = np.array([
                        vehicle_location.x + offset_x,
                        vehicle_location.y + offset_y,
                        vehicle_location.z  # same height as center
                    ])
                    self.traj_logger.update(front_location)
                if self.replayer: # in replaying mode

                    if self.replayer.data_collector: 
                        image_kwargs = (
                            {"I0": self.sensor_manager.get_sensor_data("rgb")} |
                            {"Mask": self.sensor_manager.get_sensor_data("semantic_segmentation")} |
                            {"MU": unrouted_map, "MR": routed_map[:, :, ::-1]} 
                        )

                        self.replayer.step(**image_kwargs)
                    else: 
                        self.replayer.step()
                

                if self.inference and self.controller.model_autopilot: # in inference mode
                    
                    if frame_id % 1 == 0:
                        input_metadata = {
                            "I0": frame,
                            "MU": unrouted_map,
                            "MR": routed_map,
                        }
                        if "multi_images_list" in locals(): 
                            for idx, image in enumerate(multi_images_list):
                                input_metadata[f"I{idx + 1}"] = image

                        inp, debug = self.inference.pytorch.preprocessor(**input_metadata) # VENL preprocessor
                        self.inference.put(inp, self.sub_turn_signal.receive())
                        output = self.inference.get()
                        if output is not None:
                            if not isinstance(output, tuple): # the model has no extra information
                                if isinstance(output, (float, np.float32)): # if output is just steering
                                    self.send_model_steer.send(float(output))
                                else: # output is waypoints
                                    ...
                            else: # with extra info
                                output, *extra = output
                                if len(output.shape) == 1:
                                    self.send_model_steer.send(float(output[0]))
                                else:
                                    norm_steer = lateral_control(output, Ld = 10, wheelbase = self.virt_vehicle.wheelbase, max_steer = self.virt_vehicle.max_steer)
                                    speed = longitudinal_speed(output, 6, 0.04)
                                    self.send_model_speed.send(float(speed))
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
                                    # routed_map = overlay_gmm_on_map(
                                    #     map_img = routed_map,
                                    #     weights = np.exp(weights.copy()), 
                                    #     mu      = muys.copy(),
                                    #     sigma   = sigmas.copy(),
                                    #     scale       = self.map_processor.final_scale,   # (px/m_fwd, px/m_lat)
                                    #     alpha       = .1,
                                    #     n_std       = 1.0,
                                    #     swap_axes   = True,         # match your [lat, fwd] convention
                                    #     flip_lat    = False,
                                    #     origin      = "center"
                                    # )
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

                    if self.controller.toggle_map and "routed_map" in locals():
                        submap_h, submap_w, _ = routed_map.shape
                        if old_map is None:
                            routed_map = draw_border(routed_map, border_thicc = 3, border_color = (255, 255, 200, 100))
                        self.hud.draw_frame(routed_map, (self.width - submap_w - 10, 0 + 10))
                # ======================== RENDER ==========================

                if self.replayer is not None and self.duration <= self.sub_server_runtime.receive():
                    self.log.INFO("Reached replay limit. Goodbye.")
                    break
                else:
                    if self.pbar is not None: # Not in data collect mode
                        progress, task_id = self.pbar
                        elapsed = round(self.sub_server_runtime.receive(), 1)
                        progress.update(task_id, completed=min(elapsed, self.duration))


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
            if self.traj_logger:
                self.traj_logger.finalize()
            if self.inference is not None:
                self.inference.stop()
                
    def close(self) -> None:
        
        print()
        self.log.WARNING("Closing CarlaViewer...")

        for name, sensor in list(self.sensors_list.items()):
            sensor.destroy()
        # self.event.set()
        # self.step_world_th.join()

        try:
            if not self.headless and pygame.get_init():
                pygame.quit()
                self.log.CUSTOM("SUCCESS", "Pygame closed successfully!")
            elif self.headless:
                self.log.INFO("Headless mode: keeping pygame alive for next run")
        except Exception as e:
            self.log.ERROR("Pygame quit failed", full_traceback = e)
            

    def data_bus(self, filter_ctrl=False):
        snapshot = self.world.get_snapshot()
        current_platform_time = snapshot.timestamp.platform_timestamp  # server wall clock
        if self.last_platform_time is not None:
            dt_real = current_platform_time - self.last_platform_time
            self.server_fps = 1.0 / dt_real if dt_real > 0 else 0
        self.last_platform_time = current_platform_time

        # Extract sensor data with safe fallback
        imu_data = self.sensor_manager.get_sensor_data('imu')
        try:
            heading = imu_data.Compass * 180 / np.pi if imu_data else "N/A"
        except:
            heading = "N/A"
        try:
            accel = imu_data.Acceleration if imu_data else "N/A"
        except:
            accel = "N/A"
        try:
            gyro = imu_data.Gyroscope if imu_data else "N/A"
        except:
            gyro = "N/A"
        
        gnss_data = self.sensor_manager.get_sensor_data('gnss')
        try:
            enu = gnss_data.ENU if gnss_data else "N/A"
        except:
            enu = "N/A"
        try:
            geo = gnss_data.Geodetic if gnss_data else "N/A"
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
        self.send_enu.send(enu.to_numpy() if hasattr(enu, 'to_numpy') else enu)
        self.send_geo.send(geo.to_numpy() if hasattr(geo, 'to_numpy') else geo)
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
        
    def generate_randpath(self):
        self.log.INFO("CREATING RANDOM PATH FOR MAP")
        prev_loc = np.array([self.vehicle.get_location().x, self.vehicle.get_location().y])
        extended_path = None
        for _ in range(PATH_ITER):
            while True:
                distant_nodes = self.path_optimizer.find_distant_nodes(prev_loc, np.random.randint(MIN_DISTANT_NODE, MAX_DISTANT_NODE), max_distance = MAX_DISTANT_NODE)
                if distant_nodes:
                    rand_node = np.random.randint(0, len(distant_nodes))
                    fartest_id, farthes_distance, farthest_pos = distant_nodes[rand_node]
                    break

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
        return extended_path

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


class InfractionManager:
    """
    A comprehensive manager to detect off-road infractions.
    
    This class tracks two types of events that constitute an "off-road" infraction:
    1.  Hard collisions with static objects (walls, poles, buildings).
    2.  Driving over a curb onto a non-drivable surface (sidewalk, grass).
    
    The final score is reported as a combined rate of these events per kilometer.
    """
    def __init__(self, world: carla.World, vehicle: carla.Actor):
        """
        Initializes the manager and attaches the necessary sensors to the vehicle.
        
        :param world: The CARLA world object.
        :param vehicle: The vehicle actor to monitor.
        """
        self.world = world
        self.vehicle = vehicle
        
        # --- 1. Initialize trackers ---
        self.infraction_count = 0
        self.total_distance = 0.0
        self.last_location = vehicle.get_location()

        # --- 2. Attach sensors and register callbacks ---
        bp_library = world.get_blueprint_library()

        # Collision Sensor for hard impacts
        collision_bp = bp_library.find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)

        # Lane Invasion Sensor for driving over curbs/sidewalks
        lane_bp = bp_library.find('sensor.other.lane_invasion')
        self.lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(self._on_lane_invasion)

        print("OffRoadInfractionManager initialized and sensors attached.")

    def _on_collision(self, event: carla.CollisionEvent):
        """
        Callback executed by CARLA upon a physical collision.
        Counts collisions with static objects only.
        """
        other_actor = event.other_actor
        
        # We only care about hitting static parts of the world.
        if 'vehicle' not in other_actor.type_id and 'pedestrian' not in other_actor.type_id:
            self.infraction_count += 1
            print(f"Off-Road Infraction: Collision with static object ({other_actor.type_id}).")

    def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
        """
        Callback executed by CARLA upon crossing a lane marking.
        Counts infractions for driving over curbs onto sidewalks.
        """
        for marking in event.crossed_lane_markings:
            # A 'Curb' marking separates a drivable lane from a non-drivable sidewalk/grass area.
            if marking.type == carla.LaneMarkingType.Curb:
                self.infraction_count += 1
                print("Off-Road Infraction: Drove over a curb.")

    def tick(self):
        """
        This function must be called at every simulation step to track distance.
        """
        current_location = self.vehicle.get_location()
        # Add distance traveled since the last tick (in meters)
        self.total_distance += current_location.distance(self.last_location)
        self.last_location = current_location

    def get_infraction_rate(self) -> float:
        """
        Calculates the final normalized score.
        
        :return: The total number of off-road infractions per 1000 kilometers.
        """
        distance_km = self.total_distance / 1000.0
        
        if distance_km == 0:
            return 0.0  # Avoid division by zero on very short or stationary runs

        # Normalize the total infraction count by the distance
        rate = self.infraction_count / distance_km
        return rate
        
    def destroy_sensors(self):
        """
        Cleans up and destroys the attached sensors. Call this at the end of an evaluation.
        """
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        if self.lane_sensor and self.lane_sensor.is_alive:
            self.lane_sensor.destroy()
        print("Infraction manager sensors destroyed.")