import os, sys
import toml
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)

import pygame
import numpy as np
import time
import carla


from src.control import (
    World, Controller, SensorManager
)
from . import HUD, draw_border, overlay_waypoints_on_map, overlay_gmm_on_map
from src.control import Vehicle, lateral_control, longitudinal_speed

from config.enum import (
    CameraView,
    JOYBINDS, 
    KEYBINDS, 
)
from src.messages import (
    MessageSubscriber,
    MessageSender,
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
    SteerAngle, 
    ClearNPCs
)
from src.messages.logger import Logger

from typing import Optional
from typing import Union, Dict
from tqdm.auto import tqdm
from collections import deque
from abc import ABC, abstractmethod

from model.inference import AsyncInference
from src.others.data_processor import TrajectoryBuffer, ReplayHandler
from src.math import Map, OptimizePath

        
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

max_steer = conf["Vehicle"]['Physics']['max_steer']


VIEWER_REGISTRY = {}
def register_mode(name):
    def decorator(func):
        VIEWER_REGISTRY[name] = func
        return func
    return decorator


class Viewer(ABC):
    override_render_map = True

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
        self.send_steer_angle = MessageSender(SteerAngle)
        
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
                self.log.CUSTOM("SUCCESS", f"System was able to attach {name} pluggin.")
            else:
                self.log.ERROR(f"Not able to use plugin {name}. It is not a predefined attribute")
        
    def init_sensor(self, sensors_metadata: dict):
        """Lazy initialize sensors"""
        type_dict = {}
        for sensor, [transform, ego_attach] in sensors_metadata.items():
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

            if sensor_name not in type_dict.keys():
                type_dict.update({sensor_name: 0})
            else: type_dict[sensor_name] += 1
            
            sensor_name += f"_{type_dict[sensor_name]}"
            
            # Spawn sensor with transform
            if ego_attach == True:
                sensor.spawn(attach_to=self.vehicle, **transform)
            else:
                sensor.spawn(**transform)
            
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

    def close(self) -> None:
        
        print()
        self.log.WARNING("Closing CarlaViewer...")

        for name, sensor in list(self.sensors_list.items()):
            sensor.destroy()

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
        imu_data = self.sensor_manager.get_sensor_data('imu_0')
        try:
            heading = imu_data.Compass * 180 / np.pi if imu_data else "N/A"
        except:
            heading = 0.0
        try:
            accel = imu_data.Acceleration if imu_data else "N/A"
        except:
            accel = 0.0
        try:
            gyro = imu_data.Gyroscope if imu_data else "N/A"
        except:
            gyro = "N/A"
        
        gnss_data = self.sensor_manager.get_sensor_data('gnss_0')
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
        self.send_steer_angle.send(self.ctrl['steer'] * max_steer)
        
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
    
    def _setup_simulation(self):
        """Common simulation setup for all viewers"""
        self.virt_vehicle.set_autopilot(self.controller.autopilot)
        
        # # Generate random path if no replayer is active
        # if not hasattr(self, 'replayer') or self.replayer is None:
        #     random_path = self.generate_randpath()
        #     midlane_wp = self.map_processor.precompute_waypoints(random_path)
    
    def _get_map_data(self, frame_id, replay_mode=False):
        """Retrieve and process map data"""
        unrouted_map, old_map = self.map_processor.retrieve_map(
            display = self.controller.toggle_map or replay_mode or self.override_render_map
        )
        
        # Assign routed_map based on old_map availability
        if old_map is not None:
            routed_map = old_map
        else:
            routed_map = np.zeros([*self.map_processor.resize_to, 3])
        
        return unrouted_map, routed_map
    
    def _handle_multi_camera(self):
        """Handle multi-camera setup and return images list"""
        multi_images_list = None
        if "multi" in "".join(list(self.sensors_list.keys())):
            self.log.INFO("Multi camera sensor setup detected. Displaying it", once = True)
            sensor_tname = self.choosen_sensor.name.split(".")[-1]
            multi_tname  = "multi" + sensor_tname
            
            if multi_tname in "".join(list(self.sensors_list.keys())):
                data = self.sensors_list[multi_tname].extract_data()
                multi_images_list = []
                for images in data.values():
                    multi_images_list += [images]
        return multi_images_list
    
    def _handle_controller_inputs(self):
        """Handle view changes and camera switching"""
        if self.controller.view_changed:
            self.change_view_all(self.controller.view_name)
        if self.controller.camera_changed:
            self.switch_camera(self.controller.camera_step)
    
    def _apply_vehicle_controls(self):
        """Apply vehicle control settings"""
        self.virt_vehicle.set_autopilot(self.controller.autopilot)
        self.virt_vehicle.set_model_autopilot(self.controller.model_autopilot)
        if self.controller.autopilot == False:
            self.virt_vehicle.apply_control(self.controller.regulate_speed,
                                            self.controller.has_joystick, 
                                            self.controller.model_autopilot)
    
    def _render_base_hud(self, frame):
        """Render basic HUD elements"""
        if not self.headless:
            self.hud.draw_frame(frame)
            self.hud.draw_measurement()
            self.hud.draw_controls()
            self.hud.draw_logging()
    
    def _render_multi_camera(self, multi_images_list):
        """Render multi-camera display if available"""
        if not self.headless and multi_images_list is not None:
            multi_images = np.hstack([draw_border(image, 3, (255, 100, 200, 255)) for image in multi_images_list])
            multi_h, multi_w, _ = multi_images.shape
            self.hud.draw_frame(multi_images, (self.width - multi_w - 10, self.height - multi_h - 10))
    
    def _render_map(self, routed_map):
        """Render map overlay if toggled"""
        if not self.headless and self.controller.toggle_map and routed_map is not None:
            submap_h, submap_w, _ = routed_map.shape
            routed_map = draw_border(routed_map, border_thicc = 3, border_color = (255, 255, 200, 100))
            self.hud.draw_frame(routed_map, (self.width - submap_w - 10, 0 + 10))
    
    def _finalize_frame(self):
        """Finalize frame rendering"""
        if not self.headless:
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.fps)
    
    @abstractmethod
    def run(self):
        ...


@register_mode("manual")
class ManualViewer(Viewer):
    def run(self):
        """Run manual control mode for CARLA simulation with keyboard/joystick input"""
        try:
            self.virt_vehicle.set_autopilot(self.controller.autopilot)

            frame_id = 0
            while True if self.headless else self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.data_bus(False)  # No control filtering for manual mode
                
                frame = None
                if not self.headless:
                    frame = self.choosen_sensor.extract_data()

                # Get map data
                unrouted_map, routed_map = self._get_map_data(frame_id)

                # Handle common operations
                multi_images_list = self._handle_multi_camera()
                self._handle_controller_inputs()
                self._apply_vehicle_controls()

                # Render (manual mode - no model predictions)
                self._handle_trajectory_logging()
                self._render_base_hud(frame)
                self._render_multi_camera(multi_images_list)
                self._render_map(routed_map)
                self._finalize_frame()

                frame_id += 1

        except KeyboardInterrupt:
            self.log.WARNING("ManualViewer interrupted by user")
            self.controller.running = False
        except Exception as e:
            self.log.ERROR("ManualViewer error", full_traceback = e)
            self.controller.running = False
        finally:
            self.close()
            if self.traj_logger:
                self.traj_logger.finalize()

    def _handle_trajectory_logging(self):
        """Handle trajectory logging for recording mode"""
        if self.traj_logger:
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
            vehicle_rotation = np.array(            vehicle_rotation = np.array([
                vehicle_rotation.roll,
                vehicle_rotation.pitch,
                vehicle_rotation.yaw
            ])[
                vehicle_rotation.roll,
                vehicle_rotation.pitch,
                vehicle_rotation.yaw
            ])
            self.traj_logger.update(front_location, vehicle_location)
    
@register_mode("replay")
class ReplayViewer(Viewer):
    def run(self):
        """Run replay mode for CARLA recordings, data collection, and headless operations"""
        try:
            self.virt_vehicle.set_autopilot(self.controller.autopilot)
            sub_npc_clear = MessageSubscriber(ClearNPCs)
            clear_npcs = sub_npc_clear.receive()

            commands = []
            if clear_npcs:
                commands = self._init_clear_cmds()
            frame_id = 0
            while True if self.headless else self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.data_bus(True)  # Filter control for replay mode
                
                self.virt_world.client.apply_batch_sync(commands)
                
                frame = None
                if not self.headless:
                    frame = self.choosen_sensor.extract_data()

                # Get map data (always display in replay mode)
                unrouted_map, routed_map = self._get_map_data(frame_id, replay_mode=True)

                # Handle common operations
                multi_images_list = self._handle_multi_camera()
                self._handle_controller_inputs()
                self._apply_vehicle_controls()

                # Replay-specific operations
                self._handle_trajectory_logging()
                self._handle_replay_step(unrouted_map, routed_map)

                # Render (replay mode)
                self._render_base_hud(frame)
                self._render_multi_camera(multi_images_list)
                self._render_map(routed_map)
                self._finalize_frame()

                # Check replay duration and update progress
                if self._check_replay_duration():
                    break

                frame_id += 1

        except KeyboardInterrupt:
            self.log.WARNING("ReplayViewer interrupted by user")
            self.controller.running = False
        except Exception as e:
            self.log.ERROR("ReplayViewer error", full_traceback = e)
            self.controller.running = False
        finally:
            self.close()
            if self.traj_logger:
                self.traj_logger.finalize()

    def _update_playback_rate(self):
        """Update sliding window playback rate calculation"""
        elapsed = self.sub_server_runtime.receive()
        current_time = time.time()
        
        # Add current sample to sliding window
        self.playback_rate_window.append((current_time, elapsed))
        
        # Calculate rate only if we have enough samples
        if len(self.playback_rate_window) > 1:
            oldest_time, oldest_elapsed = self.playback_rate_window[0]
            newest_time, newest_elapsed = self.playback_rate_window[-1]
            
            time_delta = newest_time - oldest_time
            elapsed_delta = newest_elapsed - oldest_elapsed
            
            if time_delta > 0:
                self.playback_rate = elapsed_delta / time_delta
            else:
                self.playback_rate = 0.0
        else:
            self.playback_rate = 0.0
    
    def _check_replay_duration(self):
        """Check replay duration and update progress. Returns True if replay should stop"""

        # Sliding window for playback rate calculation (window_size in seconds)
        if not hasattr(self, 'playback_rate') or not hasattr(self, 'playback_rate_window'):
            self.playback_rate_window = deque(maxlen=30)  # ~2-3 seconds at 10-15 FPS
            self.playback_rate = 0.0
        if self.duration and self.duration <= self.sub_server_runtime.receive():
            self.log.INFO("Reached replay limit. Goodbye.")
            return True
        else:
            if self.pbar is not None:  # Progress tracking for replay
                progress, task_id = self.pbar
                elapsed = self.sub_server_runtime.receive()
                
                self._update_playback_rate()
                
                # Update with description showing playback rate
                progress.update(task_id, completed=min(elapsed, self.duration), 
                              description=f"Play duration ({elapsed:.1f}/{self.duration:.1f}s @ {self.playback_rate:.2f}x)")
            return False

    def _handle_replay_step(self, unrouted_map, routed_map):
        """Handle replay functionality step"""
        if self.replayer:
            if self.replayer.data_collector: 
                image_kwargs = (
                    {"I0": self.sensor_manager.get_sensor_data("rgb_0")} 
                )
                self.replayer.step(**image_kwargs)
            else: 
                self.replayer.step()

    def _init_clear_cmds(self, ):
        ego_name = self.virt_vehicle.vehicle.attributes.get('role_name')
        current_vehicles = self.world.get_actors().filter("vehicle.*")
        for actor in current_vehicles:
            if actor.attributes.get("role_name") == ego_name:
                ego_id = actor.id
                self.log.INFO(f"Found ego vehicle with id={ego_id}")
                break
        commands = []
        for actor in current_vehicles:
            if actor.id != ego_id:
                commands.append(carla.command.DestroyActor(actor.id))
        return commands

    def _handle_trajectory_logging(self):
        """Handle trajectory logging for recording mode"""
        if self.traj_logger:
            self.log.WARNING("RE RECORDING TRAJECTORY IN PROGRESS!", once = True)
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
            vehicle_rotation = np.array([
                vehicle_rotation.roll,
                vehicle_rotation.pitch,
                vehicle_rotation.yaw
            ])
            self.traj_logger.update(front_location, vehicle_rotation)

@register_mode("inference")
class InferenceViewer(Viewer):
    def run(self):
        """Run inference mode for AI model testing and prediction visualization"""
        try:
            self._setup_simulation()
            
            # Generate random path for inference testing
            random_path = self.generate_randpath()
            midlane_wp = self.map_processor.precompute_waypoints(random_path)

            frame_id = 0
            while True if self.headless else self.controller.process_events(server_time = 1 / self.server_fps if self.server_fps != 0 else 0):
                self.step_world()
                self.data_bus(False)  # No control filtering for inference mode
                
                frame = None
                if not self.headless:
                    frame = self.choosen_sensor.extract_data()

                # Get map data
                unrouted_map, routed_map = self._get_map_data(frame_id)

                # Handle common operations
                multi_images_list = self._handle_multi_camera()
                self._handle_controller_inputs()
                self._apply_vehicle_controls()

                # Model inference logic
                output, extra = self._handle_model_inference(frame, multi_images_list, frame_id)

                # Render (inference mode with model predictions)
                self._render_with_predictions(frame, multi_images_list, routed_map, output, extra)

                frame_id += 1

        except KeyboardInterrupt:
            self.log.WARNING("InferenceViewer interrupted by user")
            self.controller.running = False
        except Exception as e:
            self.log.ERROR("InferenceViewer error", full_traceback = e)
            self.controller.running = False
        finally:
            self.close()
            if self.inference is not None:
                self.inference.stop()

    def _render_with_predictions(self, frame, multi_images_list, routed_map, output, extra):
        """Render with model predictions overlay"""
        self._render_base_hud(frame)
        self._render_multi_camera(multi_images_list)
        
        # Overlay model predictions on map
        if output is not None and self.controller.model_autopilot:
            try:
                if extra is not None and len(extra) == 3:
                    weights, muys, sigmas = extra
                    # GMM overlay functionality (commented out in original)
            finally: 
                pass

            # Overlay waypoints on map
            routed_map = overlay_waypoints_on_map(
                map_img = routed_map,
                waypoints = output.copy(), 
                scale = self.map_processor.final_scale, 
                meters_span = self.map_processor.original_range,
                color=(0, 255, 0),
                thickness=2,
                swap_axes=True,
                flip_lat=False,
                origin = "center"
            )
        
        self._render_map(routed_map)
        self._finalize_frame()

    def _handle_model_inference(self, frame, multi_images_list, frame_id):
        """Handle model inference logic and return output and extra data"""
        output = None
        extra = None
        if self.inference and self.controller.model_autopilot:
            if frame_id % 1 == 0:
                input_metadata = {
                    "I0": frame,
                }
                if multi_images_list is not None: 
                    for idx, image in enumerate(multi_images_list):
                        input_metadata[f"I{idx + 1}"] = image

                inp, debug = self.inference.pytorch.preprocessor(**input_metadata)
                self.inference.put(inp, self.sub_turn_signal.receive())
                output = self.inference.get()
                
                if output is not None:
                    if not isinstance(output, tuple):  # Model has no extra information
                        if isinstance(output, (float, np.float32)):  # If output is just steering
                            self.send_model_steer.send(float(output))
                        else:  # Output is waypoints
                            pass
                    else:  # With extra info
                        output, *extra = output
                        if len(output.shape) == 1:
                            self.send_model_steer.send(float(output[0]))
                        else:
                            norm_steer = lateral_control(output, Ld = 10, wheelbase = self.virt_vehicle.wheelbase, max_steer = self.virt_vehicle.max_steer)
                            speed = longitudinal_speed(output, 6, 0.04)
                            self.send_model_speed.send(float(speed))
                            self.send_model_steer.send(norm_steer)
        return output, extra


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
