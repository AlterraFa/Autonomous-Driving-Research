import pygame
import numpy as np
import time

from control.world import World

from typing import Optional
from enum import Enum
from rich import print


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
    def __init__(self, world: World, vehicle, width: int, height: int, sync: bool = False, fps: int = 144):
        self.virt_world = world
        self.world = world.world
        self.width = width
        self.height = height
        self.sync = sync
        self.fps = fps
        self.vehicle = vehicle

        self.display: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.running = False
        self.rgb_sensor = None  
        self.sensors_list = dict()
        self.camera_keys = []
        
        self.cleanup_done = False
        
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


    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

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
        pygame.display.flip()

    def step_world(self) -> None:
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
            
    def change_view(self, view_name):
        self.choosen_sensor.spawn(attach_to = self.vehicle, **getattr(CameraView, view_name).value)

    def run(self) -> None:
        if self.display is None:
            self.init_win()

        self.controller = KeyboardController()
        try:
            while self.controller.process_events():
                self.step_world()
                start = time.time()
                if not self.handle_events():
                    self.controller.running = False
                    break

                data = self.choosen_sensor.extract_data()
                if isinstance(data, tuple): frame = data[0]
                else: frame = data
                if frame is not None:
                    self.draw_frame(frame)
                    
                    
                if self.controller.view_changed:
                    self.choosen_sensor.change_view(**getattr(CameraView, self.controller.view_name).value)
                if self.controller.camera_changed:
                    self.switch_camera(self.controller.camera_step)
                
                # print(1 / (time.time() - start), end = '\r')
                if self.clock:
                    self.clock.tick(self.fps)
        except KeyboardInterrupt:
            print("[yellow][INFO]: Viewer interrupted by user[/]")
            self.controller.running = False
        except Exception as e:
            print(f"[red][ERROR]: Viewer error: {e}[/]")
            self.controller.running = False
        finally:
            self.close()

    def close(self) -> None:
        if self.cleanup_done:
            return
        self.controller.running = False
        self.cleanup_done = True
        
        print("[blue][INFO]: Closing CarlaViewer...[/]")
        try:
            self.virt_world.factory_reset()
        except Exception as e:
            print(f"[red][ERROR]: World reset failed: {e}[/]")

        for name, sensor in list(self.sensors_list.items()):
            try:
                sensor.destroy()
            except Exception as e:
                print(f"[red][ERROR]: Failed to cleanup sensor {name}: {e}[/]")
        self.sensors_list.clear()

        try:
            if pygame.get_init():
                pygame.quit()
                print("[green][INFO]: Pygame closed successfully[/]")
        except Exception as e:
            print(f"[red][ERROR]: Pygame quit failed: {e}[/]")


    def __del__(self):
        """Destructor to ensure cleanup"""
        if not self.cleanup_done:
            self.close()

class KeyboardController:
    def __init__(self):
        pygame.key.set_repeat()  # no auto-repeat by default
        self.running = True
        
        self.view_name = "FIRST_PERSON"; self.view_changed = False
        self.camera_step = 1; self.camera_changed = False
        self.prev_keys_view = pygame.key.get_pressed()

    def process_events(self):
        """Process keyboard + window events.
        Returns False if the program should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_k] or keys[pygame.K_ESCAPE]:
            self.running = False

        self.process_view()
        return self.running
    
    def process_view(self):
        self.view_changed = False; self.camera_changed = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.view_name = "FIRST_PERSON"
            self.view_changed = True
        elif keys[pygame.K_DOWN]:
            self.view_name = "THIRD_PERSON"
            self.view_changed = True
            
        if keys[pygame.K_RIGHT] and not self.prev_keys[pygame.K_RIGHT]:
            self.camera_changed = True
            self.camera_step = 1
        elif keys[pygame.K_LEFT] and not self.prev_keys[pygame.K_LEFT]:
            self.camera_changed = True
            self.camera_step = -1
            
        self.prev_keys = keys