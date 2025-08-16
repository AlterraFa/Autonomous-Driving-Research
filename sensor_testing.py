import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import carla
import queue
import numpy as np
import cv2
import traceback
from utils.spawner import Spawn
from utils.sensor_spawner import (
    SemanticSegmentation, 
    RGB,
    CarlaLabel as Clabel
)


tm_port = 8000

client = carla.Client("localhost", 2000)
world  = client.get_world()
tm     = client.get_trafficmanager(tm_port) 

settings = world.get_settings()
blueprints = world.get_blueprint_library()
spawn_pts  = world.get_map().get_spawn_points()


def get_frame(imageQueue: queue.Queue, targetFrame: int):
    while True:
        img = imageQueue.get()
        if img.frame == targetFrame:
            return img


if __name__ == "__main__":
    sync = True
    delta = 0.05

    # Apply settings to both world and traffic manager
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = None if not sync else delta
    world.apply_settings(settings)
    tm.set_synchronous_mode(sync)
    
    spawner = Spawn(world, tm)

    # spawn vehicle 
    spawner.spawn_mass_vehicle(10, autopilot = True)
    spawner.spawn_single_vehicle(autopilot = True)
    vehicle = spawner.single_vehicle
    
    image_queues = []; sensors = []

    semantic_sensor = SemanticSegmentation(blueprints, world)    
    semantic_sensor.set_attribute(name = "image_size_x", value = 800)
    semantic_sensor.set_attribute(name = "image_size_y", value = 600)
    semantic_sensor.spawn(attach_to = vehicle, z = 2, yaw = 0)
    
    rgb_sensor = RGB(blueprints, world)
    rgb_sensor.set_attribute(name = "image_size_x", value = 800)
    rgb_sensor.set_attribute(name = "image_size_y", value = 600)
    rgb_sensor.spawn(attach_to = vehicle, z = 2, yaw = 0)
    

    try:

        while True:
            frame_id = world.tick()
            semantic_image = semantic_sensor.extract_data(alpha = 1.0, layers = [Clabel.Road, Clabel.RoadLine, Clabel.Bicycle, Clabel.Bridge, Clabel.Ground, Clabel.SideWalk, Clabel.Vegetation])
            rgb_image = rgb_sensor.extract_data()
            
            
            cv2.imshow("Sensor stack", np.hstack([semantic_image, rgb_image]))
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
                
            
    except KeyboardInterrupt:
        print("Exiting ...")
    except Exception as e:
        traceback.print_exc()
    finally:
        semantic_sensor.destroy()
        rgb_sensor.destroy()
        spawner.destroy_vehicle()
        for walker in world.get_actors().filter("*walker*"):
            walker.destroy()
        
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        tm.set_synchronous_mode(False)
        world.apply_settings(settings)