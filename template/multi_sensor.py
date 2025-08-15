import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import carla
import random
import queue
import numpy as np
import cv2
import time
import traceback
from utils.lidar_visualization import LIDARVisualizer
from utils.spawner import Spawn


tm_port = 8000

client = carla.Client("localhost", 2000)
world  = client.get_world()
tm     = client.get_trafficmanager(tm_port) 

settings = world.get_settings()
blueprints = world.get_blueprint_library()
spawn_pts  = world.get_map().get_spawn_points()

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
    
    # walker_bp     = blueprints.filter("*walker*") 
    # controller_bp = blueprints.filter("controller.ai.walker")
    # exist_walkers = []
    # for _ in range(5):
    #     walker_spawn = world.get_random_location_from_navigation()
    #     walker_trans = carla.Transform(walker_spawn)
    #     walker       = world.try_spawn_actor(random.choice(walker_bp), walker_trans)
    #     if walker:
    #         controller     = world.try_spawn_actor(random.choice(controller_bp), walker_trans, attach_to = walker)
    #         exist_walkers += [walker]
        

        
    vehicle = spawner.single_vehicle
    sensors = blueprints.filter('sensor*')
    image_queues = []; sensors = []
    
    for angle in [0]:
        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        transform = carla.Transform(carla.Location(z = 2.0), carla.Rotation(yaw = angle))
        
        camera      = world.spawn_actor(camera_bp, transform, attach_to = vehicle)
        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        image_queues += [image_queue]
        sensors += [camera]

    lidar_bp  = blueprints.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('dropoff_general_rate', '0.1')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', f"{1 / delta}")
    lidar_bp.set_attribute('lower_fov', '-25.0')
    lidar_bp.set_attribute('upper_fov', '15.0')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '500000')
    transform = carla.Transform(carla.Location(z = 2.0))
    lidar     = world.spawn_actor(lidar_bp, transform, attach_to = vehicle)

    lidar_queue = queue.Queue()
    lidar.listen(lidar_queue.put)
    sensors += [lidar]
    
    # Initialize open3d
    lidar_visual = LIDARVisualizer(50)
    
    
    try:
        while True:
            world.tick()
            images = []
            for buffer in image_queues:
                data    = buffer.get()
                image   = np.frombuffer(data.raw_data, dtype = np.uint8).reshape((data.height, data.width, 4))
                images += [image]
            image_arr = np.hstack(images)

            lidar_data = lidar_queue.get()
            pcd_arr    = np.frombuffer(lidar_data.raw_data, dtype = np.float32).reshape((-1, 4))
            lidar_visual.display(pcd_arr[:, :-1], pcd_arr[:, -1])

            
            
            cv2.imshow("Sensor stack", image_arr)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
                
            
    except KeyboardInterrupt:
        print("Exiting ...")
    except Exception as e:
        traceback.print_exc()
    finally:
        lidar_visual.destroy_window()
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()
        spawner.destroy_vehicle()
        for walker in world.get_actors().filter("*walker*"):
            walker.destroy()
        
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        tm.set_synchronous_mode(False)
        world.apply_settings(settings)