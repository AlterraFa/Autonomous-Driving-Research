import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import carla
import numpy as np
import cv2
import traceback
from functools import partial

from utils.spawn.actor_spawner import Spawn, VehicleClass as VClass
from utils.spawn.sensor_spawner import (
    SemanticSegmentation, 
    RGB,
    LidarRaycast,
    GNSS,
    IMU, 
    Depth,
    InstanceSegmentation,
    CarlaLabel as Clabel
)
from rich import print


tm_port = 8000

client = carla.Client("localhost", 2000)
client.load_world("Town10HD")
world  = client.get_world()
tm     = client.get_trafficmanager(tm_port) 

settings = world.get_settings()
blueprints = world.get_blueprint_library()
spawn_pts  = world.get_map().get_spawn_points()


if __name__ == "__main__":
    sync = True
    delta = 0.07

    # Apply settings to both world and traffic manager
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = None if not sync else delta
    world.apply_settings(settings)
    tm.set_synchronous_mode(sync)
    
    spawner = Spawn(world, tm)

    # spawn vehicle 
    spawner.spawn_mass_vehicle(10, autopilot = True, exclude = [VClass.Bikes, VClass.Trucks, VClass.Large])
    spawner.spawn_single_vehicle(autopilot = True, random_offset = 30, exclude = [VClass.Large, VClass.Medium, VClass.Tiny])
    vehicle = spawner.single_vehicle
    
    image_queues = []; sensors = []

    semantic_sensor = SemanticSegmentation(world, convert_to = SemanticSegmentation.SemanticData.to_image)    
    semantic_sensor.set_attribute(name = "image_size_x", value = 800)
    semantic_sensor.set_attribute(name = "image_size_y", value = 600)
    semantic_sensor.spawn(attach_to = vehicle, z = 2, yaw = 0)
    
    rgb_sensor = RGB(world)
    rgb_sensor.set_attribute(name = "image_size_x", value = 800)
    rgb_sensor.set_attribute(name = "image_size_y", value = 600)
    rgb_sensor.spawn(attach_to = vehicle, z = 2, yaw = 0)
    
    lidar_sensor = LidarRaycast(world)
    lidar_sensor.set_attribute("dropoff_general_rate", 0.1)
    lidar_sensor.set_attribute("range", 100)
    lidar_sensor.set_attribute("rotation_frequency", 1 / delta)
    lidar_sensor.set_attribute('lower_fov', -25.9)
    lidar_sensor.set_attribute('upper_fov', 15.0)
    lidar_sensor.set_attribute('channels', 64)
    lidar_sensor.set_attribute('points_per_second', 500000)
    lidar_sensor.spawn(attach_to = vehicle, z = 2)

    gsss_sensor  = GNSS(world)
    gsss_sensor.set_attribute("role_name", 'what')
    gsss_sensor.spawn(attach_to = vehicle, z = 0)
    
    imu_sensor   = IMU(world)
    imu_sensor.spawn(attach_to = vehicle)
    
    depth_sensor = Depth(world, convert_to = partial(Depth.DepthMap.to_log, invert = False))
    depth_sensor.spawn(attach_to = vehicle, z = 2)
    
    instance_sensor = InstanceSegmentation(world)
    instance_sensor.spawn(attach_to = vehicle, z = 2)
    
    steer = 0; max_steer = 30
    throttle = 0; max_throttle = 100

    try:

        while True:
            frame_id = world.tick()
            # semantic_image = semantic_sensor.extract_data()
            # rgb_image      = rgb_sensor.extract_data()
            # geo_location   = gsss_sensor.extract_data(return_ecf = True, return_enu = True)
            # imu_data       = imu_sensor.extract_data()
            # depth_data     = depth_sensor.extract_data()
            # lidar_data     = lidar_sensor.extract_data()
            # lidar_sensor.visualize
            
            # cv2.imshow("Sensor stack", rgb_image)
            
            # key = cv2.waitKey(1)
            # if key == ord("w"):
            #     throttle += 1
            # if key == ord('s'):
            #     throttle -= 1
            # if abs(throttle) > max_throttle: throttle = max_throttle * (throttle / abs(throttle))
            # if key == ord("a"):
            #     steer += 1
            # if key == ord('d'):
            #     steer -= 1
            # if abs(steer) > max_steer: steer = max_steer * (steer / abs(steer))
            # vehicle.apply_control(carla.VehicleControl(throttle = throttle, steer = steer))
            
            
            # if key == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
                
            
    except KeyboardInterrupt:
        print("Exiting ...")
    except Exception as e:
        traceback.print_exc()
    finally:
        semantic_sensor.destroy()
        rgb_sensor     .destroy()
        lidar_sensor   .destroy()
        gsss_sensor    .destroy()
        imu_sensor     .destroy()
        depth_sensor   .destroy()
        instance_sensor.destroy()
        spawner.destroy_all_vehicles()
        for walker in world.get_actors().filter("*walker*"):
            walker.destroy()
        
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        tm.set_synchronous_mode(False)
        world.apply_settings(settings)  