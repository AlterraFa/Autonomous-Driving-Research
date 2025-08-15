import carla
import argparse
import queue
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser(description = "Asynchronous mode selector")

parser.add_argument("--carla-port", default = 2000, type = int, help = "Port to carla")
parser.add_argument("--tm-port", default = 8000, type = int, help = "Port to traffic manager")
parser.add_argument("--sync", action = "store_true", help = "Enable synchronous mode")
args = parser.parse_args()

client = carla.Client("localhost", args.carla_port)
world  = client.get_world()
tm     = client.get_trafficmanager(args.tm_port)
settings = world.get_settings()

if __name__ == "__main__":
    sync = args.sync 
    
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 0.05
    
    tm.set_synchronous_mode(sync)
    world.apply_settings(settings)

    exist_vehicles = world.get_actors().filter("*vehicle*")
    for vehicle in exist_vehicles:
        vehicle.destroy()
        
    
    # Initializing blueprints and spawning loc        
    blueprints = world.get_blueprint_library()
    cam_bp     = blueprints.find("sensor.camera.rgb")
    vehicle_bp = blueprints.filter("*vehicle*")
    spawn_points = world.get_map().get_spawn_points()
    ego_camera = carla.Transform(carla.Location(z = 2.0, x = -5.0))

    # Add in vehicles
    exist_vehicles = []
    for i in range(20):
        vehicle = world.try_spawn_actor(random.choice(vehicle_bp), random.choice(spawn_points))
        if vehicle:
            exist_vehicles += [vehicle]
            print(f"Added vehicle: {i + 1}")

    # Add in camera (attach to a random car) and extract image
    camera = world.try_spawn_actor(cam_bp, ego_camera, attach_to = exist_vehicles[-1])   
    image_buffer = queue.Queue()
    camera.listen(image_buffer.put) 
    

    # The world need to tick a few frame for traffic manager to pick up vehicle for autopilot
    for _ in range(2):
        world.tick()

    for vehicle in exist_vehicles:
        vehicle.set_autopilot(True, args.tm_port)

    try:
        while sync:
            world.tick()
            image = image_buffer.get()
            image_color = np.frombuffer(image.raw_data, dtype = np.uint8).reshape((image.height, image.width, 4))
            
            cv2.imshow("Ego Camera", image_color) 
            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
            
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("Exiting ...")
    finally:
        camera.stop()
        camera.destroy()
        print("Camera nuked")