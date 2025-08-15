#!/home/alterraonix/miniconda/envs/Core/bin/python
"""If we don't run sync mode, then we won't be able to see other actors in the scene
In async mode, actors update their environment independently"""
import carla
import argparse
import queue
import cv2
import numpy as np

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
    settings.fixed_delta_seconds = None
    
    tm.set_synchronous_mode(sync)
    world.apply_settings(settings)
    
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    spawn     = world.get_map().get_spawn_points()[0]
    camera    = world.try_spawn_actor(blueprint, spawn)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    
    # when in synchronous mode tick the world (useful for training)
    try:
        while True:
            world.tick()

            transform = camera.get_transform()

            location = transform.location; location.x += .1
            rotation = transform.rotation; rotation.roll += .1

            camera.set_transform(carla.Transform(location, rotation))
            
            image = image_queue.get()
            array = np.frombuffer(image.raw_data, dtype = np.uint8)
            image = array.reshape((image.height, image.width, 4))
            
            cv2.imshow("Image", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(e)
    finally:
        camera.stop()
        camera.destroy()
        print("Camera destroyed")