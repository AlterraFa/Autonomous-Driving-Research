import carla
import time
import random

tm_port = 8000

client   = carla.Client("localhost", 2000)
world    = client.get_world()
settings = world.get_settings()
blueprints = world.get_blueprint_library()
tm       = client.get_trafficmanager(tm_port)
spawn    = world.get_map().get_spawn_points()

replay = True

if __name__ == "__main__":
    
    if not replay:
        client.start_recorder("log1.log", additional_data = True)

        for vehicle in world.get_actors().filter("*vehicle*"):
            vehicle.destroy()

        car_bp = blueprints.filter("*vehicle*")
        exist_vehicles = []
        for id in range(30):
            vehicle = world.try_spawn_actor(random.choice(car_bp), random.choice(spawn))
            if vehicle: exist_vehicles += [vehicle]
            
        for vehicle in exist_vehicles:
            vehicle.set_autopilot(True)


        start = time.time()
        while time.time() - start < 50:
            ...
            
        client.stop_recorder()
            
    else:
        print(client.show_recorder_file_info("log1.log", False))
        client.replay_file("log1.log", 0, 0, 76)