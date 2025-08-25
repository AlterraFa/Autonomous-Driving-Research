import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import carla
import argparse
import pygame
import datetime
from functools import partial

from utils.spawner import Spawn, VehicleClass as VClass
from utils.sensor_spawner import (
    SemanticSegmentation as SemSeg, 
    RGB,
    GNSS,
    IMU, 
    Depth,
    CarlaLabel as Clabel
)

from control.world import World
from control.vehicle_control import Vehicle, wait_for_actor_by_role
from control.viewer import CarlaViewer


    
def main(args):
    pygame.init()

    client = carla.Client(args.host, args.port)
    virt_world = World(client, args.traffic_port)
    virt_world.sync = args.sync
    virt_world.delta = args.delay
    virt_world.disable_render = True
    virt_world.apply_settings()

    rgb_sensor = RGB(virt_world.world)
    semantic_sensor = SemSeg(virt_world.world, convert_to = partial(SemSeg.SemanticData.to_image))
    gnss_sensor = GNSS(virt_world.world)
    imu_sensor = IMU(virt_world.world)
    depth_sensor = Depth(virt_world.world, convert_to = partial(Depth.DepthMap.to_log, invert = False, max_depth = 100))
    
    if args.replay != "None" and args.record == True:
        raise NotImplementedError(f"Replay and recording simultaneously selected.")
    
    if args.replay != "None":
        folder = os.path.dirname(__file__)
        path_2_recording = folder + "/" + args.replay
        client.show_recorder_file_info(path_2_recording, False)
        client.replay_file(path_2_recording, 0, 0, 0)

        vehicle = wait_for_actor_by_role(virt_world.world, "ego")
        if vehicle is None:
            raise RuntimeError("Could not find a vehicle with role_name='ego' in the replay.")
        controlling_vehicle = Vehicle(vehicle, virt_world.world)
        
        game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync)
        game_viewer.init_sensor([rgb_sensor, semantic_sensor, gnss_sensor, imu_sensor, depth_sensor])
        game_viewer.run()

        return
    else:
        
        spawner = Spawn(virt_world.world, virt_world.tm)
        spawner.spawn_mass_vehicle(10, exclude = [VClass.Large, VClass.Tiny])
        spawner.spawn_single_vehicle(bp_id = "vehicle.dodge.charger_2020", exclude = [VClass.Large, VClass.Medium, VClass.Tiny], autopilot = False)
        controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
        
        if args.record:
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.dirname(__file__)
            client.start_recorder(f"{folder}/log/recording_{date}.log")
        game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync)
        game_viewer.init_sensor([rgb_sensor, semantic_sensor, gnss_sensor, imu_sensor, depth_sensor])
        game_viewer.run()
        if args.record:
            client.stop_recorder()

        spawner.destroy_all_vehicles()

        return
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        "--traffic-port",
        metavar = "TMP",
        default = 8000,
        type = int,
        help = "Traffic manager port for actor autopilot function"
    )
    argparser.add_argument(
        "--delay",
        default = 0.05,
        type = float,
        help = "Max fps for synchronize running"
    )
    argparser.add_argument(
        "--record",
        action = "store_true",
        help = "Record carla state"
    )
    argparser.add_argument(
        "--replay",
        type = str,
        default = "None",
        help = "Replay Carla recording"   
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)