import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import carla
import argparse
import pygame
import numpy as np
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

from rich import print

from control.world import World
from control.vehicle_control import Vehicle
from control.viewer import CarlaViewer


    
def main(args):
    pygame.init()

    client = carla.Client(args.host, args.port)
    virt_world = World(client, args.traffic_port)
    virt_world.sync = args.sync
    virt_world.delta = args.delay
    virt_world.disable_render = True
    virt_world.apply_settings()
    
    spawner = Spawn(virt_world.world, virt_world.tm)
    spawner.spawn_mass_vehicle(10, exclude = [VClass.Large, VClass.Tiny])
    spawner.spawn_single_vehicle(bp_id = None, exclude = [VClass.Large, VClass.Medium, VClass.Tiny], autopilot = False)
    controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
    
    rgb_sensor = RGB(virt_world.world)
    semantic_sensor = SemSeg(virt_world.world, convert_to = partial(SemSeg.SemanticData.to_image, alpha = .5))
    gnss_sensor = GNSS(virt_world.world)
    imu_sensor = IMU(virt_world.world)
    depth_sensor = Depth(virt_world.world, convert_to = partial(Depth.DepthMap.to_log, invert = False, max_depth = 100))
    
    
    
    game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync)
    game_viewer.init_sensor([rgb_sensor, semantic_sensor, gnss_sensor, imu_sensor, depth_sensor])
    game_viewer.run()


    spawner.destroy_all_vehicles()
    
    
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
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)