import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import carla
import argparse
import pygame

from utils.spawn.actor_spawner import Spawn, VehicleClass as VClass
from utils.spawn.sensor_spawner import (
    RGB,
    GNSS,
    IMU, 
)
from utils.spawn.multicam import MultiCamera

from utils.control.world import World
from utils.control.vehicle_control import Vehicle
from utils.render.viewer import CarlaViewer

    
def main(args):
    pygame.init()

    CarlaViewer.override_render_map = args.render_map


    client = carla.Client(args.host, args.port)
    virt_world = World(client, args.traffic_port)
    virt_world.sync = args.sync
    virt_world.delta = args.delay
    virt_world.disable_render = True
    virt_world.apply_settings()

    spawner = Spawn(virt_world.world, virt_world.tm)
    spawner.despawn_vehicles()
    spawner.spawn_mass_vehicle(6, exclude = [VClass.Large, VClass.Tiny])
    spawner.spawn_single_vehicle(bp_id = "vehicle.dodge.charger_2020", exclude = [VClass.Large, VClass.Medium, VClass.Tiny], autopilot = False)

    rgb_sensor      = RGB(virt_world.world)
    gnss_sensor     = GNSS(virt_world.world)
    imu_sensor      = IMU(virt_world.world)
    multi_rgb       = MultiCamera(virt_world.world, RGB, quantity = 2)
    # Set this first
    multi_rgb.set_attribute("image_size_x", value = 200)
    multi_rgb.set_attribute("image_size_y", value = 80)
    multi_rgb.set_attribute("fov", value = 60)
    
    controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
    
    game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync)
    game_viewer.init_sensor({
            rgb_sensor     : None, 
            gnss_sensor    : None, 
            imu_sensor     : None,
            multi_rgb      : {'x' : 0, 'y': [-0.25, .25], 'z': 2, 'pitch': -20, 'yaw': [-30, 30], 'roll': [-5, 5]}
        })
    game_viewer.run(model_path = args.model_path)
    
    
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
        "--traffic-port",
        metavar = "TMP",
        default = 8000,
        type = int,
        help = "Traffic manager port for actor autopilot function"
    )
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
        "--delay",
        default = 0.05,
        type = float,
        help = "Max fps for synchronize running"
    )
    argparser.add_argument(
        "--debug",
        action = "store_true",
        help = "Draw debugging waypoints onto the world"
    )
    argparser.add_argument(
        "--model-path",
        type = str,
        help = "Path to models file as well as its class reference",
        required = True
    )
    argparser.add_argument(
        "--render-map",
        action = "store_true",
        help = "Force render map"
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)