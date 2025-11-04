import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import carla
import argparse
import pygame
import re

from utils.spawn.actor_spawner import Spawn, VehicleClass as VClass
from utils.spawn.sensor_spawner import (
    SemanticSegmentation as SemSeg, 
    RGB,
    GNSS,
    IMU, 
    Depth,
    CarlaLabel as Clabel
)
from utils.spawn.multicam import MultiCamera
from utils.math.path import ReplayHandler

from utils.control.world import World
from utils.control.vehicle_control import Vehicle
from utils.render.viewer import CarlaViewer
from utils.messages.logger import Logger

def get_recording_duration(log_path: str) -> float:
    """
    Returns the recording duration in seconds for a CARLA .log file.
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    report = client.show_recorder_file_info(log_path, True)

    m = re.search(r"Duration:\s*([0-9.]+)\s*seconds", report)
    if m:
        duration = float(m.group(1))
        print(f"Recording duration: {duration:.2f} seconds")
    else:
        print("No duration found")
    return duration

    
def main(args):
    pygame.init()

    ReplayHandler.turn_classify = args.use_turn
    Logger.set_levels("CUSTOM", "ERROR", "INFO", "WARNING")


    script_path = os.path.abspath(__file__)
    folder = os.path.dirname(script_path)
    path_2_recording = folder + "/" + args.replay + "/log.log"
    path_2_waypoints = folder + "/" + args.replay + "/trajectory.npy"

    if args.collect_data is None:
        dataset_dir = None
    else:
        dataset_dir = folder + "/" + args.collect_data + "/" + os.path.basename(args.replay) + "_" + ("temporal" if args.temporal else "spatial")
        os.makedirs(dataset_dir, exist_ok = True)

    client = carla.Client(args.host, args.port)
    virt_world = World(client, args.traffic_port)
    virt_world.sync = args.sync
    virt_world.delta = args.delay
    virt_world.disable_render = True
    virt_world.apply_settings()

    spawner = Spawn(virt_world.world, virt_world.tm)
    spawner.despawn_vehicles()
    rgb_sensor      = RGB(virt_world.world)
    # semantic_sensor = SemSeg(virt_world.world, convert_to = partial(SemSeg.SemanticData.to_image))
    gnss_sensor     = GNSS(virt_world.world)
    imu_sensor      = IMU(virt_world.world)
    # depth_sensor    = Depth(virt_world.world, convert_to = partial(Depth.DepthMap.to_log, invert = False, max_depth = 100))
    multi_rgb       = MultiCamera(virt_world.world, RGB, quantity = 2)
    # Set this first
    multi_rgb.set_attribute("image_size_x", value = 200)
    multi_rgb.set_attribute("image_size_y", value = 80)
    multi_rgb.set_attribute("fov", value = 60)
    
        

    duration = get_recording_duration(path_2_recording)
    client.show_recorder_file_info(path_2_recording, False)
    start = 1.2; stop = 4
    duration -= start + stop
    client.replay_file(path_2_recording, start, duration, 0)

    spawner.wait_for_actor_by_role("ego")
    controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
    
    game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync, headless = args.headless)
    game_viewer.init_sensor({
        rgb_sensor     : None, 
        # semantic_sensor: None, 
        gnss_sensor    : None, 
        imu_sensor     : None, 
        # depth_sensor   : None,
        multi_rgb      : {'x' : 0, 'y': [-0.25, .25], 'z': 2, 'pitch': -20, 'yaw': [-30, 30], 'roll': [-5, 5]}
    })
    game_viewer.run(replay_logging = [path_2_waypoints, duration], use_temporal_wp = args.temporal, data_collect_dir = dataset_dir, debug = args.debug, model_path = args.model_path)

    client.stop_replayer(True)
    
    
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
        "--replay",
        type = str,
        default = "None",
        help = "Replay Carla recording (.log file path is needed, recording time of .npy must correspond to .log)",   
        required = True
    )
    argparser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporal (time-based) waypoint generation instead of spatial."
    )
    argparser.add_argument(
        "--debug",
        action = "store_true",
        help = "Draw debugging waypoints onto the world"
    )
    argparser.add_argument(
        "--collect-data",
        type = str,
        default = None,
        help = "Data collection directory for DNN training"
    )
    argparser.add_argument(
        "--model-path",
        type = str,
        help = "Path to models file as well as its class reference",
        default = None
    )
    argparser.add_argument(
        "--use-turn",
        action = "store_true",
        help = "Turn on turn classification at junctions"
    )
    argparser.add_argument(
        "--headless",
        action = "store_true",
        help = "Enable Pygame headless rendering"
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)