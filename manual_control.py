import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import carla
import argparse
import pygame
import datetime
from functools import partial
import re

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
    
    script_path = os.path.abspath(__file__)
    folder = os.path.dirname(script_path)
    if args.replay != "None":
        spawner = Spawn(virt_world.world, virt_world.tm)
        spawner.destroy_all_vehicles()

        path_2_recording = folder + "/" + args.replay + "/log.log"
        path_2_waypoints = folder + "/" + args.replay + "/trajectory.npy"
        duration = get_recording_duration(path_2_recording)

        client.show_recorder_file_info(path_2_recording, False)
        client.replay_file(path_2_recording, 0, 0, 0) # Start replay: start=0.0, duration=0.0 (entire), follow_id=0 (don't auto-follow)

        vehicle = wait_for_actor_by_role(virt_world.world, "ego")
        if vehicle is None:
            raise RuntimeError("Could not find a vehicle with role_name='ego' in the replay.")
        controlling_vehicle = Vehicle(vehicle, virt_world.world)
        
        game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync)
        game_viewer.init_sensor([rgb_sensor, semantic_sensor, gnss_sensor, imu_sensor, depth_sensor])
        game_viewer.run(replay_logging = [path_2_waypoints, duration], use_temporal_wp = args.temporal, debug = True)

        client.stop_replayer(True)
        return
    else:
        
        spawner = Spawn(virt_world.world, virt_world.tm)
        spawner.destroy_all_vehicles()
        spawner.spawn_mass_vehicle(6, exclude = [VClass.Large, VClass.Tiny])
        spawner.spawn_single_vehicle(bp_id = "vehicle.dodge.charger_2020", exclude = [VClass.Large, VClass.Medium, VClass.Tiny], autopilot = False)
        controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
        virt_world.tm.ignore_signs_percentage(controlling_vehicle.vehicle, args.ignore_signs)
        
        if args.record:
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = f"{folder}/log/recording_{date}"
            os.mkdir(directory)
            client.start_recorder(f"{directory}/log.log")
        game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync)
        game_viewer.init_sensor([rgb_sensor, semantic_sensor, gnss_sensor, imu_sensor, depth_sensor])
        if args.record:
            game_viewer.run(save_logging = directory)
            client.stop_recorder()
        else:
            game_viewer.run()

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
        help = "Replay Carla recording (.log file path is needed, recording time of .npy must correspond to .log)"   
    )
    argparser.add_argument(
        "--ignore-signs",
        type = float,
        default = 0,
        help = "Ignore traffic sign rules (by percentage)"
    )
    argparser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporal (time-based) waypoint generation instead of spatial."
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)
    []
    [0, 2, 1, 3, 4, 5, 7]
    [0, 2, 3, 6, 10, 15, 22]