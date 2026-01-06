import os, sys
import resource
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import toml
import carla
import argparse
import pygame
import datetime
import re
import numpy as np
import line_profiler

from utils.spawn.actor_spawner import Spawn, VehicleClass as VClass
from utils.spawn.sensor_spawner import (
    RGB,
    GNSS,
    IMU, 
)
from utils.spawn.multicam import MultiCamera

from tqdm.auto import tqdm
from utils.control.world import World
from utils.control.vehicle_control import Vehicle
from utils.render.viewer import CarlaViewer

# -- Plugins
from model.inference import AsyncInference
from utils.others.data_processor import TrajectoryBuffer
from utils.math.world_map import Map
from utils.math.path import ReplayHandler, OptimizePath

def get_recording_duration(log_path: str) -> float:
    """
    Returns the recording duration in seconds for a CARLA .log file.
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    report = client.show_recorder_file_info(log_path, True)

    m = re.search(r"Duration:\s*([0-9.]+)\s*seconds", report)
    if m:
        duration = float(m.group(1))
        print(f"Recording duration: {duration:.2f} seconds")
    else:
        print("No duration found")
    return duration

conf = toml.load(os.path.join(root, "./config/config.toml"))

map_conf = conf.get("MapRender", {})
RECT_DIM   = tuple(map_conf.get("rect_dim", [4, 3]))
MAP_OFFSET = tuple(map_conf.get("map_offset", [100, 100]))
MAP_RANGE  = tuple(map_conf.get("map_range", [50, 50]))
MAP_RESIZE = tuple(map_conf.get("map_resize", [200, 200]))
MAP_SCALE  = map_conf.get("map_scale", 3)

path_optim_conf = conf.get("PathOptimizer", {})
PATH_STEP = path_optim_conf.get("path_step", 2.0) 
EXCLUDE_PARAMS = path_optim_conf.get("exclude_params", [0, 0, 0])

MIN_SAVING_DIST = 0.4

    
def main(args):
    pygame.init()

    client = carla.Client(args.host, args.port)
    virt_world = World(client, args.traffic_port)
    virt_world.sync = args.sync
    virt_world.delta = args.delay
    virt_world.disable_render = True
    virt_world.apply_settings()

    rgb_sensor = RGB(virt_world.world)
    gnss_sensor = GNSS(virt_world.world)
    imu_sensor = IMU(virt_world.world)
    
    
    script_path = os.path.abspath(__file__)
    folder = os.path.dirname(script_path)
        
    spawner = Spawn(virt_world.world)
    spawner.despawn_vehicles()
    
    viewer_args = {
        "world"  : virt_world, 
        "width"  : args.width,
        "height" : args.height,
        "sync"   : args.sync, 
        "fps"    : args.fps
    }
    

    if args.mode == "manual" or args.mode == "inference":
        spawner.spawn_mass_vehicle(0, exclude = [VClass.Large, VClass.Tiny])
        spawner.spawn_single_vehicle(bp_id = "vehicle.dodge.charger_2020", exclude = [VClass.Large, VClass.Medium, VClass.Tiny], autopilot = False)
        controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
        viewer_args.update({"vehicle": controlling_vehicle})

    if args.mode == "replay":
        path_2_recording = folder + "/" + args.replay_dir + "/log.log"
        path_2_waypoints = folder + "/" + args.replay_dir + "/trajectory.npy"
        
        if args.collect_data is None:
            dataset_dir = None
        else:
            dataset_dir = folder + "/" + args.collect_data + "/" + os.path.basename(args.replay_dir) + "_" + ("temporal" if args.temporal else "spatial")
            os.makedirs(dataset_dir, exist_ok = True)

        duration = get_recording_duration(path_2_recording)
        client.show_recorder_file_info(path_2_recording, False)
        start = 0; stop = 4
        duration -= start + stop
        client.replay_file(path_2_recording, start, duration, 0)
        
        spawner.wait_for_actor_by_role("ego")
        controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)  

        viewer_args.update({"vehicle": controlling_vehicle, "duration": duration, "headless": args.headless})
    
    game_viewer = CarlaViewer(**viewer_args)
    game_viewer.init_sensor({
        rgb_sensor     : None, 
        gnss_sensor    : None, 
        imu_sensor     : None, 
    })

    lp = line_profiler.LineProfiler()
    lp.add_function(game_viewer.run)
    lp.add_function(game_viewer.step_world)
    lp_wrapper = lp(game_viewer.run)

    map_processor = Map(
        world = virt_world,
        rect_dim = RECT_DIM,
        map_offset = MAP_OFFSET,
        range_ = MAP_RANGE,
        resize_to = MAP_RESIZE,
        scale = MAP_SCALE
    )
    path_optimizer = OptimizePath(
        virt_world, 
        step = PATH_STEP, 
        exclude_circle = EXCLUDE_PARAMS,
    )
    game_viewer.attach_plugins(path_optimizer = path_optimizer)
    
    if args.mode == "replay":
        true_trajectories    = np.load(path_2_waypoints)
        midlane_trajectories = map_processor.precompute_waypoints(true_trajectories)
        replayer             = ReplayHandler(virt_world, true_trajectories, midlane_trajectories, dataset_dir, args.temporal, args.debug)
        pbar                 = tqdm(total = duration, unit = 'server second', desc = "Play duration", ncols = 125)
        game_viewer.attach_plugins(
            replayer = replayer, 
            pbar = pbar, 
            map_processor = map_processor
        )
        lp.add_function(game_viewer.map_processor.retrieve_map)
        lp_wrapper()

    if args.mode == "manual":
        virt_world.tm.ignore_signs_percentage(controlling_vehicle.vehicle, args.ignore_signs)
        if args.record:
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = f"{folder}/{args.record}/recording_{date}"
            os.mkdir(directory)
            client.start_recorder(f"{directory}/log.log")

            trajectory_logging = TrajectoryBuffer(directory, min_dt_s = MIN_SAVING_DIST)
            game_viewer.attach_plugins(
                traj_logger = trajectory_logging, 
                map_processor = map_processor
            )
            
            lp_wrapper()
            client.stop_recorder()
        else:
            game_viewer.attach_plugins(map_processor = map_processor)
            lp_wrapper()

    if args.mode == "inference":
        inference = AsyncInference(args.model_path, device = 'cuda', batch_output = False)
        game_viewer.override_render_map = True
        game_viewer.attach_plugins(
            inference = inference, 
            map_processor = map_processor
        )
        lp_wrapper()

    lp.dump_stats("profile_results.lprof")
    spawner.despawn_vehicles()
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description = "CARLA Runner")

    # ====================================================== #
    #                   SHARED ARGUMENT
    # ====================================================== #
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
        help = "Time step for synchronize server running"
    )

    argparser.add_argument(
        "--fps",
        default = 144,
        type = float,
        help = "Max fps for pygame rendering"
    )

    subparser = argparser.add_subparsers(dest = "mode", help = "Execution mode", required = True)

    
    # ====================================================== #
    #                MANUAL CONTROL ARGUMENT
    # ====================================================== #
    manual_parser = subparser.add_parser("manual", help = "Manual driving and recording states")    
    manual_parser.add_argument(
        "--record",
        type = str,
        default = None,
        help = "Specify the relative root log directory and enable the record mode"
    )
    manual_parser.add_argument(
        "--ignore-signs",
        type = float,
        default = 0,
        help = "Ignore traffic sign rules (by percentage)"
    )
    
    
    # ====================================================== #
    #                   REPLAY ARGUMENT
    # ====================================================== #
    replay_parser = subparser.add_parser("replay", help = "Replay the recorded CARLA's states")
    replay_parser.add_argument(
        "--replay-dir",
        type = str,
        default = "None",
        help = "Replay Carla recording (.log file path is needed, recording time of .npy must correspond to .log)",   
        required = True
    )
    replay_parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporal (time-based) waypoint generation instead of spatial."
    )
    replay_parser.add_argument(
        "--debug",
        action = "store_true",
        help = "Draw debugging waypoints onto the world"
    )
    replay_parser.add_argument(
        "--collect-data",
        type = str,
        default = None,
        help = "Data collection directory for DNN training"
    )
    replay_parser.add_argument(
        "--use-turn",
        action = "store_true",
        help = "Turn on turn classification at junctions"
    )
    replay_parser.add_argument(
        "--headless",
        action = "store_true",
        help = "Enable Pygame headless rendering"
    )
    

    # ====================================================== #
    #                  INFERENCE ARGUMENT
    # ====================================================== #
    infer_parser = subparser.add_parser("inference", help = "Autonomous inference")
    infer_parser.add_argument(
        "--debug",
        action = "store_true",
        help = "Draw debugging waypoints onto the world"
    )
    infer_parser.add_argument(
        "--model-path",
        type = str,
        help = "Path to models file as well as its class reference",
        required = True
    )
    infer_parser.add_argument(
        "--render-map",
        action = "store_true",
        help = "Force render map"
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)