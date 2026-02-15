import os, sys
import time
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
from utils.messages.logger import Logger
logger = Logger()

from utils.spawn.actor_spawner import Spawn
from utils.spawn.sensor_spawner import (
    RGB,
    GNSS,
    IMU, 
    SemanticSegmentation,
    SensorSpawn
)
from config.enum import (
    VehicleClass as VClass,
    CarlaLabel as CLabel
)

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
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
        logger.INFO(f"Recording duration: {duration:.2f} seconds")
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

gnss_conf = conf.get("GPS", {})
MEAN_DELAY   = gnss_conf.get("mean_delay", 0)
STDDEV_DELAY = gnss_conf.get("stddev_delay", 0)
LAT_STDDEV   = gnss_conf.get("lat_stddev", 0)
LON_STDDEV   = gnss_conf.get("lon_stddev", 0)
FREQ         = gnss_conf.get("frequency", 50)

MIN_SAVING_DIST = 0.4


def _expand_replay_dirs(replay_dirs):
    if isinstance(replay_dirs, str):
        replay_dirs = [replay_dirs]
    expanded = []
    for item in replay_dirs:
        if "," in item:
            expanded.extend([p.strip() for p in item.split(",") if p.strip()])
        else:
            expanded.append(item)
    return expanded

def refresh_world_references(client, virt_world, spawner=None):
    """
    Refresh world references after loading a new map.
    Updates virt_world.world, reapplies settings, and updates spawner.world if provided.
    """
    virt_world.world = client.get_world()
    virt_world.apply_settings()
    
    if spawner is not None:
        spawner.world = virt_world.world
    
    logger.INFO(f"World references refreshed")

def reinit_sensors(virt_world):
    """
    Reinitialize sensors with the new world reference.
    """
    rgb_sensor = RGB(virt_world.world)
    gnss_sensor = GNSS(virt_world.world, freq_hz = FREQ, mu_ms = MEAN_DELAY, sigma_ms = STDDEV_DELAY)
    gnss_sensor.set_attribute("noise_lat_stddev", LAT_STDDEV / 111320.0)
    gnss_sensor.set_attribute("noise_lon_stddev", LON_STDDEV / 111320.0)
    imu_sensor = IMU(virt_world.world)
    imu_sensor.set_attribute("noise_gyro_bias_x", 0.005)
    imu_sensor.set_attribute("noise_gyro_bias_y", 0.005)
    
    logger.INFO("Sensors reinitialized with new world")
    return rgb_sensor, gnss_sensor, imu_sensor

def load_recording(client, virt_world, spawner, folder, replay_dir):
    map_name = replay_dir.split("/")[-2]
    logger.INFO(f"Loading map: {map_name}")
    client.load_world(map_name)
    
    refresh_world_references(client, virt_world, spawner)
    
    logger.INFO(f"Stabilizing world after map load...")
    for _ in range(20):
        if virt_world.world.get_settings().synchronous_mode:
            virt_world.world.tick()
        else:
            time.sleep(0.01)
    
    path_2_recording = folder + "/" + replay_dir + "/log.log"
    path_2_waypoints = folder + "/" + replay_dir + "/trajectory.npy"

    if not os.path.exists(path_2_recording):
        logger.ERROR(f"Replay log not found: {path_2_recording}")
        return False, None, None, None
    if not os.path.exists(path_2_waypoints):
        logger.ERROR(f"Trajectory file not found: {path_2_waypoints}")
        return False, None, None, None

    if args.collect_data is None:
        dataset_dir = None
    else:
        dataset_dir = folder + "/" + args.collect_data + "/" + os.path.basename(replay_dir) + "_" + ("temporal" if args.temporal else "spatial")
        os.makedirs(dataset_dir, exist_ok = True)

    return True, path_2_recording, path_2_waypoints, dataset_dir
    
def main(args):
    pygame.init()

    Logger.set_levels("INFO", "WARNING", "ERROR", "CUSTOM", "DEBUG" if args.debug else "INFO")

    lp = None

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    virt_world = World(client, args.traffic_port)
    virt_world.sync = args.sync
    virt_world.delta = args.delay
    virt_world.disable_render = True
    virt_world.apply_settings()

    rgb_sensor = RGB(virt_world.world)
    gnss_sensor = GNSS(virt_world.world, freq_hz = FREQ, mu_ms = MEAN_DELAY, sigma_ms = STDDEV_DELAY)
    gnss_sensor.set_attribute("noise_lat_stddev", LAT_STDDEV / 111320.0)
    gnss_sensor.set_attribute("noise_lon_stddev", LON_STDDEV / 111320.0)
    imu_sensor = IMU(virt_world.world)
    imu_sensor.set_attribute("noise_gyro_bias_x", 0.005)
    imu_sensor.set_attribute("noise_gyro_bias_y", 0.005)
    
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
        spawner.spawn_mass_vehicle(15, exclude = [VClass.Large, VClass.Tiny])
        spawner.spawn_single_vehicle(bp_id = "vehicle.dodge.charger_2020", exclude = [VClass.Large, VClass.Medium, VClass.Tiny], autopilot = False)
        controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
        viewer_args.update({"vehicle": controlling_vehicle})

    if args.mode == "replay":
        replay_dirs = _expand_replay_dirs(args.replay_dir)
        

        for idx, replay_dir in enumerate(replay_dirs,start = 1):
            
            # -- Load in the recordings as well as the ego state
            ret, path_2_recording, path_2_waypoints, dataset_dir = load_recording(
                client, virt_world, spawner, folder, replay_dir
            )
            if not ret:
                logger.ERROR(f"Failed to load log at {path_2_recording}")
                continue
            
            # Reinitialize sensors with new world
            rgb_sensor, gnss_sensor, imu_sensor = reinit_sensors(virt_world)

            # Read the recordings parameters, clean previous vehicle and play the recording
            duration = get_recording_duration(path_2_recording)
            client.show_recorder_file_info(path_2_recording, False)
            
            start_offset, stop_offset = 0, 4
            duration -= start_offset + stop_offset
            if duration <= 0:
                logger.ERROR(f"Skip replay (duration <= 0): {replay_dir}")
                continue
            spawner.despawn_vehicles()
            client.replay_file(path_2_recording, start_offset, duration, 0)
            
            
            # -- Tick for n times to check replay's stability
            logger.INFO(f"Waiting for replay to stabilize...")
            for _ in range(50):
                if virt_world.world.get_settings().synchronous_mode:
                    virt_world.world.tick()
                else:
                    time.sleep(0.05)
            
            # Spawn actor which is alive and kicking and add to the Vehicle object for controlling
            ego_actor = spawner.wait_for_live_actor("ego", timeout_s=30.0, settle_ticks=90)
            if ego_actor is None:
                logger.ERROR(f"Could not find live ego actor for replay: {replay_dir}")
                continue
            logger.INFO(f"Ego actor found and settled")
            spawner.single_vehicle = ego_actor
            controlling_vehicle = Vehicle(ego_actor, virt_world.world)

            # -- Arguments for viewer
            per_viewer_args = viewer_args | {"vehicle": controlling_vehicle, "duration": duration, "headless": args.headless}
            game_viewer = CarlaViewer(**per_viewer_args)
            
            sensors_metadata = {
                rgb_sensor     : None, 
                gnss_sensor    : None, 
                imu_sensor     : None, 
            }
            if not SensorSpawn.test_sensor(game_viewer, sensors_metadata):
                logger.ERROR(f"Skipping replay due to sensor initialization failure: {replay_dir}")
                continue

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
                scale = MAP_SCALE,
                relative_pos = "forward"
            )
            path_optimizer = OptimizePath(
                virt_world, 
                step = PATH_STEP, 
                exclude_circle = EXCLUDE_PARAMS,
            )
            game_viewer.attach_plugins(path_optimizer = path_optimizer)

            true_trajectories    = np.load(path_2_waypoints)
            midlane_trajectories = map_processor.precompute_waypoints(true_trajectories)
            replayer             = ReplayHandler(virt_world, true_trajectories, midlane_trajectories, dataset_dir, args.temporal, args.draw_waypoints if hasattr(args, 'draw_waypoints') else False)
            
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.completed:.1f}/{task.total:.1f}"),
                TimeRemainingColumn()
            )
            Logger.set_progress_console(progress.console)
            
            pbar = progress.add_task(f"Play duration ({idx}/{len(replay_dirs)})", total=round(duration, 2))
            game_viewer.attach_plugins(
                replayer = replayer, 
                pbar = (progress, pbar),  # Pass both progress and task_id
                map_processor = map_processor
            )
            lp.add_function(game_viewer.map_processor.retrieve_map)
            
            with progress:
                lp_wrapper()
            progress.stop()

            spawner.despawn_vehicles()
            time.sleep(1.0)

    if args.mode != "replay":
        game_viewer = CarlaViewer(**viewer_args)
        game_viewer.init_sensor({
            rgb_sensor     : None, 
            gnss_sensor    : None, 
            imu_sensor     : None, 
            # semseg_sensor  : None,
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
            scale = MAP_SCALE,
            relative_pos = "forward"
        )
        path_optimizer = OptimizePath(
            virt_world, 
            step = PATH_STEP, 
            exclude_circle = EXCLUDE_PARAMS,
        )
        game_viewer.attach_plugins(path_optimizer = path_optimizer)

    if args.mode == "manual":
        virt_world.tm.ignore_signs_percentage(controlling_vehicle.vehicle, args.ignore_signs)
        if args.record:
            # Get map name automatically
            map_name = virt_world.world.get_map().name.split("/")[-1]
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = f"{folder}/{args.record}/{map_name}/recording_{date}"
            os.makedirs(directory, exist_ok=True)
            client.start_recorder(f"{directory}/log.log", True)

            trajectory_logging = TrajectoryBuffer(directory, min_dt_s = MIN_SAVING_DIST)
            game_viewer.attach_plugins(
                traj_logger = trajectory_logging, 
                map_processor = map_processor
            )

            
            lp_wrapper()
            client.stop_recorder()
        else:
            game_viewer.attach_plugins(map_processor = map_processor)
            lp.add_function(game_viewer.run)
            lp.add_function(game_viewer.step_world)
            lp.add_function(map_processor.__init__)
            lp.add_function(path_optimizer._build_detailed_graph)
            lp_wrapper()

    if args.mode == "inference":
        inference = AsyncInference(args.model_path, device = 'cuda', batch_output = False)
        game_viewer.override_render_map = True
        game_viewer.attach_plugins(
            inference = inference, 
            map_processor = map_processor
        )
        lp_wrapper()

    if lp is not None:
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
        "--timeout",
        default = 10,
        type = float,
        help = "Set timeout for carla client"
    )
    argparser.add_argument(
        "--fps",
        default = 144,
        type = float,
        help = "Max fps for pygame rendering"
    )
    argparser.add_argument(
        "--debug",
        action = "store_true",
        help = "Set logger debugging flag"
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
        nargs = "+",
        help = "Replay Carla recording (.log file path is needed, recording time of .npy must correspond to .log)",   
        required = True
    )
    replay_parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporal (time-based) waypoint generation instead of spatial."
    )
    replay_parser.add_argument(
        "--draw-waypoints",
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
        "--draw-waypoints",
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