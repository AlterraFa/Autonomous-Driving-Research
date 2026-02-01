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
)
from config.enum import (
    VehicleClass as VClass,
    CarlaLabel as CLabel
)

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


def _wait_for_live_actor_by_role(world, role_name: str, timeout_s: float = 30.0, settle_ticks: int = 50):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        vehicles = world.get_actors().filter('vehicle.*')
        for v in vehicles:
            try:
                if v.attributes.get('role_name', '') == role_name and v.is_alive:
                    logger.INFO(f"Found {role_name} actor, settling...")
                    prev_pos = None
                    for tick_idx in range(settle_ticks):
                        if world.get_settings().synchronous_mode:
                            world.tick()
                        else:
                            time.sleep(0.05)
                        try:
                            curr_pos = v.get_location()
                            if prev_pos is not None and prev_pos.distance(curr_pos) > 0.1:
                                logger.WARNING(f"Actor still moving (tick {tick_idx}), continuing settle...")
                            prev_pos = curr_pos
                        except:
                            pass
                    logger.INFO(f"Actor settled after {settle_ticks} ticks")
                    return v
            except Exception:
                pass
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            time.sleep(0.05)
    return None


def _init_sensor_with_retry(game_viewer, sensors_metadata, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            game_viewer.init_sensor(sensors_metadata)
            return True
        except RuntimeError as e:
            if "parent actor not found" in str(e):
                if attempt < max_retries - 1:
                    logger.WARNING(f"Sensor init failed (attempt {attempt + 1}/{max_retries}): {e}")
                    wait_ticks = 20 + (attempt * 10)
                    logger.WARNING(f"Retrying after {wait_ticks} ticks...")
                    for _ in range(wait_ticks):
                        if game_viewer.virt_world.world.get_settings().synchronous_mode:
                            game_viewer.virt_world.world.tick()
                        else:
                            time.sleep(0.1)
                else:
                    logger.ERROR(f"Sensor init failed after {max_retries} attempts: {e}")
                    return False
            else:
                logger.ERROR(f"Sensor init error: {e}")
                return False
    return False

    
def main(args):
    pygame.init()

    lp = None

    client = carla.Client(args.host, args.port)
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
    semseg_sensor = SemanticSegmentation(
        virt_world.world, 
        filter_labels = [CLabel.Road], 
        binarize = True,
    )
    
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
        replay_dirs = _expand_replay_dirs(args.replay_dir)

        for idx, replay_dir in enumerate(replay_dirs, start = 1):
            path_2_recording = folder + "/" + replay_dir + "/log.log"
            path_2_waypoints = folder + "/" + replay_dir + "/trajectory.npy"

            if not os.path.exists(path_2_recording):
                logger.ERROR(f"Replay log not found: {path_2_recording}")
                continue
            if not os.path.exists(path_2_waypoints):
                logger.ERROR(f"Trajectory file not found: {path_2_waypoints}")
                continue

            if args.collect_data is None:
                dataset_dir = None
            else:
                dataset_dir = folder + "/" + args.collect_data + "/" + os.path.basename(replay_dir) + "_" + ("temporal" if args.temporal else "spatial")
                os.makedirs(dataset_dir, exist_ok = True)

            duration = get_recording_duration(path_2_recording)
            client.show_recorder_file_info(path_2_recording, False)
            start = 1.5; stop = 4
            duration -= start + stop
            if duration <= 0:
                logger.ERROR(f"Skip replay (duration <= 0): {replay_dir}")
                continue

            spawner.despawn_vehicles()
            logger.INFO(f"Waiting for replay system to stabilize...")
            for _ in range(50):
                if virt_world.world.get_settings().synchronous_mode:
                    virt_world.world.tick()
                else:
                    time.sleep(0.05)
            client.replay_file(path_2_recording, start, duration, 0)

            ego_actor = _wait_for_live_actor_by_role(virt_world.world, "ego", timeout_s=30.0, settle_ticks=100)
            if ego_actor is None:
                logger.ERROR(f"Could not find live ego actor for replay: {replay_dir}")
                continue
            else:
                logger.INFO(f"Replaying actor spawn")
            spawner.single_vehicle = ego_actor
            controlling_vehicle = Vehicle(ego_actor, virt_world.world)

            per_viewer_args = viewer_args | {"vehicle": controlling_vehicle, "duration": duration, "headless": args.headless}

            game_viewer = CarlaViewer(**per_viewer_args)
            
            # Refresh ego actor immediately before sensor init
            ego_actor_refresh = _wait_for_live_actor_by_role(virt_world.world, "ego", timeout_s=5.0, settle_ticks=10)
            if ego_actor_refresh is None:
                logger.ERROR(f"Ego actor became invalid before sensor init: {replay_dir}")
                continue
            game_viewer.vehicle = ego_actor_refresh
            game_viewer.virt_vehicle.vehicle = ego_actor_refresh
            sensors_metadata = {
                rgb_sensor     : None, 
                gnss_sensor    : None, 
                imu_sensor     : None, 
                semseg_sensor  : None,
            }
            if not _init_sensor_with_retry(game_viewer, sensors_metadata):
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
            replayer             = ReplayHandler(virt_world, true_trajectories, midlane_trajectories, dataset_dir, args.temporal, args.debug)
            pbar                 = tqdm(total = round(duration, 2), unit = 'server second', desc = f"Play duration ({idx}/{len(replay_dirs)})", ncols = 125, leave = True)
            game_viewer.attach_plugins(
                replayer = replayer, 
                pbar = pbar, 
                map_processor = map_processor
            )
            lp.add_function(game_viewer.map_processor.retrieve_map)
            lp_wrapper()
            pbar.close()

            spawner.despawn_vehicles()
            time.sleep(1.0)

    if args.mode != "replay":
        game_viewer = CarlaViewer(**viewer_args)
        game_viewer.init_sensor({
            rgb_sensor     : None, 
            gnss_sensor    : None, 
            imu_sensor     : None, 
            semseg_sensor  : None,
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