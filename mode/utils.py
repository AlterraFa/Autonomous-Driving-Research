import os
import sys
import re
import time

import toml
import carla
import numpy as np

from src.messages.logger import Logger
from src.spawn.sensor_spawner import GNSS, IMU
from src.math.world_map import Map
from src.math.path import OptimizePath

logger = Logger()

# ── Root & config ──────────────────────────────────────────────────────────────
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
conf  = toml.load(os.path.join(_root, "config/config.toml"))

# Map rendering
map_conf   = conf.get("MapRender", {})
RECT_DIM   = tuple(map_conf.get("rect_dim",   [4, 3]))
MAP_OFFSET = tuple(map_conf.get("map_offset", [100, 100]))
MAP_RANGE  = tuple(map_conf.get("map_range",  [50, 50]))
MAP_RESIZE = tuple(map_conf.get("map_resize", [200, 200]))
MAP_SCALE  = map_conf.get("map_scale", 3)

# Path optimiser
path_optim_conf = conf.get("PathOptimizer", {})
PATH_STEP      = path_optim_conf.get("path_step", 2.0)
EXCLUDE_PARAMS = path_optim_conf.get("exclude_params", [0, 0, 0])

# GNSS / GPS
gnss_conf    = conf.get("GPS", {})
MEAN_DELAY   = gnss_conf.get("mean_delay",   0)
STDDEV_DELAY = gnss_conf.get("stddev_delay", 0)
LAT_STDDEV   = gnss_conf.get("lat_stddev",   0)
LON_STDDEV   = gnss_conf.get("lon_stddev",   0)
FREQ         = gnss_conf.get("frequency",    50)

# Replay
replay_conf = conf.get("Replay", {})
start_at    = replay_conf.get("start_at",  0)
stop_at     = replay_conf.get("stop_at",  -1)

# Spawn
spawn_conf = conf.get("Spawn", {})
num_npc    = spawn_conf.get("num_npc", 0)

# Misc
MIN_SAVING_DIST = 0.4


# ── Shared helpers ─────────────────────────────────────────────────────────────

def get_recording_duration(log_path: str) -> float:
    """Returns the recording duration in seconds for a CARLA .log file."""
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


def refresh_world_references(client, virt_world, spawner=None):
    """
    Refresh world references after loading a new map.
    Updates virt_world.world, reapplies settings, and updates spawner.world if provided.
    """
    virt_world.world = client.get_world()
    virt_world.apply_settings()

    if spawner is not None:
        spawner.world = virt_world.world

    logger.INFO("World references refreshed")


def _copy_blueprint_attributes(source_bp, target_bp):
    if source_bp is None or target_bp is None:
        return

    try:
        for attr in source_bp:
            try:
                target_bp.set_attribute(attr.id, source_bp.get_attribute(attr.id).as_str())
            except Exception:
                continue
    except TypeError:
        return


def _recreate_sensor(world, sensor):
    if isinstance(sensor, GNSS):
        new_sensor = sensor.__class__(
            world,
            freq_hz=1.0 / sensor.sample_interval if sensor.sample_interval > 0 else FREQ,
            mu_ms=sensor.mu * 1000.0,
            sigma_ms=sensor.sigma * 1000.0,
        )
    else:
        new_sensor = sensor.__class__(world)

    _copy_blueprint_attributes(sensor.sensor_bp, new_sensor.sensor_bp)
    return new_sensor


def reinit_sensors(virt_world, *sensors):
    """Recreate sensors for the updated world and return them in order."""
    if not sensors:
        logger.WARNING("No sensors provided for reinitialization")
        return tuple()

    world = virt_world.world
    new_sensors = tuple(_recreate_sensor(world, sensor) for sensor in sensors)

    logger.INFO("Sensors reinitialized with new world")
    return new_sensors


def load_recording(args, client, virt_world, spawner, folder, replay_dir):
    map_name = replay_dir.split("/")[-2]
    logger.INFO(f"Loading map: {map_name}")
    client.load_world(map_name)

    refresh_world_references(client, virt_world, spawner)

    logger.INFO("Stabilizing world after map load...")
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
        dataset_dir = (
            folder + "/" + args.collect_data + "/" +
            os.path.basename(replay_dir) + "_" +
            ("temporal" if args.temporal else "spatial")
        )
        os.makedirs(dataset_dir, exist_ok=True)

    return True, path_2_recording, path_2_waypoints, dataset_dir


def expand_replay_dirs(replay_dirs):
    if isinstance(replay_dirs, str):
        replay_dirs = [replay_dirs]
    expanded = []
    for item in replay_dirs:
        if "," in item:
            expanded.extend([p.strip() for p in item.split(",") if p.strip()])
        else:
            expanded.append(item)
    return expanded


def make_map_and_optimizer(virt_world):
    map_processor = Map(
        world        = virt_world,
        rect_dim     = RECT_DIM,
        map_offset   = MAP_OFFSET,
        range_       = MAP_RANGE,
        resize_to    = MAP_RESIZE,
        scale        = MAP_SCALE,
        relative_pos = "forward",
    )
    path_optimizer = OptimizePath(
        virt_world,
        step           = PATH_STEP,
        exclude_circle = EXCLUDE_PARAMS,
    )
    return map_processor, path_optimizer
