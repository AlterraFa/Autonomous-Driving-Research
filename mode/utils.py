import os
import sys
import re
import time

import carla
import numpy as np
from config import CONFIG

from src.messages.logger import Logger
from src.spawn.sensor_spawner import GNSS, IMU
from src.math.world_map import Map
from src.math.path import OptimizePath

logger = Logger("Global")

# Map rendering
RECT_DIM   = CONFIG.map_render.rect_dim
MAP_OFFSET = CONFIG.map_render.map_offset
MAP_RANGE  = CONFIG.map_render.map_range
MAP_RESIZE = CONFIG.map_render.map_resize
MAP_SCALE  = CONFIG.map_render.map_scale

# Path optimiser
PATH_STEP      = CONFIG.path_optimizer.path_step
EXCLUDE_PARAMS = CONFIG.path_optimizer.exclude_params

# GNSS / GPS
MEAN_DELAY   = CONFIG.gps.mean_delay
STDDEV_DELAY = CONFIG.gps.stddev_delay
LAT_STDDEV   = CONFIG.gps.lat_stddev
LON_STDDEV   = CONFIG.gps.lon_stddev
FREQ         = CONFIG.gps.frequency

# Replay
start_at    = CONFIG.replay.start_at
stop_at     = CONFIG.replay.stop_at

# Spawn
num_npc    = CONFIG.spawn.num_npc

# Misc
MIN_SAVING_DIST = 0.4


# ── Shared helpers ─────────────────────────────────────────────────────────────

def get_recording_duration(client, log_path: str) -> float:
    """Returns the recording duration in seconds for a CARLA .log file."""
    report = client.show_recorder_file_info(log_path, False)
    m = re.search(r"Duration:\s*([0-9.]+)\s*seconds", report)
    if m:
        duration = float(m.group(1))
        logger.INFO(f"Recording duration: {duration:.2f} seconds")
    else:
        print("No duration found")
    return duration


def refresh_world_references(client, virt_world, spawner=None):
    # 1. Force a reload of the world object
    virt_world.refresh()
    
    # 2. Wait for the Map to be actually accessible
    retries = 10
    while retries > 0:
        try:
            m = virt_world.world.get_map()
            if m.name != "": break
        except:
            time.sleep(0.5)
            retries -= 1
            
    virt_world.apply_settings()
    if spawner is not None:
        spawner.world = virt_world.world
    
    logger.CUSTOM("SUCCESS", f"World references refreshed for map: {virt_world.world.get_map().name}")


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

    current_map = virt_world.world.get_map().name.split("/")[-1]
    if map_name not in current_map:
        logger.INFO(f"Loading map: {map_name}")
        virt_world.switch_map(map_name)
        refresh_world_references(client, virt_world, spawner)

    logger.INFO("Stabilizing world after map load...")
    for _ in range(60):
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
