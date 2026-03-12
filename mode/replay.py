import time
import numpy as np
import line_profiler

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.messages.logger import Logger
from src.spawn.sensor_spawner import SensorSpawn
from src.control.vehicle_control import Vehicle
from src.math.path import ReplayHandler
from src.render.viewer import VIEWER_REGISTRY

from .utils import (
    logger,
    start_at, stop_at,
    get_recording_duration,
    reinit_sensors,
    load_recording,
    expand_replay_dirs,
    make_map_and_optimizer,
)


def run_replay(args, client, virt_world, sensors, spawner, folder, viewer_args):
    global start_at, stop_at
    rgb_sensor, gnss_sensor, imu_sensor = sensors
    replay_dirs = expand_replay_dirs(args.replay_dir)
    lp = None

    for idx, replay_dir in enumerate(replay_dirs, start=1):

        # -- Load recordings and ego state
        ret, path_2_recording, path_2_waypoints, dataset_dir = load_recording(
            args, client, virt_world, spawner, folder, replay_dir
        )
        if not ret:
            logger.ERROR(f"Failed to load log at {path_2_recording}")
            continue

        rgb_sensor, gnss_sensor, imu_sensor = reinit_sensors(
            virt_world, rgb_sensor, gnss_sensor, imu_sensor
        )

        full_duration = get_recording_duration(path_2_recording)
        client.show_recorder_file_info(path_2_recording, False)

        if stop_at < 0:
            stop_at = full_duration
        if full_duration <= 0:
            logger.ERROR(f"Skip replay (duration <= 0): {replay_dir}")
            continue
        actual_duration = stop_at - start_at

        spawner.despawn_vehicles()
        client.replay_file(path_2_recording, start_at, actual_duration + 10, 0)
        logger.INFO(f"Showing a replay segment of {actual_duration: .2f}s")

        # -- Tick for n times to check replay's stability
        logger.INFO("Waiting for replay to stabilize...")
        for _ in range(50):
            if virt_world.world.get_settings().synchronous_mode:
                virt_world.world.tick()
            else:
                time.sleep(0.05)

        # Spawn actor which is alive and kicking and add to the Vehicle object for controlling
        ego_actor = spawner.wait_for_live_actor("ego", timeout_s=30.0, settle_ticks=30)
        if ego_actor is None:
            logger.ERROR(f"Could not find live ego actor for replay: {replay_dir}")
            continue
        logger.INFO("Ego actor found and settled")
        spawner.single_vehicle = ego_actor
        controlling_vehicle = Vehicle(ego_actor, virt_world.world)

        per_viewer_args = viewer_args | {
            "vehicle"  : controlling_vehicle,
            "duration" : actual_duration,
            "headless" : args.headless,
        }
        game_viewer = VIEWER_REGISTRY["replay"](**per_viewer_args)

        sensors_metadata = {rgb_sensor: None, gnss_sensor: None, imu_sensor: None}
        if not SensorSpawn.test_sensor(game_viewer, sensors_metadata):
            logger.ERROR(f"Skipping replay due to sensor initialization failure: {replay_dir}")
            continue

        lp = line_profiler.LineProfiler()
        lp.add_function(game_viewer.run)
        lp.add_function(game_viewer.step_world)
        lp_wrapper = lp(game_viewer.run)

        map_processor, path_optimizer = make_map_and_optimizer(virt_world)
        game_viewer.attach_plugins(path_optimizer=path_optimizer)

        true_trajectories    = np.load(path_2_waypoints)
        midlane_trajectories = map_processor.precompute_waypoints(true_trajectories)
        replayer             = ReplayHandler(
            virt_world, true_trajectories, midlane_trajectories,
            dataset_dir, args.temporal,
            args.draw_waypoints if hasattr(args, "draw_waypoints") else False,
        )

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed:.1f}/{task.total:.1f}"),
            TimeRemainingColumn(),
        )
        Logger.set_progress_console(progress.console)

        pbar = progress.add_task(
            f"Play duration ({idx}/{len(replay_dirs)})",
            total=round(actual_duration, 2),
        )
        game_viewer.attach_plugins(
            replayer      = replayer,
            pbar          = (progress, pbar),
            map_processor = map_processor,
        )
        lp.add_function(game_viewer.map_processor.retrieve_map)

        with progress:
            lp_wrapper()
        progress.stop()

        spawner.despawn_vehicles()
        time.sleep(1.0)

    return lp
