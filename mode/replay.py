import os, sys
import time
import numpy as np
import line_profiler
import carla
from config import CONFIG

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from config.enum import CameraView
from src.messages.logger import Logger
from src.messages.message_handler import MessageSender
from src.messages.all_messages import ClearNPCs
from src.others.data_processor import TrajectoryBuffer
from src.spawn.sensor_spawner import SensorSpawn
from src.control.vehicle_control import Vehicle
from src.others.data_processor import ReplayHandler
from src.render.viewer import VIEWER_REGISTRY
from src.math import ContractingWP

from .utils import (
    logger,
    start_at, stop_at,
    MIN_SAVING_DIST,
    get_recording_duration,
    load_recording,
    expand_replay_dirs,
    make_map_and_optimizer,
)

from src.spawn.actor_spawner import Spawn
from src.spawn.sensor_spawner import RGB, GNSS, IMU
from mode.utils import FREQ, MEAN_DELAY, STDDEV_DELAY, LAT_STDDEV, LON_STDDEV

temp_stop = stop_at

def run_replay(args, client: carla.Client, virt_world, folder, viewer_args):
    global start_at, stop_at, temp_stop
    replay_dirs = expand_replay_dirs(args.replay_dir)
    lp = None

    spawner = Spawn(virt_world.world)
    spawner.despawn_vehicles()

    for idx, replay_dir in enumerate(replay_dirs, start=1):

        # -- Load recordings and ego state
        ret, path_2_recording, path_2_waypoints, dataset_dir = load_recording(
            args, client, virt_world, spawner, folder, replay_dir
        )
        if not ret:
            logger.ERROR(f"Failed to load log at {path_2_recording}")
            continue
        

        full_duration = get_recording_duration(client, path_2_recording)
        client.show_recorder_file_info(path_2_recording, False)

        if temp_stop < 0:
            temp_stop = full_duration
        if full_duration <= 0:
            logger.ERROR(f"Skip replay (duration <= 0): {replay_dir}")
            continue
        actual_duration = temp_stop - start_at

        spawner.despawn_vehicles()
        client.replay_file(
            path_2_recording,
            start_at,
            actual_duration + CONFIG.replay_runtime.duration_padding_s,
            0,
        )
        logger.INFO(f"Showing a replay segment of {actual_duration: .2f}s for log: {replay_dir}")

        # -- Tick for n times to check replay's stability
        logger.INFO("Waiting for replay to stabilize...")
        for _ in range(CONFIG.replay_runtime.stability_wait_iterations):
            if virt_world.world.get_settings().synchronous_mode:
                virt_world.world.tick()
            else:
                time.sleep(CONFIG.replay_runtime.stability_sleep_s)

        # Spawn actor which is alive and kicking and add to the Vehicle object for controlling
        ego_actor = spawner.wait_for_live_actor(
            "ego",
            timeout_s=CONFIG.replay_runtime.actor_spawn_timeout_s,
            settle_ticks=CONFIG.replay_runtime.actor_settle_ticks,
        )
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

        # -- Init sensors
        rgb_sensor  = RGB(virt_world.world)
        gnss_sensor = GNSS(virt_world.world, freq_hz=FREQ, mu_ms=MEAN_DELAY, sigma_ms=STDDEV_DELAY)
        gnss_sensor.set_attribute("noise_lat_stddev", LAT_STDDEV / CONFIG.gps.meters_per_degree)
        gnss_sensor.set_attribute("noise_lon_stddev", LON_STDDEV / CONFIG.gps.meters_per_degree)
        imu_sensor  = IMU(virt_world.world)
        imu_sensor.set_attribute("noise_gyro_bias_x", CONFIG.sensor.imu_gyro_bias_x)
        imu_sensor.set_attribute("noise_gyro_bias_y", CONFIG.sensor.imu_gyro_bias_y)

        sensors_metadata = {
            rgb_sensor: [CameraView.FIRST_PERSON.value, True], 
            gnss_sensor: [None, True], 
            imu_sensor: [None, True]
        }
        if args.clear_npcs:
            sensors_metadata = sensors_metadata | {RGB(virt_world.world): [None, False]}
        if not SensorSpawn.test_sensor(game_viewer, sensors_metadata):
            logger.ERROR(f"Skipping replay due to sensor initialization failure: {replay_dir}")
            continue

        map_processor, path_optimizer = make_map_and_optimizer(virt_world)
        game_viewer.attach_plugins(path_optimizer=path_optimizer)

        if args.redo_traj:
            traj_logger = TrajectoryBuffer(replay_dir, min_dt_s = MIN_SAVING_DIST)
            replayer = None
            contracting_wp = None
        else: 
            traj_logger = None
            recorded_traj = np.load(path_2_waypoints)
            map_processor.register_wp(recorded_traj)
            replayer = ReplayHandler(
                virt_world, recorded_traj,
                dataset_dir, args.temporal,
                debug = args.draw_waypoints if hasattr(args, "draw_waypoints") else False,
            )
            contracting_wp = ContractingWP(
                world=virt_world.world,
                ego_vehicle=ego_actor,
                containment_mode="circle",
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
            replayer       = replayer,
            pbar           = (progress, pbar),
            map_processor  = map_processor,
            traj_logger    = traj_logger,
            contracting_wp = contracting_wp
        )


        lp = line_profiler.LineProfiler()
        lp.add_function(game_viewer.run)
        lp.add_function(game_viewer.step_world)
        lp.add_function(game_viewer.map_processor.retrieve_map)
        if args.redo_traj == False:
            lp.add_function(game_viewer.map_processor.path_handler.waypoints)
            lp.add_function(game_viewer.replayer.step)
            lp.add_function(game_viewer.replayer.path_handler.project)
            lp.add_function(game_viewer.replayer.path_handler.update_state)
        lp_wrapper = lp(game_viewer.run)

        with progress:
            send_clear_npcs = MessageSender(ClearNPCs)
            send_clear_npcs.send(args.clear_npcs)
            lp_wrapper()
        progress.stop()
        logger.CUSTOM("SUCCESS", "Stopped playing for log: {}", replay_dir)

        time.sleep(CONFIG.replay_runtime.final_wait_s)
        temp_stop = stop_at

    return lp
