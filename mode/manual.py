import os
import datetime
import line_profiler

from utils.messages.logger import Logger
from utils.control.vehicle_control import Vehicle
from utils.others.data_processor import TrajectoryBuffer
from utils.render.viewer import VIEWER_REGISTRY
from config.enum import VehicleClass as VClass

from .utils import (
    logger,
    num_npc,
    MIN_SAVING_DIST,
    make_map_and_optimizer,
)


def run_manual(args, client, virt_world, sensors, spawner, folder, viewer_args):
    rgb_sensor, gnss_sensor, imu_sensor = sensors

    spawner.spawn_mass_vehicle(num_npc, exclude=[VClass.Large, VClass.Tiny])
    spawner.spawn_single_vehicle(
        bp_id    = "vehicle.dodge.charger_2020",
        exclude  = [VClass.Large, VClass.Medium, VClass.Tiny],
        autopilot = False,
    )
    controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)

    game_viewer = VIEWER_REGISTRY["manual"](**{**viewer_args, "vehicle": controlling_vehicle})
    game_viewer.init_sensor({rgb_sensor: None, gnss_sensor: None, imu_sensor: None})

    lp = line_profiler.LineProfiler()
    lp.add_function(game_viewer.run)
    lp.add_function(game_viewer.step_world)
    lp_wrapper = lp(game_viewer.run)

    map_processor, path_optimizer = make_map_and_optimizer(virt_world)
    game_viewer.attach_plugins(path_optimizer=path_optimizer)

    virt_world.tm.ignore_signs_percentage(controlling_vehicle.vehicle, args.ignore_signs)

    if args.record:
        map_name  = virt_world.world.get_map().name.split("/")[-1]
        date      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = f"{folder}/{args.record}/{map_name}/recording_{date}"
        os.makedirs(directory, exist_ok=True)
        client.start_recorder(f"{directory}/log.log", True)

        game_viewer.attach_plugins(
            traj_logger   = TrajectoryBuffer(directory, min_dt_s=MIN_SAVING_DIST),
            map_processor = map_processor,
        )
        lp_wrapper()
        client.stop_recorder()
    else:
        game_viewer.attach_plugins(map_processor=map_processor)
        lp.add_function(map_processor.__init__)
        lp.add_function(path_optimizer._build_detailed_graph)
        lp_wrapper()

    return lp
