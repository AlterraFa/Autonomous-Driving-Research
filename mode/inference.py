import line_profiler

from src.control.vehicle_control import Vehicle
from src.render.viewer import VIEWER_REGISTRY
from model.inference import AsyncInference
from config.enum import VehicleClass as VClass

from .utils import (
    num_npc,
    make_map_and_optimizer,
)


def run_inference(args, client, virt_world, sensors, spawner, folder, viewer_args):
    rgb_sensor, gnss_sensor, imu_sensor = sensors

    spawner.spawn_mass_vehicle(num_npc, exclude=[VClass.Large, VClass.Tiny])
    spawner.spawn_single_vehicle(
        bp_id     = "vehicle.dodge.charger_2020",
        exclude   = [VClass.Large, VClass.Medium, VClass.Tiny],
        autopilot = False,
    )
    controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)

    game_viewer = VIEWER_REGISTRY["inference"](**{**viewer_args, "vehicle": controlling_vehicle})
    game_viewer.init_sensor({rgb_sensor: None, gnss_sensor: None, imu_sensor: None})

    lp = line_profiler.LineProfiler()
    lp.add_function(game_viewer.run)
    lp.add_function(game_viewer.step_world)
    lp_wrapper = lp(game_viewer.run)

    map_processor, path_optimizer = make_map_and_optimizer(virt_world)
    game_viewer.attach_plugins(path_optimizer=path_optimizer)

    game_viewer.override_render_map = True
    game_viewer.attach_plugins(
        inference     = AsyncInference(args.model_path, device="cuda", batch_output=False),
        map_processor = map_processor,
    )
    lp_wrapper()

    return lp
