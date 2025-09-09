import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)

import re
import carla
import argparse
import importlib
import pygame
import torch
from functools import partial

from utils.spawn.actor_spawner import Spawn, VehicleClass as VClass
from utils.spawn.sensor_spawner import (
    SemanticSegmentation as SemSeg, 
    RGB,
    GNSS,
    IMU, 
    Depth,
    CarlaLabel as Clabel
)

from utils.control.world import World
from utils.control.vehicle_control import Vehicle, wait_for_actor_by_role
from utils.control.viewer import CarlaViewer

def load_model_from_checkpoint(path: str, device: str = "cpu", **model_kwargs):
    """
    Load a model checkpoint, automatically inferring class name and module path from filename.

    Args:
        path (str): Path to checkpoint file (e.g. 'model/PilotNet/best_PilotNetStatic_run1.pt')
        device (str): Device to load model onto ('cpu' or 'cuda')
        **model_kwargs: Extra arguments to pass to the model constructor (needed if state_dict only)

    Returns:
        torch.nn.Module: Loaded model
    """
    fname = os.path.basename(path)
    match = re.search(r"best_(.+?)_run\d+\.pt", fname)
    if not match:
        raise ValueError(f"Could not parse class name from filename: {fname}")
    class_name = match.group(1)

    dir_path = os.path.dirname(path)           # "model/PilotNet"
    module_path = dir_path.replace("/", ".")   # "model.PilotNet"
    module_path = module_path + ".model"       # append ".model"

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    try:
        state_dict = torch.load(path, map_location=device)
        if isinstance(state_dict, dict):
            model = cls(**model_kwargs)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            return model
    except Exception:
        torch.serialization.add_safe_globals([cls])
        model = torch.load(path, weights_only=False, map_location=device)
        model.to(device).eval()
        return model

    
def main(args):
    pygame.init()

    model = load_model_from_checkpoint(args.model_path, device = "cuda")
    model = torch.compile(model).eval().to(next(model.parameters()).device)

    client = carla.Client(args.host, args.port)
    virt_world = World(client, args.traffic_port)
    virt_world.sync = args.sync
    virt_world.delta = args.delay
    virt_world.disable_render = True
    virt_world.apply_settings()

    spawner = Spawn(virt_world.world, virt_world.tm)
    spawner.destroy_all_vehicles()
    spawner.spawn_mass_vehicle(6, exclude = [VClass.Large, VClass.Tiny])
    spawner.spawn_single_vehicle(bp_id = "vehicle.dodge.charger_2020", exclude = [VClass.Large, VClass.Medium, VClass.Tiny], autopilot = False)

    rgb_sensor      = RGB(virt_world.world)
    semantic_sensor = SemSeg(virt_world.world, convert_to = partial(SemSeg.SemanticData.to_image))
    gnss_sensor     = GNSS(virt_world.world)
    imu_sensor      = IMU(virt_world.world)
    depth_sensor    = Depth(virt_world.world, convert_to = partial(Depth.DepthMap.to_log, invert = False, max_depth = 100))
    
    controlling_vehicle = Vehicle(spawner.single_vehicle, virt_world.world)
    
    game_viewer = CarlaViewer(virt_world, controlling_vehicle, args.width, args.height, sync = args.sync)
    game_viewer.init_sensor([rgb_sensor, semantic_sensor, gnss_sensor, imu_sensor, depth_sensor])
    game_viewer.run(model = model)
    
    
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
        "--traffic-port",
        metavar = "TMP",
        default = 8000,
        type = int,
        help = "Traffic manager port for actor autopilot function"
    )
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
        "--delay",
        default = 0.05,
        type = float,
        help = "Max fps for synchronize running"
    )
    argparser.add_argument(
        "--debug",
        action = "store_true",
        help = "Draw debugging waypoints onto the world"
    )
    argparser.add_argument(
        "--model-path",
        type = str,
        help = "Path to models file as well as its class reference",
        required = True
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)