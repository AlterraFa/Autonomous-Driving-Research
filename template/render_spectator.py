#!/home/alterraonix/miniconda/envs/Core/bin/python
import carla
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--disable", action = "store_true")
args = parser.parse_args()

client = carla.Client("localhost", 2000)
world  = client.get_world()
settings = world.get_settings()

if __name__ == "__main__":

    settings.no_rendering_mode = args.disable
    world.apply_settings(settings)