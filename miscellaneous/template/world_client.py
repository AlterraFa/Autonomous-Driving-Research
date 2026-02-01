#!/home/alterraonix/miniconda/envs/Core/bin/python
import carla

client = carla.Client("localhost", 2000)
client.load_world("Town02")

world  = client.get_world()

level  = world.get_map()
weather = world.get_weather()
blueprint = world.get_blueprint_library()