import carla

client = carla.Client("localhost", 2000)
client.load_world("BFMC_Road_Building")

world  = client.get_world()

level  = world.get_map()
weather = world.get_weather()
blueprint = world.get_blueprint_library()