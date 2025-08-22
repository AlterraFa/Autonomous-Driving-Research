import carla
import numpy as np

class Vehicle:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        self.vehicle = vehicle
        self.world = world