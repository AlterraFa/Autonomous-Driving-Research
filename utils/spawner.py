import carla
import random

from rich import print
from typing import Literal

Vehicle_BP = Literal[
 'vehicle.audi.a2', 'vehicle.citroen.c3', 'vehicle.chevrolet.impala',
 'vehicle.dodge.charger_police_2020', 'vehicle.micro.microlino',
 'vehicle.dodge.charger_police', 'vehicle.audi.tt',
 'vehicle.jeep.wrangler_rubicon', 'vehicle.mercedes.coupe',
 'vehicle.yamaha.yzf', 'vehicle.mercedes.coupe_2020',
 'vehicle.harley-davidson.low_rider', 'vehicle.dodge.charger_2020',
 'vehicle.ford.ambulance', 'vehicle.lincoln.mkz_2020',
 'vehicle.mini.cooper_s_2021', 'vehicle.ford.crown', 'vehicle.toyota.prius',
 'vehicle.carlamotors.european_hgv', 'vehicle.carlamotors.carlacola',
 'vehicle.vespa.zx125', 'vehicle.nissan.patrol_2021',
 'vehicle.mercedes.sprinter', 'vehicle.audi.etron', 'vehicle.seat.leon',
 'vehicle.volkswagen.t2_2021', 'vehicle.tesla.cybertruck',
 'vehicle.lincoln.mkz_2017', 'vehicle.carlamotors.firetruck',
 'vehicle.ford.mustang', 'vehicle.volkswagen.t2',
 'vehicle.mitsubishi.fusorosa', 'vehicle.tesla.model3',
 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets',
 'vehicle.bmw.grandtourer', 'vehicle.bh.crossbike', 'vehicle.kawasaki.ninja',
 'vehicle.nissan.patrol', 'vehicle.nissan.micra', 'vehicle.mini.cooper_s'
]

class Spawn:
    def __init__(self, world: carla.Client, traffic_manager: carla.Client):
        self.blueprints    = world.get_blueprint_library()
        self.vehicle_spawn_pts = world.get_map().get_spawn_points()

        self.world = world
        self.traffic_manager = traffic_manager
        self.tm_port = self.traffic_manager.get_port()

        self.vehicles = []; self.walkers = []; self.sensors = []

    def spawn_mass_vehicle(self, size: int, transform: carla.Transform = None, autopilot = True):
        if size < 0:
            print(f"[red][ERROR][/]: Number of spawning vehicles must be a positive integer")
            exit(12)
        if transform is not None:
            if size != len(transform):
                print(f"[red][ERROR][/]: Number of spawning location must equal to spawning vehicles")
                exit(12)
        
        vehicle_bp = self.blueprints.filter("*vehicle*")
        for _ in range(size):
            vehicle = self.world.try_spawn_actor(random.choice(vehicle_bp), random.choice(self.vehicle_spawn_pts))
            if vehicle:
                vehicle.set_autopilot(autopilot)
                self.vehicles += [vehicle]
        
        self.world.tick()
        print(f"[green][INFO][/]: Spawned mass successfully. {self.get_size} vehicles in environment")
        
        
    def spawn_single_vehicle(self, bp_id: Vehicle_BP = None, transform: carla.Transform = None, autopilot = True):
        if bp_id is None:
            while True:
                vehicle_name = random.choice(list(Vehicle_BP.__args__))
                try:
                    vehicle_bp = self.blueprints.find(random.choice(list(Vehicle_BP.__args__)))
                    break
                except: 
                    print(f"[yellow][WARNING][/]: Did not find {vehicle_name}. Retrying ...")
                    continue
        else:
            vehicle_bp = self.blueprints.find(bp_id)

        self.single_vehicle = self.world.try_spawn_actor(vehicle_bp, transform if transform is not None else random.choice(self.vehicle_spawn_pts))        
        while self.single_vehicle is None:
            self.single_vehicle = self.world.try_spawn_actor(vehicle_bp, transform if transform is not None else random.choice(self.vehicle_spawn_pts))        
        
        self.single_vehicle.set_autopilot(autopilot)
        self.vehicles += [self.single_vehicle]

        self.world.tick()
        print(f"[green][INFO][/]: Spawned single successfully. {self.get_size} vehicles in environment")
        
    def destroy_vehicle(self):
        for vehicle in self.get_vehicles:
            vehicle.destroy()
        print("[green][INFO][/]: Destroyed all vehicles")

    @property
    def get_size(self):
        return len(self.get_vehicles)

    @property
    def get_vehicles(self):
        return self.world.get_actors().filter("*vehicle*")