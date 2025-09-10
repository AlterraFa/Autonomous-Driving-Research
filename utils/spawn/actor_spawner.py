import carla
import random

from rich import print
from typing import Literal
from config.enum import VehicleClass

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

    def spawn_mass_vehicle(self, size: int, transform: carla.Transform = None, autopilot = True, exclude: list[VehicleClass] = None):
        if size < 0:
            print(f"[[red]ERROR[/] [purple]({self.__class__.__name__})[/]]: Number of spawning vehicles must be a positive integer")
            exit(12)
        if transform is not None:
            if size != len(transform):
                print(f"[[red]ERROR[/] [purple]({self.__class__.__name__})[/]]: Number of spawning location must equal to spawning vehicles")
                exit(12)
        
        vehicle_bp = self.blueprints.filter("*vehicle*")
        flattened_exclude = list(self.flatten(exclude)) if exclude is not None else []
        for _ in range(size):
            choosen_bp = random.choice(vehicle_bp)
            while True and exclude is not None:
                if str(choosen_bp.id) not in flattened_exclude: break
                choosen_bp = random.choice(vehicle_bp)


            vehicle = self.world.try_spawn_actor(choosen_bp, random.choice(self.vehicle_spawn_pts))
            if vehicle:
                vehicle.set_autopilot(autopilot)
                self.vehicles += [vehicle]
            else:
                print(f"[[red]ERROR[/] [purple]({self.__class__.__name__})[/]]: A random vehicle has not spawned successfully. Skipping.")
        
        self.world.tick()
        print(f"[[green]INFO[/] [purple]({self.__class__.__name__})[/]]: Spawned mass successfully. {self.get_size} vehicles in environment")
        
        
    def spawn_single_vehicle(self, bp_id: Vehicle_BP = None, name: str = "ego", transform: carla.Transform = None, random_offset: float = 0, autopilot = True, exclude: Vehicle_BP = None):
        if bp_id is None:
            flattened_exclude = list(self.flatten(exclude)) if exclude is not None else []
            while True:
                vehicle_name = random.choice(list(Vehicle_BP.__args__))
                if vehicle_name in flattened_exclude:
                    continue
                try:
                    vehicle_bp = self.blueprints.find(vehicle_name)
                    break
                except: 
                    print(f"[[red]ERROR[/] [purple]({self.__class__.__name__})[/]]: Did not find {vehicle_name}. Retrying ...")
                    continue
        else:
            print("[[green]INFO[/] [purple]({self.__class__.__name__})[/]]: A specific blueprint was choosen, skipping exclude")
            vehicle_bp = self.blueprints.find(bp_id)
            
        vehicle_bp.set_attribute("role_name", name)  
        

        while True:
            spawn_pts = random.choice(self.vehicle_spawn_pts)
            random_yaw = random.uniform(-random_offset, random_offset)
            spawn_pts.rotation.yaw += random_yaw
            if random_offset != 0: 
                print(f"[[yellow]WARNING[/] [purple]({self.__class__.__name__})[/]]: Random yaw selected: {random_yaw:.2f} degree.")
            
            self.single_vehicle = self.world.try_spawn_actor(vehicle_bp, transform if transform is not None else spawn_pts)

            if self.single_vehicle is not None: break
            else: print(f"[[red]ERROR[/] [purple]({self.__class__.__name__})[/]]: Did not find a suitable spawn point for controlling vehicle. Retrying ...")

        self.single_vehicle.set_autopilot(autopilot)
        self.vehicles += [self.single_vehicle]

        self.world.tick()
        print(f"[[green]INFO[/] [purple]({self.__class__.__name__})[/]]: Spawned single successfully. Name: {str(vehicle_bp.id)}. {self.get_size} vehicles in environment")
        
    def destroy_all_vehicles(self):
        for vehicle in self.get_vehicles:
            vehicle.destroy()
        print(f"[[green]INFO[/] [purple]({self.__class__.__name__})[/]]: Destroyed all vehicles")

    @property
    def get_size(self):
        return len(self.get_vehicles)

    @property
    def get_vehicles(self):
        return self.world.get_actors().filter("*vehicle*")
    
    def flatten(self, seq):
        if not isinstance(seq, VehicleClass):
            for x in seq:
                if isinstance(x, (list, tuple)):
                    yield from self.flatten(x)
                elif isinstance(x, VehicleClass): 
                    yield from self.flatten(x.value)
                else:
                    yield x
        else: return seq.value