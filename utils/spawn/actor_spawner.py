import carla
import random
import time

from typing import Literal
from config.enum import VehicleClass
from utils.messages.logger import Logger

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
    def __init__(self, world: carla.Client, logger: Logger = None):
        self.blueprints    = world.get_blueprint_library()
        self.vehicle_spawn_pts = world.get_map().get_spawn_points()

        self.world = world
        self.vehicles = []; self.walkers = []; self.sensors = []

        self.log = logger if logger else Logger()

    def spawn_mass_vehicle(self, size: int, transform: carla.Transform = None, autopilot = True, exclude: list[VehicleClass] = None):
        if size < 0:
            self.log.ERROR(f"Number of spawning vehicles must be a positive integer")
            exit(12)
        if transform is not None:
            if size != len(transform):
                self.log.ERROR(f"Number of spawning location must equal to spawning vehicles")
                exit(12)
        
        vehicle_bp = self.blueprints.filter("*vehicle*")
        flattened_exclude = list(self.__flatten__(exclude)) if exclude is not None else []
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
                self.log.ERROR(f"A random vehicle has not spawned successfully. Skipping.")
        
        self.world.tick()
        self.log.CUSTOM("SUCCESS", f"Spawned mass successfully. {self.get_size} vehicles in environment")
        
        
    def spawn_single_vehicle(self, bp_id=None, name: str = "ego", transform: carla.Transform = None, 
                            random_offset: float = 0, autopilot = True, exclude = None):
        # --- 1. Blueprint Logic (Keep your existing logic) ---
        if bp_id is None:
            flattened_exclude = list(self.__flatten__(exclude)) if exclude is not None else []
            while True:
                vehicle_name = random.choice(list(Vehicle_BP.__args__))
                if vehicle_name in flattened_exclude: continue
                try:
                    vehicle_bp = self.blueprints.find(vehicle_name)
                    break
                except: 
                    self.log.ERROR(f"Did not find {vehicle_name}. Retrying ...")
                    continue
        else:
            vehicle_bp = self.blueprints.find(bp_id)
            
        vehicle_bp.set_attribute("role_name", name)  
        
        # # Helper to check the forbidden circle
        # def is_in_forbidden_zone(loc):
        #     if not self.exclude_params: return False
        #     cx, cy, radius = self.exclude_params
        #     return math.hypot(loc.x - cx, loc.y - cy) <= radius

        # --- 2. Improved Spawn Logic ---
        self.single_vehicle = None
        max_retries = 50
        retry_count = 0

        while retry_count < max_retries:
            retry_count += 1
            
            if transform is not None:
                # User provided a specific spot
                spawn_transform = transform
            else:
                # "Any location" logic: Use waypoints for 100% road coverage
                # We use the map to get a random waypoint
                all_waypoints = self.world.get_map().generate_waypoints(distance=2.0)
                target_waypoint = random.choice(all_waypoints)
                spawn_transform = target_waypoint.transform
                # Lift it slightly to avoid spawning 'into' the road
                spawn_transform.location.z += 0.5 


            if random_offset != 0:
                random_yaw = random.uniform(-random_offset, random_offset)
                spawn_transform.rotation.yaw += random_yaw
                
            self.single_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)

            if self.single_vehicle is not None:
                break
            else:
                if transform is not None:
                    self.log.ERROR("Fixed transform collision. Failed to spawn.")
                    return None
                # If random, the loop will just try a different waypoint

        # --- 3. Finalization ---
        if self.single_vehicle is None:
            self.log.ERROR("Failed to spawn vehicle after multiple attempts.")
            return None

        self.single_vehicle.set_autopilot(autopilot)
        self.vehicles += [self.single_vehicle]

        self.world.tick()
        self.log.CUSTOM("SUCCESS", f"Spawned {vehicle_bp.id} at {spawn_transform.location}. Autopilot: {autopilot}")
        return self.single_vehicle

    def despawn_vehicles(self):
        for vehicle in self.get_vehicles:
            vehicle.destroy()
        self.log.CUSTOM("SUCCESS", f"Destroyed all vehicles")

    @property
    def get_size(self):
        return len(self.get_vehicles)

    @property
    def get_vehicles(self):
        return self.world.get_actors().filter("*vehicle*")
    
    def __flatten__(self, seq):
        if not isinstance(seq, VehicleClass):
            for x in seq:
                if isinstance(x, (list, tuple)):
                    yield from self.__flatten__(x)
                elif isinstance(x, VehicleClass): 
                    yield from self.__flatten__(x.value)
                else:
                    yield x
        else: 
            return seq.value

    def wait_for_live_actor(self, role_name: str, timeout_s: float = 30.0, settle_ticks: int = 50):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            vehicles = self.world.get_actors().filter('vehicle.*')
            for v in vehicles:
                try:
                    if v.attributes.get('role_name', '') == role_name and v.is_alive:
                        self.log.INFO(f"Found {role_name} actor, settling...")
                        prev_pos = None
                        for tick_idx in range(settle_ticks):
                            if self.world.get_settings().synchronous_mode:
                                self.world.tick()
                            else:
                                time.sleep(0.05)
                            try:
                                curr_pos = v.get_location()
                                if prev_pos is not None and prev_pos.distance(curr_pos) > 0.1:
                                    self.log.WARNING(f"Actor still moving (tick {tick_idx}), continuing settle...")
                                prev_pos = curr_pos
                            except:
                                pass
                        self.log.INFO(f"Actor settled after {settle_ticks} ticks")
                        return v
                except Exception:
                    pass
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                time.sleep(0.05)
        return None