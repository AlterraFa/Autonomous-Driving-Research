from enum import Enum, IntEnum
import pygame

class VehicleClass(Enum):
    Cars = [
        'vehicle.audi.a2', 'vehicle.citroen.c3', 'vehicle.chevrolet.impala',
        'vehicle.audi.tt', 'vehicle.jeep.wrangler_rubicon',
        'vehicle.mercedes.coupe', 'vehicle.mercedes.coupe_2020',
        'vehicle.dodge.charger_2020', 'vehicle.lincoln.mkz_2020',
        'vehicle.mini.cooper_s_2021', 'vehicle.ford.crown',
        'vehicle.toyota.prius', 'vehicle.audi.etron', 'vehicle.seat.leon',
        'vehicle.lincoln.mkz_2017', 'vehicle.ford.mustang',
        'vehicle.mini.cooper_s', 'vehicle.nissan.patrol',
        'vehicle.nissan.patrol_2021', 'vehicle.nissan.micra',
        'vehicle.bmw.grandtourer', 'vehicle.tesla.model3'
    ]

    VansPickups = [
        'vehicle.tesla.cybertruck', 'vehicle.volkswagen.t2',
        'vehicle.volkswagen.t2_2021', 'vehicle.mercedes.sprinter'
    ]

    Emergency = [
        'vehicle.dodge.charger_police', 'vehicle.dodge.charger_police_2020',
        'vehicle.ford.ambulance', 'vehicle.carlamotors.firetruck'
    ]

    Trucks = [
        'vehicle.carlamotors.european_hgv', 'vehicle.mitsubishi.fusorosa',
        'vehicle.carlamotors.carlacola'
    ]

    Bikes = [
        'vehicle.yamaha.yzf', 'vehicle.harley-davidson.low_rider',
        'vehicle.vespa.zx125', 'vehicle.kawasaki.ninja',
        'vehicle.diamondback.century', 'vehicle.gazelle.omafiets',
        'vehicle.bh.crossbike'
    ]
    
    Tiny = [
        'vehicle.yamaha.yzf', 'vehicle.harley-davidson.low_rider',
        'vehicle.vespa.zx125', 'vehicle.kawasaki.ninja',
        'vehicle.diamondback.century', 'vehicle.gazelle.omafiets',
        'vehicle.bh.crossbike', 'vehicle.micro.microlino'
    ]
    Small = [
        'vehicle.audi.a2', 'vehicle.citroen.c3', 'vehicle.mini.cooper_s',
        'vehicle.mini.cooper_s_2021', 'vehicle.nissan.micra',
        'vehicle.seat.leon', 'vehicle.toyota.prius'
    ]
    Medium = [
        'vehicle.chevrolet.impala', 'vehicle.audi.tt',
        'vehicle.jeep.wrangler_rubicon', 'vehicle.mercedes.coupe',
        'vehicle.mercedes.coupe_2020', 'vehicle.dodge.charger_2020',
        'vehicle.lincoln.mkz_2017', 'vehicle.lincoln.mkz_2020',
        'vehicle.ford.crown', 'vehicle.audi.etron',
        'vehicle.ford.mustang', 'vehicle.bmw.grandtourer',
        'vehicle.tesla.model3', 'vehicle.nissan.patrol',
        'vehicle.nissan.patrol_2021',
        # moved pickups here
        'vehicle.tesla.cybertruck', 'vehicle.volkswagen.t2',
        'vehicle.volkswagen.t2_2021'
    ]
    Large = [
        'vehicle.mercedes.sprinter',
        'vehicle.dodge.charger_police', 'vehicle.dodge.charger_police_2020',
        'vehicle.ford.ambulance', 'vehicle.carlamotors.firetruck',
        'vehicle.carlamotors.european_hgv', 'vehicle.mitsubishi.fusorosa',
        'vehicle.carlamotors.carlacola'
    ]


class CarlaLabel(IntEnum):
    Unlabeled     = 0
    Road          = 1
    SideWalk      = 2
    Building      = 3
    Wall          = 4
    Fence         = 5
    Pole          = 6
    TrafficLight  = 7
    TrafficSign   = 8
    Vegetation    = 9
    Terrain       = 10
    Sky           = 11
    Pedestrian    = 12
    Rider         = 13
    Car           = 14
    Truck         = 15
    Bus           = 16
    Train         = 17
    Motorcycle    = 18
    Bicycle       = 19
    Static        = 20
    Dynamic       = 21
    Other         = 22
    Water         = 23
    RoadLine      = 24
    Ground        = 25
    Bridge        = 26
    RailTrack     = 27
    GuardRail     = 28
class CameraView(Enum):
    FIRST_PERSON = {
        "x": 0.0, "y": 0.0, "z": 2,    # position
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0
    }
    THIRD_PERSON = {
        "x": -6.0, "y": 0.0, "z": 3.0,   # behind & above
        "roll": 0.0, "pitch": -10.0, "yaw": 0.0
    }
class JoyControl:

    class JoyKey(IntEnum):
        A = 0
        B = 1
        X = 2
        Y = 3
        
        LB = 4
        RB = 5

    class JoyStick(IntEnum):
        LX = 0
        LY = 1
        LT = 2
        RX = 3
        RY = 4
        RT = 5
        
KEYBINDS = {
    pygame.K_1: "toggle_autopilot",
    pygame.K_2: "toggle_model_autopilot",
    pygame.K_q: "toggle_reverse",
    pygame.K_SPACE: "toggle_hand_brake",
    pygame.K_f: "toggle_regulate_speed",
}

JOYBINDS = {
    JoyControl.JoyKey.A: "toggle_autopilot",
    JoyControl.JoyKey.X: "toggle_model_autopilot",
    JoyControl.JoyKey.B: "toggle_reverse",
    JoyControl.JoyKey.LB: "toggle_hand_brake",
    JoyControl.JoyKey.RB: "toggle_regulate_speed",
}