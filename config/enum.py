from enum import Enum, IntEnum

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