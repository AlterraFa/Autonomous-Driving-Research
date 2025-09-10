from enum import Enum
import numpy as np
import torch

#################################### Measurement ####################################
class ServerFps(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 1
    msgType = (int, float)
    default = 0

class ClientFps(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 2
    msgType = (int, float)
    default = 0

class VehicleName(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 3
    msgType = (str,)
    default = ""

class WorldName(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 4
    msgType = (str,)
    default = ""

class Velocity(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 5
    msgType = (int, float)
    default = 0.0

class Speed(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 5
    msgType = (int, float)
    default = 0.0

class Heading(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 6
    msgType = (float, int, np.ndarray, torch.Tensor)
    default = 0.0

class Accel(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 7
    msgType = (list, np.ndarray, torch.Tensor)
    default = staticmethod(lambda: np.zeros(3, dtype=float))

class Gyro(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 8
    msgType = (list, np.ndarray, torch.Tensor)
    default = staticmethod(lambda: np.zeros(3, dtype=float))

class Enu(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 9
    msgType = (dict, np.ndarray, torch.Tensor)
    default = staticmethod(lambda: np.zeros(3, dtype=float))  # [east, north, up]

class Geo(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 10
    msgType = (dict, np.ndarray, torch.Tensor)
    default = staticmethod(lambda: np.zeros(3, dtype=float))  # [lat, lon, alt]

class ClientRuntime(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 11
    msgType = (float, int)
    default = 0.0

class ServerRuntime(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 12
    msgType = (float, int)
    default = 0.0

class Location(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 13
    msgType = (list, np.ndarray, torch.Tensor)
    default = staticmethod(lambda: np.zeros(3, dtype=float))


#################################### Control ####################################
class Throttle(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 20
    msgType = (float, int, np.ndarray, torch.Tensor)
    default = 0.0

class Steer(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 21
    msgType = (float, int, np.ndarray, torch.Tensor)
    default = 0.0

class Brake(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 22
    msgType = (float, int)
    default = 0.0

class Reverse(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 23
    msgType = (bool,)
    default = False

class Handbrake(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 24
    msgType = (bool,)
    default = False

class Manual(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 25
    msgType = (bool,)
    default = False

class Gear(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 26
    msgType = (int,)
    default = 0

class Autopilot(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 27
    msgType = (bool,)
    default = False

class RegulateSpeed(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 28
    msgType = (bool,)
    default = False


#################################### Logging ####################################
class TurnSignal(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 40
    msgType = (int,)
    default = -1

#################################### Inference ##################################

class ModelSteer(Enum):
    Queue = "General"
    Owner = "Model"
    msgID = 41
    msgType = (float, )
    default = None

class ModelThrottle(Enum):
    Queue = "General"
    Owner = "Model"
    msgID = 42
    msgType = (float, )
    default = None

class ModelBrake(Enum):
    Queue = "General"
    Owner = "Model"
    msgID = 43
    msgType = (float, )
    default = None
    
class ModelAutopilot(Enum):
    Queue = "General"
    Owner = "Model"
    msgID = 44
    msgType = (bool, )
    default = False

#################################### Logging ####################################
class ThrottleLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 50
    msgType = (float, int, np.ndarray, torch.Tensor)
    default = 0.0

class SteerLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 51
    msgType = (float, int, np.ndarray, torch.Tensor)
    default = 0.0

class BrakeLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 52
    msgType = (float, int)
    default = 0.0

class ReverseLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 53
    msgType = (bool,)
    default = False

class HandbrakeLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 54
    msgType = (bool,)
    default = False

class ManualLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 55
    msgType = (bool,)
    default = False

class GearLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 56
    msgType = (int,)
    default = 0

class AutopilotLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 57
    msgType = (bool,)
    default = False

class RegulateSpeedLog(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 58
    msgType = (bool,)
    default = False