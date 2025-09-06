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
