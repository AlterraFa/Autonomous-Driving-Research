from enum import Enum
import numpy as np
import torch

#################################### Measurement ####################################
class ServerFps(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 1
    msgType = (int, float)

class ClientFps(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 2
    msgType = (int, float)

class VehicleName(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 3
    msgType = (str,)

class WorldName(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 4
    msgType = (str,)

class Velocity(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 5
    msgType = (int, float)

class Speed(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 5
    msgType = (int, float)

class Heading(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 6
    msgType = (float, int, np.ndarray, torch.Tensor)

class Accel(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 7
    msgType = (list, np.ndarray, torch.Tensor)   # 3 floats (x,y,z)

class Gyro(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 8
    msgType = (list, np.ndarray, torch.Tensor)   # 3 floats (x,y,z)

class Enu(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 9
    msgType = (dict, np.ndarray, torch.Tensor)   # structured or vector form

class Geo(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 10
    msgType = (dict, np.ndarray, torch.Tensor)

class ClientRuntime(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 11
    msgType = (float, int)

class ServerRuntime(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 12
    msgType = (float, int)


#################################### Control ####################################
class Throttle(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 20
    msgType = (float, int, np.ndarray, torch.Tensor)

class Steer(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 21
    msgType = (float, int, np.ndarray, torch.Tensor)

class Brake(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 22
    msgType = (float, int)

class Reverse(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 23
    msgType = (bool,)

class Handbrake(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 24
    msgType = (bool,)

class Manual(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 25
    msgType = (bool,)

class Gear(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 26
    msgType = (int,)

class Autopilot(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 27
    msgType = (bool,)

class RegulateSpeed(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 28
    msgType = (bool,)


#################################### Logging ####################################
class TurnSignal(Enum):
    Queue = "General"
    Owner = "HUD"
    msgID = 40
    msgType = (int,)   # -1, 0, 1, 2