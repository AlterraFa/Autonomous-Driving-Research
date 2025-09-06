"""INSPIRATION TAKEN FROM BFMC ECC"""

import inspect
import queue

from multiprocessing import Pipe, Queue
from typing import Literal
from rich import print

from utils.messages.all_messages import *

class MessageBroker:
    latest_values = {}  # (owner, msgID) -> last payload

    @classmethod
    def put(cls, _, message):
        key = (message["Owner"], message["msgID"])
        cls.latest_values[key] = message

    @classmethod
    def get(cls, owner, msg_id):
        return cls.latest_values.get((owner, msg_id))

class MessageSender:
    """Helper to send typed messages to the broker"""

    def __init__(self, message):
        self.message = message

    def send(self, value):
        payload = {
            "Owner": self.message.Owner.value,
            "msgID": self.message.msgID.value,
            "msgType": self.message.msgType.value,
            "msgValue": value,
        }
        MessageBroker.put(self.message.Queue.value, payload)
class MessageSubscriber:
    def __init__(self, message):
        self._message = message

    def receive(self, return_payload = False):
        msg = MessageBroker.get(self._message.Owner.value, self._message.msgID.value)
        if not msg:
            return None

        expected_types = self._message.msgType.value
        if not isinstance(msg["msgValue"], expected_types):
            print(
                f"[WARN]: Type mismatch for {self._message}: "
                f"got {type(msg['msgValue']).__name__}, "
                f"expected {[t.__name__ for t in expected_types]}"
            )
        return msg["msgValue"] if not return_payload else msg
    
    
class MessagingSubscribers:
    """Convenience wrapper to init all subscribers."""

    def __init__(self):
        self._init_subscribers()

    def _init_subscribers(self):
        # Measurement
        self.sub_server_fps     = MessageSubscriber(ServerFps)
        self.sub_client_fps     = MessageSubscriber(ClientFps)
        self.sub_vehicle_name   = MessageSubscriber(VehicleName)
        self.sub_world_name     = MessageSubscriber(WorldName)
        self.sub_velocity       = MessageSubscriber(Velocity)
        self.sub_heading        = MessageSubscriber(Heading)
        self.sub_accel          = MessageSubscriber(Accel)
        self.sub_gyro           = MessageSubscriber(Gyro)
        self.sub_enu            = MessageSubscriber(Enu)
        self.sub_geo            = MessageSubscriber(Geo)
        self.sub_client_runtime = MessageSubscriber(ClientRuntime)
        self.sub_server_runtime = MessageSubscriber(ServerRuntime)

        # Control
        self.sub_throttle       = MessageSubscriber(Throttle)
        self.sub_steer          = MessageSubscriber(Steer)
        self.sub_brake          = MessageSubscriber(Brake)
        self.sub_reverse        = MessageSubscriber(Reverse)
        self.sub_handbrake      = MessageSubscriber(Handbrake)
        self.sub_manual         = MessageSubscriber(Manual)
        self.sub_gear           = MessageSubscriber(Gear)
        self.sub_autopilot      = MessageSubscriber(Autopilot)
        self.sub_regulate_speed = MessageSubscriber(RegulateSpeed)

        # Logging
        self.sub_turn_signal    = MessageSubscriber(TurnSignal)

class MessagingSenders:
    """Convenience wrapper to init all senders."""

    def __init__(self):
        self._init_senders()

    def _init_senders(self):
        # Measurement
        self.send_server_fps     = MessageSender(ServerFps)
        self.send_client_fps     = MessageSender(ClientFps)
        self.send_vehicle_name   = MessageSender(VehicleName)
        self.send_world_name     = MessageSender(WorldName)
        self.send_velocity       = MessageSender(Velocity)
        self.send_heading        = MessageSender(Heading)
        self.send_accel          = MessageSender(Accel)
        self.send_gyro           = MessageSender(Gyro)
        self.send_enu            = MessageSender(Enu)
        self.send_geo            = MessageSender(Geo)
        self.send_client_runtime = MessageSender(ClientRuntime)
        self.send_server_runtime = MessageSender(ServerRuntime)

        # Control
        self.send_throttle       = MessageSender(Throttle)
        self.send_steer          = MessageSender(Steer)
        self.send_brake          = MessageSender(Brake)
        self.send_reverse        = MessageSender(Reverse)
        self.send_handbrake      = MessageSender(Handbrake)
        self.send_manual         = MessageSender(Manual)
        self.send_gear           = MessageSender(Gear)
        self.send_autopilot      = MessageSender(Autopilot)
        self.send_regulate_speed = MessageSender(RegulateSpeed)

        # Logging
        self.send_turn_signal    = MessageSender(TurnSignal)
