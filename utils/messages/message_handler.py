"""INSPIRATION TAKEN FROM BFMC ECC"""

import inspect
import queue

from multiprocessing import Pipe, Queue
from typing import Literal
from rich import print

from utils.messages.all_messages import *

class MessageBroker:
    """Central registry of all subscriptions (owner, msgID) -> list of queues"""
    subscriptions = {}

    @classmethod
    def put(cls, _, message):
        key = (message["owner"], message["msgid"])
        if key in cls.subscriptions:
            for q in cls.subscriptions[key]:
                try:
                    q.put_nowait(message)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(message)
                    except Exception:
                        pass

    @classmethod
    def subscribe(cls, owner, msg_id, _=None, maxsize: int = 5):
        q = Queue(maxsize=maxsize)
        cls.subscriptions.setdefault((owner, msg_id), []).append(q)
        return q

    @classmethod
    def unsubscribe(cls, owner, msg_id, q):
        subs = cls.subscriptions.get((owner, msg_id))
        if subs and q in subs:
            subs.remove(q)


class MessageSender:
    """Helper to send typed messages to the broker"""

    def __init__(self, message):
        self.message = message

    def send(self, value):
        payload = {
            "owner": self.message.Owner.value,
            "msgid": self.message.msgID.value,
            "msgType": self.message.msgType.value,
            "msgValue": value,
        }
        MessageBroker.put(self.message.Queue.value, payload)


class MessageSubscriber:
    """Helper to receive messages from the broker"""

    def __init__(self, message, deliveryMode: Literal["fifo", "lastonly"] = "fifo", subscribe=False):
        self._message = message
        self._deliveryMode = deliveryMode.lower()
        self._queue = None
        self._receiver = inspect.currentframe().f_back.f_locals["self"].__class__.__name__

        if subscribe:
            self.subscribe()

        if self._deliveryMode not in ("fifo", "lastonly"):
            print(f"[yellow][WARN][/]: Wrong delivery mode '{deliveryMode}', expected 'fifo' or 'lastonly'. Using FIFO.")
            self._deliveryMode = "fifo"

    def receive(self, block=False):
        """Get next message value, or None if no data (when block=False)."""
        if self._queue is None:
            return None

        try:
            msg = self._queue.get(block=block, timeout=0.01 if not block else None)
        except queue.Empty:
            return None

        if self._deliveryMode == "lastonly":
            # Drain the queue, keep only the latest
            while not self._queue.empty():
                msg = self._queue.get_nowait()

        # Type check
        expected_types = self._message.msgType.value
        if not isinstance(msg["msgValue"], expected_types):
            print(
                f"[yellow][WARN][/]: Type mismatch for {self._message}: "
                f"got {type(msg['msgValue']).__name__}, "
                f"expected one of {[t.__name__ for t in expected_types]}"
            )

        return msg["msgValue"]

    def empty(self):
        """Drain all messages without returning them."""
        if self._queue:
            while not self._queue.empty():
                self._queue.get_nowait()

    def isDataInQueue(self):
        """Check if there’s a message available right now."""
        return self._queue is not None and not self._queue.empty()

    def setDeliveryModeToFIFO(self):
        self._deliveryMode = "fifo"

    def setDeliveryModeToLastOnly(self):
        self._deliveryMode = "lastonly"

    def subscribe(self):
        """Subscribe this subscriber to a message type."""
        self._queue = MessageBroker.subscribe(
            self._message.Owner.value,
            self._message.msgID.value,
        )

    def unsubscribe(self):
        """Unsubscribe this subscriber."""
        if self._queue:
            MessageBroker.unsubscribe(
                self._message.Owner.value,
                self._message.msgID.value,
                self._queue,
            )
            self._queue = None

class MessagingSubscribers:
    """Convenience wrapper to init all subscribers."""

    def __init__(self):
        self._init_subscribers()

    def _init_subscribers(self):
        # Measurement
        self.sub_server_fps     = MessageSubscriber(ServerFps, "fifo", True)
        self.sub_client_fps     = MessageSubscriber(ClientFps, "fifo", True)
        self.sub_vehicle_name   = MessageSubscriber(VehicleName, "fifo", True)
        self.sub_world_name     = MessageSubscriber(WorldName, "fifo", True)
        self.sub_velocity       = MessageSubscriber(Velocity, "fifo", True)
        self.sub_heading        = MessageSubscriber(Heading, "fifo", True)
        self.sub_accel          = MessageSubscriber(Accel, "fifo", True)
        self.sub_gyro           = MessageSubscriber(Gyro, "fifo", True)
        self.sub_enu            = MessageSubscriber(Enu, "fifo", True)
        self.sub_geo            = MessageSubscriber(Geo, "fifo", True)
        self.sub_client_runtime = MessageSubscriber(ClientRuntime, "fifo", True)
        self.sub_server_runtime = MessageSubscriber(ServerRuntime, "fifo", True)

        # Control
        self.sub_throttle       = MessageSubscriber(Throttle, "fifo", True)
        self.sub_steer          = MessageSubscriber(Steer, "fifo", True)
        self.sub_brake          = MessageSubscriber(Brake, "fifo", True)
        self.sub_reverse        = MessageSubscriber(Reverse, "fifo", True)
        self.sub_handbrake      = MessageSubscriber(Handbrake, "fifo", True)
        self.sub_manual         = MessageSubscriber(Manual, "fifo", True)
        self.sub_gear           = MessageSubscriber(Gear, "fifo", True)
        self.sub_autopilot      = MessageSubscriber(Autopilot, "fifo", True)
        self.sub_regulate_speed = MessageSubscriber(RegulateSpeed, "fifo", True)

        # Logging
        self.sub_turn_signal    = MessageSubscriber(TurnSignal, "fifo", True)

        # CarlaViewer specific
        self.sub_imu_data       = MessageSubscriber(ImuData, "fifo", True)
        self.sub_current_speed  = MessageSubscriber(CurrentSpeed, "fifo", True)
        self.sub_current_steer  = MessageSubscriber(CurrentSteer, "fifo", True)
        self.sub_location       = MessageSubscriber(Location, "fifo", True)

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

        # CarlaViewer specific
        self.send_imu_data       = MessageSender(ImuData)
        self.send_current_speed  = MessageSender(CurrentSpeed)
        self.send_current_steer  = MessageSender(CurrentSteer)
        self.send_location       = MessageSender(Location)
