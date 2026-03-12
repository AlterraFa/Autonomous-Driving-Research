"""INSPIRATION TAKEN FROM BFMC ECC"""
from src.messages.logger import Logger
from enum import Enum

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
        self.log = Logger()
        self._message = message

    def receive(self, return_payload = False):
        msg = MessageBroker.get(self._message.Owner.value, self._message.msgID.value)
        if not msg:
            default_msg = getattr(self._message, "default", None)
            
            if callable(default_msg):
                return default_msg()
            if isinstance(default_msg, Enum):
                default_msg = default_msg.value
            if default_msg is None:
                return None
            return default_msg

        expected_types = self._message.msgType.value
        if not isinstance(msg["msgValue"], expected_types):
            self.log.WARNING(
                f"Type mismatch for {self._message}: "
                f"got {type(msg['msgValue']).__name__}, "
                f"expected {[t.__name__ for t in expected_types]}", 
                once = True
            )
        return msg["msgValue"] if not return_payload else msg
    