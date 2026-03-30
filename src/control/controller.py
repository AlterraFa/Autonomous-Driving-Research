import pygame

from config.enum import JoyControl, JOYBINDS, KEYBINDS
from src.messages.logger import Logger
from src.messages.message_handler import MessageSender
from src.messages.all_messages import (
    Throttle,
    Steer,
    Brake,
    Reverse,
    Handbrake,
    RegulateSpeed,
    TurnSignal,
)

class Controller:
    def __init__(self):
        self.log = Logger()
        self._init_transmitter()
        pygame.joystick.init()

        self.has_joystick = pygame.joystick.get_count() > 0
        if self.has_joystick:
            self.log.INFO("Joystick detected, prioritized using it")
            joystick = pygame.joystick.Joystick(0)
            self.joystick = joystick
            self.joystick.init()
            self.log.DEBUG(f"Joystick name: {joystick.get_name()}")
            self.log.DEBUG(f"Number of axes: {joystick.get_numaxes()}")
            self.log.DEBUG(f"Number of buttons: {joystick.get_numbuttons()}")
            self.log.DEBUG(f"Number of hats: {joystick.get_numhats()}")
        else:
            self.log.WARNING("No joystick detected. Falling back to keyboard input")

        self.deadzone_stick = 0.12
        self.deadzone_trigger = 0.05
        self.steer_curve = 3  # 1.0 = linear, >1 smoother center
        
        pygame.key.set_repeat()  # no auto-repeat by default
        self.running = True
        
        
        self.view_name = "FIRST_PERSON"; self.view_changed = False
        self.camera_step = 1; self.camera_changed = False
        self.prev_keys_view = pygame.key.get_pressed()
        self.toggle_map = True
        
        self.autopilot = False; self.model_autopilot = False
        self.throt_ctrl = 0; self.steer_ctrl = 0; self.brake_ctrl = 0
        self.reverse = False
        self.hand_brake = False
        self.regulate_speed = False
        
    def _init_transmitter(self):
        """Initialize all message senders for controller."""
        self.send_throttle        = MessageSender(Throttle)
        self.send_steer           = MessageSender(Steer)
        self.send_brake           = MessageSender(Brake)
        self.send_reverse         = MessageSender(Reverse)
        self.send_handbrake       = MessageSender(Handbrake)
        self.send_regulate_speed  = MessageSender(RegulateSpeed)
        self.send_turn_signal     = MessageSender(TurnSignal)
        
    def _apply_deadzone(self, x: float, dz: float) -> float:
        if abs(x) < dz:
            return 0.0
        s = (abs(x) - dz) / (1.0 - dz)
        return s if x > 0 else -s

    def _curve(self, x: float) -> float:
        return (abs(x) ** self.steer_curve) * (1 if x >= 0 else -1)

    def _trigger_01(self, v: float) -> float:
        return max(0.0, min(1.0, (v + 1.0) * 0.5))

    def process_events(self, server_time: float):
        """Process keyboard + window events.
        Returns False if the program should quit."""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_k] or keys[pygame.K_ESCAPE]:
            self.running = False

        self.process_view(events)
        self.process_ctrl(events, server_time)
        return self.running
    
    def process_view(self, events):
        self.view_changed = False; self.camera_changed = False
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.view_name = "FIRST_PERSON"
                    self.view_changed = True
                    self.log.DEBUG(f"View toggled → [i]{'First Person'}[/i]")

                elif event.key == pygame.K_DOWN:
                    self.view_name = "THIRD_PERSON"
                    self.view_changed = True
                    self.log.DEBUG(f"View toggled → [i]{'Third Person'}[/i]")

                elif event.key == pygame.K_RIGHT:
                    self.camera_changed = True
                    self.camera_step = 1

                elif event.key == pygame.K_LEFT:
                    self.camera_changed = True
                    self.camera_step = -1
                
                elif event.key == pygame.K_m and event.type == pygame.KEYDOWN:
                    self.toggle_map = not self.toggle_map
                    self.log.INFO(f"Toogle map -> [i][{'green' if self.toggle_map else 'red'}]{'enabled' if self.toggle_map else 'disabled'}[/][/]")
                    
            # joystick hat → mirror your view/camera controls
            if self.has_joystick and event.type == pygame.JOYHATMOTION:
                hx, hy = event.value
                if hy == 1:
                    self.view_name = "FIRST_PERSON"; self.view_changed = True
                    self.log.DEBUG(f"View toggled → [i]{'First Person'}[/i]")
                elif hy == -1:
                    self.view_name = "THIRD_PERSON"; self.view_changed = True
                    self.log.DEBUG(f"View toggled → [i]{'Third Person'}[/i]")
                if hx != 0:
                    self.camera_changed = True
                    self.camera_step = 1 if hx > 0 else -1
    
    def toggle_autopilot(self):
        self.autopilot = not self.autopilot
        self.log.WARNING(
            f"Autopilot toggled → "
            f"[i][{'green' if self.autopilot else 'red'}]"
            f"{'Engaged' if self.autopilot else 'Disengaged'}[/i][/]"
        )

        if self.autopilot == True:
            self.model_autopilot = False

    def toggle_model_autopilot(self):
        self.model_autopilot = not self.model_autopilot
        self.log.WARNING(
            f"Model inference toggled → "
            f"[i][{'green' if self.model_autopilot else 'red'}]"
            f"{'Engaged' if self.model_autopilot else 'Disengaged'}[/i][/]"
        )
        
        if self.model_autopilot == True:
            self.autopilot = False

    def toggle_reverse(self):
        self.reverse = not self.reverse
        self.log.INFO(f"Reverse [bold][i]{'ON' if self.reverse else 'OFF'}[/][/]")

    def toggle_hand_brake(self):
        self.hand_brake = not self.hand_brake

    def toggle_regulate_speed(self):
        self.regulate_speed = not self.regulate_speed

    def process_ctrl(self, events, server_time):
        self.throt_ctrl = 0
        self.steer_ctrl = 0
        self.brake_ctrl = 0

        for event in events:
            # Handle quit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                return False

            # Keyboard toggles
            if event.type == pygame.KEYDOWN and event.key in KEYBINDS:
                getattr(self, KEYBINDS[event.key])()

            # Joystick toggles
            if self.has_joystick and event.type == pygame.JOYBUTTONDOWN and event.button in JOYBINDS:
                getattr(self, JOYBINDS[event.button])()

        # Keyboard continuous controls
        keys = pygame.key.get_pressed()
        steer_inc = 5e-4 * server_time * 1000
        if not self.model_autopilot:
            self.throt_ctrl = 0.01 if keys[pygame.K_w] else 0
            self.brake_ctrl = 0.2 if keys[pygame.K_s] else 0
            self.steer_ctrl = (-steer_inc if keys[pygame.K_a] else
                                steer_inc if keys[pygame.K_d] else 0)

            # Joystick continuous controls
            if self.has_joystick:
                left_x = self._apply_deadzone(self.joystick.get_axis(JoyControl.JoyStick.LX), self.deadzone_stick)
                lt = self._trigger_01(self.joystick.get_axis(JoyControl.JoyStick.LT))
                rt = self._trigger_01(self.joystick.get_axis(JoyControl.JoyStick.RT))

                lt = 0.0 if lt < self.deadzone_trigger else lt
                rt = 0.0 if rt < self.deadzone_trigger else rt

                self.steer_ctrl = self._curve(left_x) * 0.5
                self.throt_ctrl = rt
                self.brake_ctrl = lt

        if self.model_autopilot:
            if keys[pygame.K_w]:
                self.send_turn_signal.send(0)
            elif keys[pygame.K_a]:
                self.send_turn_signal.send(1)
            elif keys[pygame.K_d]:
                self.send_turn_signal.send(2)
            else:
                self.send_turn_signal.send(-1)
            
        self.send_throttle.send(self.throt_ctrl)
        self.send_steer.send(self.steer_ctrl)
        self.send_brake.send(self.brake_ctrl)
        self.send_reverse.send(self.reverse)
        self.send_handbrake.send(self.hand_brake)
        self.send_regulate_speed.send(self.regulate_speed)
