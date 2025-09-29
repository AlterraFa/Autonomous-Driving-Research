import pygame
import inspect

from utils.messages.message_handler import MessagingSubscribers

class HUD(MessagingSubscribers):
    def __init__(self, fontName="Arial", fontSize=24, height=720):
        super().__init__()  # init all subscribers
        pygame.font.init()
        self.font = pygame.font.SysFont(fontName, fontSize, bold=True)
        self.text_height = 20
        self._line_cache = {}
        
        self.overlay = pygame.Surface((310, height), pygame.SRCALPHA)
        

    @staticmethod
    def heading_to_cardinal(deg: float) -> str:
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        if not isinstance(deg, (int, float)):
            return ""
        idx = int((deg + 22.5) % 360 // 45)
        return directions[idx]

    def _time_to_str(self, t: float) -> str:
        hours   = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        millis  = int((t % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _render_line(self, surface, label: str, value: str, line_idx: int, 
                 x: int = 10, y: int = 10):
        key = (label, line_idx)
        cached_val, cached_surface = self._line_cache.get(key, (None, None))
        if value != cached_val:
            text_surface = self.font.render(
                f"{label:<15}{value:>{self.max_string}}", True, (255, 255, 255)
            )
            self._line_cache[key] = (value, text_surface)
        else:
            text_surface = cached_surface
        if text_surface:  # blit always
            surface.blit(text_surface, (x, y + self.text_height * line_idx))

    def _read(self, sub, default="N/A"):
        """Helper to read latest subscriber value or fallback."""
        val = sub.receive()
        return val if val is not None else default

    @profile
    def draw_measurement(self, surface: pygame.Surface):
        # Transparent overlay
        pygame.draw.rect(self.overlay, (0, 0, 0, 100), self.overlay.get_rect())
        surface.blit(self.overlay, (0, 0))

        # Read values directly from subscribers
        server_fps = self._read(self.sub_server_fps, 0)
        client_fps = self._read(self.sub_client_fps, 0)
        vehicle_name = self._read(self.sub_vehicle_name, "")
        world_name   = self._read(self.sub_world_name, "")
        velocity     = self._read(self.sub_velocity, 0.0)
        heading      = self._read(self.sub_heading, 0.0)
        accel        = self._read(self.sub_accel, "N/A")
        gyro         = self._read(self.sub_gyro, "N/A")
        enu          = self._read(self.sub_enu, "N/A")
        geo          = self._read(self.sub_geo, "N/A")
        client_runtime = self._read(self.sub_client_runtime, 0.0)
        server_runtime = self._read(self.sub_server_runtime, 0.0)

        accel_str = accel if isinstance(accel, str) else f"( {accel[0]: 6.2f}, {accel[1]: 6.2f}, {accel[2]: 6.2f} )"
        gyro_str  = gyro  if isinstance(gyro, str)  else f"( {gyro[0]: 6.2f}, {gyro[1]: 6.2f}, {gyro[2]: 6.2f} )"
        geo_str   = geo   if isinstance(geo, str)   else f"( {geo[0]: 6.6f}, {geo[1]: 6.6f} )"

        client_time_str = self._time_to_str(client_runtime)
        server_time_str = self._time_to_str(server_runtime)

        if isinstance(enu, str):
            h_str, loc_str = "N/A", "N/A"
        else:
            h_str   = f"{enu[2]: 6.2f} m"
            loc_str = f"( {enu[0]: 6.2f}, {enu[1]: 6.2f} )"

        # Collect lines
        value_lines = [
            ("Server side:",   f"{int(server_fps)} FPS", 0),
            ("Client side:",   f"{int(client_fps)} FPS", 1),
            ("Client runtime:", f"{client_time_str} s", 2),
            ("Server runtime:", f"{server_time_str} s", 3),
            ("Vehicle name:",   vehicle_name, 5),
            ("World name:",     world_name,   6),
            ("Velocity:",       f"{velocity:.2f} (km/h)", 8),
            ("Heading:",        f"{heading:.1f}° {self.heading_to_cardinal(heading)}", 9),
            ("Acceleration:",   accel_str, 10),
            ("Gyroscope:",      gyro_str, 11),
            ("Location:",       loc_str, 12),
            ("Geodetic:",       geo_str, 13),
            ("Height:",         h_str, 14),
        ]

        # Alignment
        self.max_string = max(max(len(v) for _, v, _ in value_lines), 15)

        for label, value, idx in value_lines:
            self._render_line(surface, label, value, idx)

    def draw_controls(self, surface, x=10, y=330):
        line_h = 20
        bar_w, bar_h = 150, 10
        bar_x = x + 100

        white = (255, 255, 255)
        green = (0, 200, 0)
        red   = (200, 0, 0)

        # Read controls directly
        throttle = self._read(self.sub_throttle_logging, 0.0)
        steer    = self._read(self.sub_steer_logging, 0.0)
        brake    = self._read(self.sub_brake_logging, 0.0)
        reverse  = self._read(self.sub_reverse_logging, False)
        handbrake= self._read(self.sub_handbrake_logging, False)
        manual   = self._read(self.sub_manual_logging, False)
        gear     = self._read(self.sub_gear_logging, 0)
        autopilot= self._read(self.sub_autopilot_logging, False)
        model_autopilot = self._read(self.sub_model_autopilot_logging, False)
        regulate = self._read(self.sub_regulate_speed_logging, False)

        # Bars
        surface.blit(self.font.render("Throttle:", True, white), (x, y))
        pygame.draw.rect(surface, white, (bar_x, y+5, bar_w, bar_h), 1)
        pygame.draw.rect(surface, green, (bar_x, y+5, int(bar_w * min(throttle,1.0)), bar_h))

        surface.blit(self.font.render("Steer:", True, white), (x, y+line_h))
        pygame.draw.rect(surface, white, (bar_x, y+line_h+5, bar_w, bar_h), 1)
        if steer >= 0:
            pygame.draw.rect(surface, green, (bar_x + bar_w//2, y+line_h+5,
                                              int((bar_w//2) * min(steer,1.0)), bar_h))
        else:
            pygame.draw.rect(surface, green, (bar_x + bar_w//2 + int((bar_w//2)*steer), y+line_h+5,
                                              int(-(bar_w//2) * steer), bar_h))

        surface.blit(self.font.render("Brake:", True, white), (x, y+2*line_h))
        pygame.draw.rect(surface, white, (bar_x, y+2*line_h+5, bar_w, bar_h), 1)
        pygame.draw.rect(surface, red, (bar_x, y+2*line_h+5, int(bar_w * min(brake,1.0)), bar_h))

        # Others as text
        spacing = 33
        surface.blit(self.font.render(f"{'Throttle:':<{spacing}} {'■' if reverse else '□'}", True, white), (x, y+3*line_h))
        surface.blit(self.font.render(f"{'Hand brake:':<{spacing}} {'■' if handbrake else '□'}", True, white), (x, y+4*line_h))
        surface.blit(self.font.render(f"{'Manual:':<{spacing}} {'■' if manual else '□'}", True, white), (x, y+5*line_h))
        surface.blit(self.font.render(f"{'Gear:':<{spacing}} {gear}", True, white), (x, y+6*line_h))
        surface.blit(self.font.render(f"{'Autopilot:':<{spacing}} {'■' if autopilot else '□'}", True, white), (x, y+7*line_h))
        surface.blit(self.font.render(f"{'Model autopilot:':<{spacing}} {'■' if model_autopilot else '□'}", True, white), (x, y+8*line_h))
        surface.blit(self.font.render(f"{'Regulate speed:':<{spacing}} {'■' if regulate else '□'}", True, white), (x, y+9*line_h))

    def draw_logging(self, surface, x=10, y=510):
        turn = self._read(self.sub_turn_signal, -1)
        if turn == -1:
            direction_str = "Keep lane"
        elif turn == 0:
            direction_str = "Go straight"
        elif turn == 1:
            direction_str = "Turn left"
        elif turn == 2:
            direction_str = "Turn right"
        else:
            direction_str = "N/A"

        line_h = 20
        spacing = 15
        text = self.font.render(f"{'Turn signal:':<{spacing}}{direction_str:>{self.max_string}}", True, (255, 255, 255))
        surface.blit(text, (x, y + 1 * line_h))