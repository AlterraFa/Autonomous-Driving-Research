import pygame
import numpy as np
import cv2

from utils.messages.message_handler import MessagingSubscribers

class HUD(MessagingSubscribers):
    def __init__(self, display, fontName="Arial", fontSize=24, height=720, headless = False):
        super().__init__()  # init all subscribers
        pygame.font.init()
        self.display = display
        self.headless = headless
        self.font = pygame.font.SysFont(fontName, fontSize, bold=True)
        self.text_height = 20
        self._line_cache = {}
        
        self.overlay = pygame.Surface((310, height), pygame.SRCALPHA)
        
    def to_surface(self, frame: np.ndarray) -> pygame.Surface:
        if self.headless: return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        frame = np.ascontiguousarray(frame)

        # RGBA Processing
        if frame.ndim == 3 and frame.shape[2] == 4:
            h, w, _ = frame.shape
            surf = pygame.image.frombuffer(frame.data, (w, h), "BGRA")
            return surf
        # RGB Processing
        if frame.ndim == 3 and frame.shape[2] == 3:
            h, w, _ = frame.shape
            surf = pygame.image.frombuffer(frame.data, (w, h), "RGB")
            return surf

        # Grayscaled Processing
        if frame.ndim == 2:
            rgb = np.repeat(frame[:, :, None], 3, axis=2)
            h, w, _ = rgb.shape
            return pygame.image.frombuffer(rgb.data, (w, h), "RGB")

        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    def draw_frame(self, frame: np.ndarray, position = (0, 0)) -> None:
        if self.headless: return
        surface = self.to_surface(frame)
        self.display.blit(surface, position)

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

    def _render_line(self, label: str, value: str, line_idx: int, 
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
            self.display.blit(text_surface, (x, y + self.text_height * line_idx))

    def _read(self, sub, default="N/A"):
        """Helper to read latest subscriber value or fallback."""
        val = sub.receive()
        return val if val is not None else default

    def draw_measurement(self):
        # Transparent overlay
        pygame.draw.rect(self.overlay, (0, 0, 0, 100), self.overlay.get_rect())
        self.display.blit(self.overlay, (0, 0))

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
            self._render_line(label, value, idx)

    def draw_controls(self, x=10, y=330):
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
        self.display.blit(self.font.render("Throttle:", True, white), (x, y))
        pygame.draw.rect(self.display, white, (bar_x, y+5, bar_w, bar_h), 1)
        pygame.draw.rect(self.display, green, (bar_x, y+5, int(bar_w * min(throttle,1.0)), bar_h))

        self.display.blit(self.font.render("Steer:", True, white), (x, y+line_h))
        pygame.draw.rect(self.display, white, (bar_x, y+line_h+5, bar_w, bar_h), 1)
        if steer >= 0:
            pygame.draw.rect(self.display, green, (bar_x + bar_w//2, y+line_h+5,
                                              int((bar_w//2) * min(steer,1.0)), bar_h))
        else:
            pygame.draw.rect(self.display, green, (bar_x + bar_w//2 + int((bar_w//2)*steer), y+line_h+5,
                                              int(-(bar_w//2) * steer), bar_h))

        self.display.blit(self.font.render("Brake:", True, white), (x, y+2*line_h))
        pygame.draw.rect(self.display, white, (bar_x, y+2*line_h+5, bar_w, bar_h), 1)
        pygame.draw.rect(self.display, red, (bar_x, y+2*line_h+5, int(bar_w * min(brake,1.0)), bar_h))

        # Others as text
        spacing = 33
        self.display.blit(self.font.render(f"{'Throttle:':<{spacing}} {'■' if reverse else '□'}", True, white), (x, y+3*line_h))
        self.display.blit(self.font.render(f"{'Hand brake:':<{spacing}} {'■' if handbrake else '□'}", True, white), (x, y+4*line_h))
        self.display.blit(self.font.render(f"{'Manual:':<{spacing}} {'■' if manual else '□'}", True, white), (x, y+5*line_h))
        self.display.blit(self.font.render(f"{'Gear:':<{spacing}} {gear}", True, white), (x, y+6*line_h))
        self.display.blit(self.font.render(f"{'Autopilot:':<{spacing}} {'■' if autopilot else '□'}", True, white), (x, y+7*line_h))
        self.display.blit(self.font.render(f"{'Model autopilot:':<{spacing}} {'■' if model_autopilot else '□'}", True, white), (x, y+8*line_h))
        self.display.blit(self.font.render(f"{'Regulate speed:':<{spacing}} {'■' if regulate else '□'}", True, white), (x, y+9*line_h))

    def draw_logging(self, x=10, y=510):
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
        self.display.blit(text, (x, y + 1 * line_h))


def draw_border(frame, border_thicc: int, border_color: tuple):
    frame = cv2.copyMakeBorder(
        frame,
        top=border_thicc, bottom=border_thicc,
        left=border_thicc, right=border_thicc,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color
    )
    return frame

def overlay_waypoints_on_map(map_img,
                              waypoints,
                              meters_span: tuple[float, float],
                              scale: tuple[float, float], 
                              color=(0, 255, 0),
                              thickness=2,
                              swap_axes: bool = False,
                              flip_lat: bool = False,
                              origin: str = "bottom_center",   # "bottom_center" or "center"
                              draw_origin: bool = False):
    """
    Overlay waypoints on routed map.
    meters_span:
      - origin == "bottom_center": (forward_range, lateral_half_range)
          forward in [0..forward_range] maps to image height
      - origin == "center": (forward_half_range, lateral_half_range)
          forward in [-fwd_half..+fwd_half] maps to image height
    Waypoints default to [forward, lateral]. Set swap_axes=True if [lateral, forward].
    """
    if map_img is None or waypoints is None:
        return map_img

    # to numpy
    try:
        import torch
        if isinstance(waypoints, torch.Tensor):
            wp = waypoints.detach().cpu().numpy()
        else:
            wp = np.asarray(waypoints)
    except Exception:
        wp = np.asarray(waypoints)

    if wp.ndim == 3:  # (B,N,2)
        wp = wp[0]
    if wp.ndim != 2 or wp.shape[1] < 2 or len(wp) == 0:
        return map_img
    
    wp[:, 0] *= scale[0]
    wp[:, 1] *= scale[1]

    if swap_axes:
        wp = wp[:, [1, 0]]

    fwd = wp[:, 0].astype(np.float32)
    lat = wp[:, 1].astype(np.float32)
    if flip_lat:
        lat = -lat

    H, W = map_img.shape[:2]
    fwd_span, lat_half = float(meters_span[0]), float(meters_span[1])

    # scales and origin
    if origin == "center":
        # full height represents [-fwd_span .. +fwd_span]
        px_per_m_fwd = H / max(2.0 * fwd_span, 1e-6)
        cx, cy = W // 2, H // 2
    else:  # "bottom_center"
        # full height represents [0 .. fwd_span]
        px_per_m_fwd = H / max(fwd_span, 1e-6)
        cx, cy = W // 2, H - 1

    px_per_m_lat = W / max(2.0 * lat_half, 1e-6)

    xs = (cx + lat).astype(np.int32)
    ys = (cy - fwd).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)

    if len(pts) > 1:
        cv2.polylines(map_img, [pts], False, color, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(map_img, tuple(p), 2, color, -1, cv2.LINE_AA)

    if draw_origin:
        cv2.drawMarker(map_img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

    return map_img

def overlay_gmm_on_map(map_img,
                        weights,
                        mu,
                        sigma,
                        scale: tuple[float, float],
                        alpha: float = 0.2,
                        n_std: float = 2.0,
                        swap_axes: bool = False,
                        flip_lat: bool = False,
                        origin: str = "center"):
    """
    Fast overlay of a variable-K diagonal GMM onto the routed map.
    Draws all rings for a component on a single temp image, then blends once.
    """
    if map_img is None:
        return map_img

    def to_np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    WTS = to_np(weights).reshape(-1)
    MU  = to_np(mu)
    SG  = to_np(sigma)

    if MU.ndim == 4: MU = MU[0]
    if SG.ndim == 4: SG = SG[0]
    if MU.shape[0] != WTS.shape[0] and MU.shape[1] == WTS.shape[0]:
        MU = np.transpose(MU, (1, 0, 2))
        SG = np.transpose(SG, (1, 0, 2))
    K = min(WTS.shape[0], MU.shape[0])

    px_per_m_fwd, px_per_m_lat = float(scale[0]), float(scale[1])
    H, W = map_img.shape[:2]
    cx = W // 2
    cy = H // 2 if origin == "center" else H - 1

    wmax = float(np.max(WTS)) if np.max(WTS) > 0 else 1.0
    ws = (WTS[:K] / wmax).clip(0.0, 1.0)

    overlay = map_img.copy()
    for k in range(K):
        comp_color = (255, 255, 0)  # BGR

        temp_ring = np.zeros_like(overlay, dtype=np.uint8)

        for t in range(MU.shape[1]):
            mu_kt = MU[k, t].astype(np.float32)
            sg_kt = SG[k, t].astype(np.float32)
            if swap_axes:
                mu_kt = mu_kt[[1, 0]]
                sg_kt = sg_kt[[1, 0]]
            fwd, lat = float(mu_kt[0]), float(mu_kt[1])
            if flip_lat:
                lat = -lat
            center = (int(cx + lat * px_per_m_lat),
                      int(cy - fwd * px_per_m_fwd))
            axes = (max(1, int(n_std * sg_kt[1] * px_per_m_lat)),
                    max(1, int(n_std * sg_kt[0] * px_per_m_fwd)))

            # Draw all rings for this (k, t) on temp_ring
            for i, s in enumerate((1.0, 1.5, 2.0)):
                # Draw directly, no per-ring blending
                cv2.ellipse(temp_ring, center, (int(axes[0]*s), int(axes[1]*s)),
                            0, 0, 360, comp_color, 2, cv2.LINE_AA)

        # Blend once per component
        overlay = cv2.addWeighted(overlay, 1.0, temp_ring, alpha * ws[k], 0)

    return overlay