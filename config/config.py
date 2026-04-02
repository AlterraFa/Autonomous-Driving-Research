from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrafficManagerConfig:
    excluded_junctions: tuple[int, ...] = (
        683,
        560,
        408,
        416,
        636,
        455,
        706,
        691,
        644,
        399,
        674,
        295,
    )


@dataclass(frozen=True)
class MapRenderConfig:
    rect_dim: tuple[int, int] = (4, 3)
    map_offset: tuple[int, int] = (100, 100)
    map_range: tuple[int, int] = (50, 50)
    map_resize: tuple[int, int] = (200, 200)
    map_scale: int = 2


@dataclass(frozen=True)
class PathOptimizerConfig:
    path_step: float = 2.0
    exclude_params: tuple[float, float, float] = (322.5, -195.5, 19.0)


@dataclass(frozen=True)
class RandPathConfig:
    min_distant_node: int = 20
    max_distant_node: int = 200
    path_iter: int = 2


@dataclass(frozen=True)
class GPSConfig:
    mean_delay: int = 550
    stddev_delay: int = 200
    lat_stddev: float = 1.5
    lon_stddev: float = 1.5
    frequency: int = 10
    max_gps_delay: int = 60
    min_gps_delay: int = 10
    meters_per_degree: float = 111320.0


@dataclass(frozen=True)
class UIConfig:
    alpha: float = 0.05
    font_size: int = 12
    text_height: int = 20
    line_h: int = 20
    bar_w: int = 150
    bar_h: int = 10
    font_name: str = "jetbrainsmononerdfontpropo"


@dataclass(frozen=True)
class OffsetsConfig:
    front_offset: float = 1.5
    temporal_offset: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)
    spatial_offset: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12)
    scout_offset_params: tuple[int, int, int] = (-18, 33, 2)


@dataclass(frozen=True)
class SpawnConfig:
    num_npc: int = 30


@dataclass(frozen=True)
class VehiclePhysicsConfig:
    max_steer: float = 70.0
    wheelbase: float = 3.047080078125


@dataclass(frozen=True)
class VehicleControlFilterConfig:
    fs: float = 2.0
    x0: float = 2.0


@dataclass(frozen=True)
class VehicleVelocityRegulatorConfig:
    kp: float = 0.1
    ki: float = 0.05
    kd: float = 0.05


@dataclass(frozen=True)
class VehicleConfig:
    decay: float = 0.2
    steer_multiplier: float = 1.2
    physics: VehiclePhysicsConfig = field(default_factory=VehiclePhysicsConfig)
    control_filter: VehicleControlFilterConfig = field(default_factory=VehicleControlFilterConfig)
    velocity_regulator: VehicleVelocityRegulatorConfig = field(default_factory=VehicleVelocityRegulatorConfig)


@dataclass(frozen=True)
class PictureConfig:
    quality: int = 90


@dataclass(frozen=True)
class ReplayConfig:
    start_at: float = 0.0
    stop_at: float = -1.0
    position_idx: int = 0


@dataclass(frozen=True)
class SensorConfig:
    imu_gyro_bias_x: float = 0.005
    imu_gyro_bias_y: float = 0.005


@dataclass(frozen=True)
class ControllerConfig:
    joystick_deadzone_stick: float = 0.12
    joystick_deadzone_trigger: float = 0.05
    steer_curve_exponent: float = 3.0
    keyboard_steer_rate: float = 5e-4
    keyboard_throttle_step: float = 0.01
    keyboard_brake_step: float = 0.2


@dataclass(frozen=True)
class RenderingConfig:
    fps_headless: int = 100
    playback_rate_window_frames: int = 30
    border_thickness_px: int = 3


@dataclass(frozen=True)
class PathHandlerConfig:
    spline_min_points: int = 20
    spline_points_multiplier: int = 5
    smoothing_blend_half_window: int = 4
    smoothing_window_size: int = 3
    align_tolerance: float = 1e-2
    spline_deduplication_tolerance: float = 1e-3
    b_smooth_s: float = 2.0
    b_smooth_k: int = 3


@dataclass(frozen=True)
class ControlConfig:
    lateral_lookahead_distance: float = 10.0
    longitudinal_waypoints_average: int = 3
    longitudinal_time_step: float = 0.2
    longitudinal_min_speed_ms: float = 10.0


@dataclass(frozen=True)
class WorldConfig:
    fixed_delta_seconds: float = 0.05
    junction_cache_max_size: int = 5000
    waypoint_rounding_decimals: int = 2


@dataclass(frozen=True)
class DataCollectionConfig:
    trajectory_buffer_capacity: int = 8192 * 8
    trajectory_distance_threshold_m: float = 0.0
    trajectory_min_dt_s: float = 0.05
    save_fps: int = 10
    additional_trajectory_max: int = 20


@dataclass(frozen=True)
class ReplayRuntimeConfig:
    duration_padding_s: float = 10.0
    stability_wait_iterations: int = 50
    stability_sleep_s: float = 0.05
    actor_spawn_timeout_s: float = 30.0
    actor_settle_ticks: int = 30
    final_wait_s: float = 1.0


@dataclass(frozen=True)
class ContractingWPConfig:
    enabled: bool = True
    containment_mode: str = "circle"
    k_nearest: int = 5
    ref_point_idx: int = 0
    containment_eps: float = 1e-3
    local_wp_inside_bbox_z_offset: float = 1.0
    local_wp_camera_z_offset: float = 0.7
    min_remaining_s: float = 0.0


@dataclass(frozen=True)
class TurnDetectionConfig:
    threshold_deg: float = 20.0


@dataclass(frozen=True)
class SimulationConfig:
    traffic_manager: TrafficManagerConfig = field(default_factory=TrafficManagerConfig)
    map_render: MapRenderConfig = field(default_factory=MapRenderConfig)
    path_optimizer: PathOptimizerConfig = field(default_factory=PathOptimizerConfig)
    rand_path: RandPathConfig = field(default_factory=RandPathConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    offsets: OffsetsConfig = field(default_factory=OffsetsConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    picture: PictureConfig = field(default_factory=PictureConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    path_handler: PathHandlerConfig = field(default_factory=PathHandlerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    replay_runtime: ReplayRuntimeConfig = field(default_factory=ReplayRuntimeConfig)
    contracting_wp: ContractingWPConfig = field(default_factory=ContractingWPConfig)
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.picture.quality <= 100:
            raise ValueError("picture.quality must be in [1, 100]")
        if self.gps.frequency <= 0:
            raise ValueError("gps.frequency must be > 0")
        if self.rendering.fps_headless <= 0:
            raise ValueError("rendering.fps_headless must be > 0")
        if len(self.offsets.spatial_offset) == 0 or len(self.offsets.temporal_offset) == 0:
            raise ValueError("offsets.spatial_offset and offsets.temporal_offset must be non-empty")
        if self.contracting_wp.k_nearest <= 0:
            raise ValueError("contracting_wp.k_nearest must be > 0")
        if self.contracting_wp.containment_mode not in ("obb", "circle"):
            raise ValueError("contracting_wp.containment_mode must be 'obb' or 'circle'")


CONFIG = SimulationConfig()
