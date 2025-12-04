from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig):
    port: str
    joint_offsets: tuple[float, ...]
    gripper_config: tuple[int, float, float] | None = None  # (id, open_angle_deg, close_angle_deg)
    start_joints_deg: list = field(default_factory = lambda: [0.0, -90.0, -90.0, -90.0, 90.0, -180.0])
