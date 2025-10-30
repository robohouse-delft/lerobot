from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig):
    port: str
    joint_offsets: tuple[float, ...]
    gripper_config: tuple[int, float, float] | None = None  # (id, open_angle_deg, close_angle_deg)
    start_joints_deg: tuple[float, ...] | None = None
