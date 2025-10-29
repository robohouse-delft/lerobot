from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig):
    port: str
    joint_ids: tuple[int, ...]
    joint_offsets: tuple[float, ...]
    joint_signs: tuple[int, ...]
    gripper_config: tuple[int, float, float] | None = None  # (id, open_angle_deg, close_angle_deg)
    start_joints: tuple[float] | None = None
