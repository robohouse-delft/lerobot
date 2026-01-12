from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("abb_dual_arm")
@dataclass
class ABBDualArmConfig(RobotConfig):
    state_feedback_hz: float
    left_port: int
    right_port: int
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
