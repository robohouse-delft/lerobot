from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("abb")
@dataclass
class ABBConfig(RobotConfig):
    state_feedback_hz: float
    port: int
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
