from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("abb")
@dataclass
class ABBConfig(RobotConfig):
    state_feedback_hz: float
