from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("abb_dual_arm")
@dataclass
class ABBDualArmConfig(RobotConfig):
    state_feedback_hz: float
    left_port: int
    right_port: int
