from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("ur")
@dataclass
class URConfig(RobotConfig):
    ip_address: str
    state_feedback_hz: float
    gripper_hostname: str
    gripper_port: int
