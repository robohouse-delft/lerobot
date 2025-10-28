from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("ur")
@dataclass
class URConfig(RobotConfig):
    ip_address: str
    state_feedback_hz: float
    gripper_hostname: str
    gripper_port: int
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist.rgb": RealSenseCameraConfig(
                serial_number_or_name="218622274553",
                color_mode="rgb",
                fps=30,
                width=640,
                height=480,
            ),
            "base.rgb": RealSenseCameraConfig(
                serial_number_or_name="916512060630",
                color_mode="rgb",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
