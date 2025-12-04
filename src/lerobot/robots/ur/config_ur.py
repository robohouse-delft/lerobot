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
    max_gripper_force: int
    start_position_deg: list = field(default_factory = lambda: [0.0, -90.0, -90.0, -90.0, 90.0, -180.0])
    x_limits: tuple[float, float] = (-1.0, 1.0)
    y_limits: tuple[float, float] = (-0.4, 0.6)
    z_limits: tuple[float, float] = (0.0, 1.5)
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
