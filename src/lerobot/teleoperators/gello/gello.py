import logging
import time

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_gello import GelloConfig
from .dynamixel_robot import DynamixelRobot

logger = logging.getLogger(__name__)


class Gello(Teleoperator):
    """
    GELLO: General, Low-Cost, and Intuitive Teleoperation Framework (https://github.com/wuphilipp/gello_software)

    Note that this assumes that you have used the GELLO repository to calibrate the device!
    """

    config_class = GelloConfig
    name = "gello"

    def __init__(self, config: GelloConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelRobot(
            joint_ids=config.joint_ids,
            joint_offsets=config.joint_offsets,
            joint_signs=config.joint_signs,
            port=config.port,
            gripper_config=config.gripper_config,
            start_joints=config.start_joints,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"joint_{i}": float for i in range(self.bus.num_dofs())}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.bus.configure()

    def setup_motors(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.get_joint_state()
        action = {f"joint_{i}": val for i, val in enumerate(action)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
