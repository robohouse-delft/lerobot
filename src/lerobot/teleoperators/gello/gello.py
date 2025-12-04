import logging
import time

import numpy as np

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
        self.robot_state_names = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
        self.start_joints = np.deg2rad(config.start_joints_deg)
        self.bus = DynamixelRobot(
            joint_ids=[1, 2, 3, 4, 5, 6],
            joint_offsets=config.joint_offsets,
            joint_signs=[1, 1, -1, 1, 1, 1],
            port=config.port,
            gripper_config=config.gripper_config,
            start_joints=self.start_joints,
        )

    @property
    def action_features(self) -> dict[str, type]:
        joints_dict = {f"{name}.pos": float for name in self.robot_state_names}
        joints_dict["gripper.pos"] = float
        return joints_dict

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
        # Ensure that we are close to the starting position before enabling operation
        max_delta_check = 0.2
        curr_joints = self.bus.get_joint_state()[:-1]
        abs_deltas = np.abs(self.start_joints - curr_joints)
        id_max_joint_delta = np.argmax(abs_deltas)
        if abs_deltas[id_max_joint_delta] > max_delta_check:
            id_mask = abs_deltas > max_delta_check
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                self.start_joints[id_mask],
                curr_joints[id_mask],
                strict=False,
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            raise RuntimeError("The initial position is too far from the robot's current position.")

    def setup_motors(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.get_joint_state()
        action_dict = {
            f"{name}.pos": val for name, val in zip(self.robot_state_names, action[:-1], strict=False)
        }
        action_dict["gripper.pos"] = action[-1]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
