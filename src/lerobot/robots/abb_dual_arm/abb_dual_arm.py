import logging
import time
from functools import cached_property

from pyparsing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.abb import ABB
from lerobot.robots.abb.config_abb import ABBConfig

from .config_abb_dual_arm import ABBDualArmConfig

logger = logging.getLogger(__name__)


class ABBDualArm(Robot):
    config_class = ABBDualArmConfig
    name = "abb_dual_arm"

    def __init__(self, config: ABBDualArmConfig):
        super().__init__(config)
        self.config = config
        left_arm_config = ABBConfig(
            port=self.config.left_port, state_feedback_hz=self.config.state_feedback_hz, cameras={}
        )
        right_arm_config = ABBConfig(
            port=self.config.right_port, state_feedback_hz=self.config.state_feedback_hz, cameras={}
        )
        self.left_arm = ABB(left_arm_config)
        self.right_arm = ABB(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect()
        self.right_arm.connect()
        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs_dict = {}
        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        prefixed_send_action_left = {}
        prefixed_send_action_right = {}
        if left_action != {}:
            send_action_left = self.left_arm.send_action(left_action)
            prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        if right_action != {}:
            send_action_right = self.right_arm.send_action(right_action)
            prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{key}": float for key in self.left_arm.action_features} | {
            f"right_{key}": float for key in self.right_arm.action_features
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )
