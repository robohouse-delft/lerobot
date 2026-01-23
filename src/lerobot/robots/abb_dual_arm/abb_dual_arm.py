import logging
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pyparsing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
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


# This is only needed if you want to convert an empty or incomplete action (e.g. from the VR setup) to a valid action (e.g. the current robot pose).
@ProcessorStepRegistry.register("abb_dual_arm_empty_action_processor")
@dataclass
class ABBDualArmEmptyActionProcessor(RobotActionProcessorStep):
    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is required to replace an empty action")

        # Convert to a dictionary that only has keys that ends with ".pos"
        obs = {k: v for k, v in observation.items() if k.endswith(".pos")}

        if obs is None:
            raise ValueError("Pose observation is require to replace an empty action")

        # print(obs)
        if "left_x.pos" not in action and "right_x.pos" not in action:
            action = obs
        elif "left_x.pos" not in action:
            action = {k: v for k, v in obs.items() if k.startswith("left")}
        elif "right_x.pos" not in action:
            action = {k: v for k, v in obs.items() if k.startswith("right")}

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


# This is only needed if you want to convert from delta actions to pose actions, i.e. for keyboard teleoperation.
@ProcessorStepRegistry.register("abb_dual_arm_delta_to_pose")
@dataclass
class ABBDualArmDeltaToPose(RobotActionProcessorStep):
    end_effector_step_sizes: dict

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is required for computing next pose")

        # Convert to a dictionary that only has keys that ends with ".pos"
        obs = {k: v for k, v in observation.items() if k.endswith(".pos")}

        if obs is None:
            raise ValueError("Pose observation is require for computing next pose")

        self._transform_action(action, obs, prefix="left")
        self._transform_action(action, obs, prefix="right")

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "left_delta_x",
            "left_delta_y",
            "left_delta_z",
            "left_gripper",
            "right_delta_x",
            "right_delta_y",
            "right_delta_z",
            "right_gripper",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in [
            "left_x",
            "left_y",
            "left_z",
            "left_qw",
            "left_qx",
            "left_qy",
            "left_qz",
            "left_gripper",
            "right_x",
            "right_y",
            "right_z",
            "right_qw",
            "right_qx",
            "right_qy",
            "right_qz",
            "right_gripper",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def _transform_action(self, action: RobotAction, observation: Any, prefix: str):
        tx = float(action.pop(f"{prefix}_delta_x"))
        ty = float(action.pop(f"{prefix}_delta_y"))
        tz = float(action.pop(f"{prefix}_delta_z"))
        gripper_pos = float(action.pop(f"{prefix}_gripper"))

        delta_p = np.array(
            [
                tx * self.end_effector_step_sizes["x"],
                ty * self.end_effector_step_sizes["y"],
                tz * self.end_effector_step_sizes["z"],
            ],
            dtype=float,
        )

        # Write action fields
        action[f"{prefix}_x.pos"] = float((delta_p[0] + observation[f"{prefix}_x.pos"]) / 1000.0)
        action[f"{prefix}_y.pos"] = float((delta_p[1] + observation[f"{prefix}_y.pos"]) / 1000.0)
        action[f"{prefix}_z.pos"] = float((delta_p[2] + observation[f"{prefix}_z.pos"]) / 1000.0)
        action[f"{prefix}_qx.pos"] = float(observation[f"{prefix}_qx.pos"])
        action[f"{prefix}_qy.pos"] = float(observation[f"{prefix}_qy.pos"])
        action[f"{prefix}_qz.pos"] = float(observation[f"{prefix}_qz.pos"])
        action[f"{prefix}_qw.pos"] = float(observation[f"{prefix}_qw.pos"])
        # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
        if gripper_pos == 0:
            action[f"{prefix}_gripper.pos"] = 1.0
        elif gripper_pos == 2:
            action[f"{prefix}_gripper.pos"] = 0.0
        else:
            action[f"{prefix}_gripper.pos"] = observation[f"{prefix}_gripper.pos"]
