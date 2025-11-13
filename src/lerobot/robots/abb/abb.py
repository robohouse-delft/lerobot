import time
from functools import cached_property

import numpy as np
from ABBRobotEGM import EGM
from pyparsing import Any
from scipy.spatial.transform import Rotation

from lerobot.robots import Robot

from .config_abb import ABBConfig


class ABB(Robot):
    config_class = ABBConfig
    name = "abb"

    def __init__(self, config: ABBConfig):
        super().__init__(config)
        self.x_limits: list[float] = [-1.0, 1.0]
        self.y_limits: list[float] = [-0.4, 0.6]
        self.z_limits: list[float] = [0.0, 1.5]
        self.max_translation_delta_mm: float = 50.0
        self.config = config
        self.outside_workspace_limits = False
        self.robot: EGM | None = None

    def connect(self, calibrate: bool = True) -> None:
        try:
            self.robot = EGM(port=self.config.port)
            print(f"Connected to EGM server at {self.robot.egm_addr}")
        except Exception as e:
            print(e)
            raise ConnectionError(e)  # noqa: B904

        self.configure()

    def disconnect(self) -> None:
        self.robot.close()

    def configure(self) -> None:
        self._flush_robot_msgs()
        success, state = self.robot.receive_from_robot(timeout=0.1)
        if success:
            print(f"Current robot pose: {state.cartesian}")
            self.prev_rot = Rotation.from_quat([state.cartesian.orient.u1, state.cartesian.orient.u2, state.cartesian.orient.u3, state.cartesian.orient.u0])
            self.prev_pos = np.array([state.cartesian.pos.x, state.cartesian.pos.y, state.cartesian.pos.z])
        else:
            raise RuntimeError("Robot problem!")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs_dict = {}
        # Read arm state
        pose_vector = self._get_pose_state()
        for i, v in enumerate(pose_vector):
            obs_dict[f"pose_{i}"] = v

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        t_start = time.perf_counter()
        dt = 1.0 / self.config.state_feedback_hz

        self._flush_robot_msgs()
        success, state = self.robot.receive_from_robot()
        if not success:
            raise RuntimeError("Robot problem!")

        action_vals = [val for _, val in action.items()]
        if "delta" in list(action.keys())[0]:
            delta_x = action.pop("delta_x")
            delta_y = action.pop("delta_y")
            delta_z = action.pop("delta_z")
            pose = [
                state.cartesian.pos.x + delta_x * 10.0,
                state.cartesian.pos.y + delta_y * 10.0,
                state.cartesian.pos.z + delta_z * 10.0,
                state.cartesian.orient.u0,
                state.cartesian.orient.u1,
                state.cartesian.orient.u2,
                state.cartesian.orient.u3,
            ]
        elif "pose" in list(action.keys())[0]:
            # Action is already in end-effector pose format [x, y, z, rot_x, rot_y, rot_z, rot_w]
            pose = action_vals
        else:
            raise ValueError("Invalid state length")

        # Check limits end-effector workspace limits before commanding the robot
        if not self._check_limits(pose):
            if not self.outside_workspace_limits:
                print(
                    "Error: Robot end-effector has been commanded to be outside of the workspace limits. Move leader arm back to within workspace."
                )
            self.outside_workspace_limits = True
            return None
        elif self.outside_workspace_limits:
            print("Robot end-effector back inside the workspace")
            self.outside_workspace_limits = False

        # Ensure that the delta movement is within the set threshold
        # TODO: should we consider rotation as well?
        pos_diff = np.array(pose[:3]) - np.array(self.prev_pos)
        if np.linalg.norm(pos_diff) > self.max_translation_delta_mm:
            direction = pos_diff / np.linalg.norm(pos_diff)
            pose[:3] = self.prev_pos + (direction * self.max_translation_delta_mm).tolist()
            print("Warning: TCP pose is too far from current pose. Clipped translation to:", pose[:3])

        # Send to robot
        self.robot.send_to_robot(cartesian=(np.array(pose[:3]), np.array(pose[3:])))

        # Convert orientation to rotation vector
        corrected_quat = self._normalise_quat(np.array(pose[3:7]))
        self.prev_pos = pose[:3]

        committed_pose_action = pose[:3] + corrected_quat.tolist()
        committed_action = {f"pose_{i}": v for i, v in enumerate(committed_pose_action)}

        # Wait for cycle to complete
        t_end = time.perf_counter()
        if t_end - t_start < dt:
            time.sleep(dt - (t_end - t_start))

        return committed_action

    @property
    def _motors_ft(self) -> dict[str, type]:
        # TODO: Maybe use a better name?
        return {
            "pose_0": float,
            "pose_1": float,
            "pose_2": float,
            "pose_3": float,
            "pose_4": float,
            "pose_5": float,
            "pose_6": float,
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return not self.robot.socket._closed

    def _flush_robot_msgs(self):
        while True:
            success, _ = self.robot.receive_from_robot(timeout=0.002)
            if not success:
                break

    def _get_pose_state(self) -> np.ndarray:
        try:
            self._flush_robot_msgs()
            success, state = self.robot.receive_from_robot(timeout=0.02)
            if not success:
                raise RuntimeError("Robot problem!")

            pos = [
                state.cartesian.pos.x,
                state.cartesian.pos.y,
                state.cartesian.pos.z,
            ]
            quat = np.array(
                [
                    state.cartesian.orient.u1,  # x,
                    state.cartesian.orient.u2,  # y,
                    state.cartesian.orient.u3,  # z
                    state.cartesian.orient.u0,  # w,
                ]
            )
            corrected_quat = self._normalise_quat(quat)
            return np.array(pos + corrected_quat.tolist())
        except RuntimeError as e:
            print(e)
            return np.zeros(6)

    def _check_limits(self, pose: list[float]) -> bool:
        pos_m = np.array(pose[:3]) / 1000.0
        return not (
            pos_m[0] < self.x_limits[0]
            or pos_m[0] > self.x_limits[1]
            or pos_m[1] < self.y_limits[0]
            or pos_m[1] > self.y_limits[1]
            or pos_m[2] < self.z_limits[0]
            or pos_m[2] > self.z_limits[1]
        )

    def _ensure_rotation_continuity(self, rot: Rotation):
        # Make quaternion sign continuous: choose sign so dot >= 0
        if np.dot(self.prev_rot.as_quat(), rot.as_quat()) < 0:
            return rot.inv()
        return rot

    def _normalise_quat(self, quat: np.ndarray) -> np.ndarray:
        curr_rot = Rotation.from_quat(quat)
        corrected_rot = self._ensure_rotation_continuity(curr_rot)
        self.prev_rot = corrected_rot

        return corrected_rot.as_quat()
