from functools import cached_property

import numpy as np
import rtde_control
import rtde_receive
from pyparsing import Any
from rtde_control import RTDEControlInterface as RTDEControl
from scipy.spatial.transform import Rotation

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_ur import URConfig
from .robotiq_gripper import RobotiqGripper


class UR(Robot):
    config_class = URConfig
    name = "ur"

    def __init__(self, config: URConfig):
        super().__init__(config)
        self.freq_hz = 500.0
        self.start_position: list | np.ndarray = np.deg2rad([0.0, -90.0, -90.0, -90.0, 90.0, -180.0])
        self.x_limits: list[float] = [-1.0, 1.0]
        self.y_limits: list[float] = [-0.4, 0.6]
        self.z_limits: list[float] = [0.0, 1.5]
        self.max_translation_delta_m: float = 0.05
        self.config = config
        self.outside_workspace_limits = False
        self.gripper = RobotiqGripper()
        self.cameras = make_cameras_from_configs(config.cameras)

    def connect(self, calibrate: bool = True) -> None:
        try:
            self.robot = rtde_control.RTDEControlInterface(
                self.config.ip_address, self.freq_hz, RTDEControl.FLAG_USE_EXT_UR_CAP
            )
        except Exception as e:
            print(e)
            print(self.config.ip_address)
            raise ConnectionError(e)  # noqa: B904

        self.r_inter = rtde_receive.RTDEReceiveInterface(self.config.ip_address)

        self.gripper.connect(hostname=self.config.gripper_hostname, port=self.config.gripper_port)
        self.gripper.activate(auto_calibrate=False)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def disconnect(self) -> None:
        self.gripper.disconnect()
        self.robot.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    def configure(self) -> None:
        self.robot.endFreedriveMode()
        self.robot.moveJ(self.start_position)
        self.tcp_offset = self.robot.getTCPOffset()
        self.prev_rot = Rotation.from_quat([0, 0, 0, 1])
        self.prev_pos = self._get_pose_state()[0:3]

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
        obs_dict["gripper"] = self._get_gripper_pos()
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        velocity = 0.0
        acceleration = 0.0
        dt = 1.0 / self.config.state_feedback_hz
        lookahead_time = 0.2
        gain = 100

        action_vals = [val for _, val in action.items()]
        if "joint" in list(action.keys())[0]:
            # Convert joint space to end-effector pose
            robot_joints = action_vals[:6]
            try:
                pose = self.robot.getForwardKinematics(robot_joints, self.tcp_offset)
            except RuntimeError as e:
                print(e)
                return None
        elif "pose" in list(action.keys())[0]:
            # Action is already in end-effector pose format [x, y, z, rot_w, rot_x, rot_y, rot_z]
            pose = action_vals[0:3] + Rotation.from_quat(action_vals[3:7]).as_rotvec().tolist()
        else:
            raise ValueError("Invalid state length")

        # Check limits end-effector workspace limits before commanding the robot
        if not self._check_limits(pose):
            if not self.outside_workspace_limits:
                print(
                    "Robot end-effector has been commanded to be outside of the workspace limits. Move leader arm back to within workspace."
                )
            self.outside_workspace_limits = True
            return None
        elif self.outside_workspace_limits:
            print("Robot end-effector back inside the workspace")
            self.outside_workspace_limits = False

        # Ensure that the delta movement is within the set threshold
        # TODO: should we consider rotation as well?
        pos_diff = np.array(pose[:3]) - np.array(self.prev_pos)
        if np.linalg.norm(pos_diff) > self.max_translation_delta_m:
            direction = pos_diff / np.linalg.norm(pos_diff)
            pose[:3] = self.prev_pos + (direction * self.max_translation_delta_m).tolist()
            print("TCP pose is too far from current pose. Clipped translation to:", pose[:3])

        t_start = self.robot.initPeriod()
        self.robot.servoL(pose, velocity, acceleration, dt, lookahead_time, gain)
        gripper_pos = action_vals[-1] * 255
        self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)

        # Convert orientation to rotation vector
        corrected_quat = self._convert_rotvec_to_quat(np.array(pose[3:6]))
        self.prev_pos = pose[:3]

        committed_pose_action = pose[:3] + corrected_quat.tolist()
        committed_action = {f"pose_{i}": v for i, v in enumerate(committed_pose_action)}
        committed_action["gripper"] = action_vals[-1]

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
            "gripper": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.r_inter.isConnected() and all(cam.is_connected for cam in self.cameras.values())

    def _get_pose_state(self) -> np.ndarray:
        try:
            pose = self.r_inter.getActualTCPPose()
            corrected_quat = self._convert_rotvec_to_quat(np.array(pose[3:6]))
            return np.array(pose[0:3] + corrected_quat.tolist())
        except RuntimeError as e:
            print(e)
            return np.zeros(6)

    def _get_gripper_pos(self) -> float:
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def _get_joint_state(self) -> np.ndarray:
        return self.r_inter.getActualQ()

    def _is_within_limits(self, q_vec: np.ndarray) -> bool:
        within_limits = True
        try:
            pose = self.robot.getForwardKinematics(q_vec, self.tcp_offset)
            within_limits = self._check_limits(pose)
        except RuntimeError as e:
            print(e)
            within_limits = False
        return within_limits

    def _check_limits(self, pose: list[float]) -> bool:
        return not (
            pose[0] < self.x_limits[0]
            or pose[0] > self.x_limits[1]
            or pose[1] < self.y_limits[0]
            or pose[1] > self.y_limits[1]
            or pose[2] < self.z_limits[0]
            or pose[2] > self.z_limits[1]
        )

    def _ensure_rotation_continuity(self, rot: Rotation):
        # Make quaternion sign continuous: choose sign so dot >= 0
        if np.dot(self.prev_rot.as_quat(), rot.as_quat()) < 0:
            return rot.inv()
        return rot

    def _convert_rotvec_to_quat(self, rotvec: np.ndarray) -> np.ndarray:
        curr_rot = Rotation.from_rotvec(rotvec)
        corrected_rot = self._ensure_rotation_continuity(curr_rot)
        self.prev_rot = corrected_rot

        return corrected_rot.as_quat()
