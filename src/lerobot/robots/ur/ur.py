from functools import cached_property

import numpy as np
import rtde_control
import rtde_receive
from pyparsing import Any
from rtde_control import RTDEControlInterface as RTDEControl

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
        self.config = config
        self.start_position = np.deg2rad(config.start_position_deg)
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
        current_pose = self.r_inter.getActualTCPPose()
        self.prev_pos = current_pose[0:3]

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
        joint_vector = self._get_joint_state()
        for i, v in enumerate(joint_vector):
            obs_dict[f"joint_{i}"] = v
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

        if self.r_inter.isProtectiveStopped() or self.r_inter.isEmergencyStopped():
            print("Error: Robot is in either a protective stop or emergency stop state")
            return {}

        if "joint" not in list(action.keys())[0]:
            raise ValueError("Invalid state type")

        action_vals = [val for _, val in action.items()]
        robot_joints = action_vals[:6]
        # Check limits end-effector workspace limits before commanding the robot
        within_limits, _ = self._is_within_limits(robot_joints)
        if not within_limits:
            if not self.outside_workspace_limits:
                print(
                    "Error: Robot end-effector has been commanded to be outside of the workspace limits. Move leader arm back to within workspace."
                )
            self.outside_workspace_limits = True
            return {}
        elif self.outside_workspace_limits:
            print("Robot end-effector back inside the workspace")
            self.outside_workspace_limits = False

        t_start = self.robot.initPeriod()
        self.robot.servoJ(robot_joints, velocity, acceleration, dt, lookahead_time, gain)
        gripper_pos = action_vals[-1] * 255
        self.gripper.move(int(gripper_pos), 255, self.config.max_gripper_force)
        self.robot.waitPeriod(t_start)

        committed_action = {f"joint_{i}": v for i, v in enumerate(robot_joints)}
        committed_action["gripper"] = action_vals[-1]

        return committed_action

    @property
    def _motors_ft(self) -> dict[str, type]:
        # TODO: Maybe use a better name?
        return {
            "joint_0": float,
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
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

    def _get_gripper_pos(self) -> float:
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def _get_joint_state(self) -> np.ndarray:
        return self.r_inter.getActualQ()

    def _is_within_limits(self, q_vec: list | np.ndarray) -> tuple[bool, np.ndarray | None]:
        try:
            pose = self.robot.getForwardKinematics(q_vec, self.tcp_offset)
            within_limits = self._check_limits(pose)
            return within_limits, pose
        except RuntimeError as e:
            print(e)
            return False, None

    def _check_limits(self, pose: list[float]) -> bool:
        return not (
            pose[0] < self.config.x_limits[0]
            or pose[0] > self.config.x_limits[1]
            or pose[1] < self.config.y_limits[0]
            or pose[1] > self.config.y_limits[1]
            or pose[2] < self.config.z_limits[0]
            or pose[2] > self.config.z_limits[1]
        )
