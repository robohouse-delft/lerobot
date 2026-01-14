import threading
import time
from functools import cached_property

import numpy as np
from ABBRobotEGM import EGM
from pyparsing import Any
from scipy.spatial.transform import Rotation

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_abb import ABBConfig


class ABB(Robot):
    config_class = ABBConfig
    name = "abb"

    def __init__(self, config: ABBConfig):
        super().__init__(config)
        # TODO: need to update these from configuration probably
        self.x_limits: list[float] = [0.3, 0.6]
        self.y_limits: list[float] = [-0.3, 0.3]
        self.z_limits: list[float] = [0.0, 0.4]
        self.max_translation_delta_mm: float = 30.0
        self.state_names = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        self.config = config
        self.outside_workspace_limits = False
        self.robot: EGM | None = None
        self.prev_gripper = 0
        self.robot_stop_event = threading.Event()
        self.egm_thread = None
        self.robot_state_lock = threading.Lock()
        self.latest_robot_state = None
        self.last_pose_found = False
        self.cameras = make_cameras_from_configs(config.cameras)

    def connect(self, calibrate: bool = True) -> None:
        try:
            self.robot = EGM(port=self.config.port)
        except Exception as e:
            print(e)
            raise ConnectionError(e)  # noqa: B904

        self.robot_stop_event.clear()
        self.egm_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.egm_thread.start()

        # Wait for the first state to arrive
        print("Waiting for robot state...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            with self.robot_state_lock:
                if self.latest_robot_state is not None:
                    break
            time.sleep(0.01)
        else:
            print("Warning: No state received from robot yet.")

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def _listen_loop(self):
        # Flush the initial messages to ensure we have the latest going forward
        while True:
            success, _ = self.robot.receive_from_robot(timeout=0.004)
            if not success:
                break
        # Now start the main listening loop
        while not self.robot_stop_event.is_set():
            if self.robot is None:
                time.sleep(0.01)
                continue
            try:
                success, state = self.robot.receive_from_robot(timeout=0.004)
                if success:
                    with self.robot_state_lock:
                        self.latest_robot_state = state
            except Exception:
                time.sleep(0.01)

    def disconnect(self) -> None:
        self.robot_stop_event.set()
        if self.egm_thread is not None:
            self.egm_thread.join()
        self.robot.send_to_robot(
            cartesian=(np.array(self.prev_pos), self.prev_rot.as_quat()),
            rapid_to_robot=np.array([0, self.prev_gripper]),
        )
        self.robot.close()
        for cam in self.cameras.values():
            cam.disconnect()

    def configure(self) -> None:
        self.last_pose_found = False
        with self.robot_state_lock:
            state = self.latest_robot_state

        if state is not None:
            print(f"Current robot pose: {state.cartesian}")
            self.prev_rot = Rotation.from_quat(
                [
                    state.cartesian.orient.u1,
                    state.cartesian.orient.u2,
                    state.cartesian.orient.u3,
                    state.cartesian.orient.u0,
                ]
            )
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
        for val, key in zip(pose_vector, self.state_names, strict=False):
            obs_dict[f"{key}.pos"] = val
        obs_dict["gripper.pos"] = self.prev_gripper

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        with self.robot_state_lock:
            state = self.latest_robot_state

        if state is None or not state.rapid_running:
            raise RuntimeError("Robot problem!")

        if state.collision_info.collsionTriggered:
            print("Robot collision or motion has stopped!")
            self.robot.send_to_robot(
                cartesian=(np.array(self.prev_pos), self.prev_rot.as_quat()),
                rapid_to_robot=np.array([0, self.prev_gripper]),
            )
            return {}

        action_vals = [val for _, val in action.items()]
        if len(action_vals) == 0:
            # Do nothing
            return action

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
            # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
            if action_vals[-1] == 0:
                action_vals[-1] = 1.0
            elif action_vals[-1] == 2:
                action_vals[-1] = 0.0
            elif action_vals[-1] == 1:
                action_vals[-1] = self.prev_gripper
        elif "qx" in list(action.keys())[3]:
            # Action is already in end-effector pose format [x, y, z, qx, qy, qz, qw]
            pose = (np.array(action_vals[:3]) * 1000.0).tolist() + [
                action_vals[6],
                action_vals[3],
                action_vals[4],
                action_vals[5],
            ]
        else:
            raise ValueError("Invalid state length")

        # Check limits end-effector workspace limits before commanding the robot
        if not self._check_limits(pose):
            if not self.outside_workspace_limits:
                print(
                    "Error: Robot end-effector has been commanded to be outside of the workspace limits. Move leader arm back to within workspace."
                )
            self.outside_workspace_limits = True
            self.last_pose_found = False
            self.robot.send_to_robot(
                cartesian=(
                    np.array([state.cartesian.pos.x, state.cartesian.pos.y, state.cartesian.pos.z]),
                    np.array(
                        [
                            state.cartesian.orient.u0,
                            state.cartesian.orient.u1,
                            state.cartesian.orient.u2,
                            state.cartesian.orient.u3,
                        ]
                    ),
                ),
                rapid_to_robot=np.array([1, self.prev_gripper]),
            )
            return {}
        elif self.outside_workspace_limits:
            print("Robot end-effector back inside the workspace")
            self.outside_workspace_limits = False

        # Ensure that we have successfully found the initial pose (or pose when left workspace)
        if not self.last_pose_found:
            current_pose = np.array([state.cartesian.pos.x, state.cartesian.pos.y, state.cartesian.pos.z])
            pos_diff = np.array(pose[:3]) - np.array(current_pose)
            if np.linalg.norm(pos_diff) > self.max_translation_delta_mm:
                print(
                    f"Error: Last pose inside workspace too far from current pose: {pose[:3]} > {current_pose}"
                )
                return {}
            else:
                self.last_pose_found = True

        # Send to robot
        self.robot.send_to_robot(
            cartesian=(np.array(pose[:3]), np.array(pose[3:])), rapid_to_robot=np.array([1, action_vals[-1]])
        )

        # Convert orientation to rotation vector
        corrected_quat = self._normalise_quat(np.array(pose[3:7]))
        self.prev_pos = pose[:3]
        self.prev_gripper = action_vals[-1]

        committed_pose_action = pose[:3] + corrected_quat.tolist()
        committed_action = {
            f"{key}.pos": val for val, key in zip(committed_pose_action, self.state_names, strict=False)
        }

        return committed_action

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Following the standard naming convention to append a `.pos` for position
        return {f"{name}.pos": float for name in self.state_names} | {"gripper.pos": float}

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
        return not self.robot.socket._closed and all(cam.is_connected for cam in self.cameras.values())

    def _get_pose_state(self) -> np.ndarray:
        try:
            with self.robot_state_lock:
                state = self.latest_robot_state

            if state is None:
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
            return np.array(pos + corrected_quat.tolist(), dtype=np.float32)
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
