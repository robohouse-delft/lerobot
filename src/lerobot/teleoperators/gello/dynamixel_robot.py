from collections.abc import Sequence

import numpy as np

from .dynamixel_driver import (
    DynamixelDriver,
    DynamixelDriverProtocol,
)


class DynamixelRobot:
    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Sequence[float] | None = None,
        joint_signs: Sequence[int] | None = None,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        gripper_config: tuple[int, float, float] | None = None,
        start_joints: np.ndarray | None = None,
    ):
        self._port = port
        self._baudrate = baudrate
        self._start_joints = start_joints
        self.gripper_open_close: tuple[float, float] | None
        if gripper_config is not None:
            assert joint_offsets is not None
            assert joint_signs is not None

            # joint_ids.append(gripper_config[0])
            # joint_offsets.append(0.0)
            # joint_signs.append(1)
            joint_ids = tuple(joint_ids) + (gripper_config[0],)
            joint_offsets = tuple(joint_offsets) + (0.0,)
            joint_signs = tuple(joint_signs) + (1,)
            self.gripper_open_close = (
                gripper_config[1] * np.pi / 180,
                gripper_config[2] * np.pi / 180,
            )
            self._start_joints = np.append(self._start_joints, [0.0])
        else:
            self.gripper_open_close = None

        self._joint_ids = joint_ids
        self._driver: DynamixelDriverProtocol

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(np.abs(self._joint_signs) == 1), f"joint_signs: {self._joint_signs}"

        self._torque_on = False
        self._last_pos = None
        self._alpha = 0.99
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self):
        print(f"attempting to connect to port: {self._port}")
        self._driver = DynamixelDriver(self._joint_ids, port=self._port, baudrate=self._baudrate)
        self._driver.set_torque_mode(False)
        self._connected = True

    def configure(self):
        # loop through all joints and add +- 2pi to the joint offsets to get the closest to start joints
        self._start_joints = np.asarray(self._start_joints)
        new_joint_offsets = []
        current_joints = self.get_joint_state()
        assert current_joints.shape == self._start_joints.shape
        if self.gripper_open_close is not None:
            current_joints = current_joints[:-1]
            self._start_joints = self._start_joints[:-1]
        for idx, (c_joint, s_joint, joint_offset) in enumerate(
            zip(current_joints, self._start_joints, self._joint_offsets, strict=False)
        ):
            new_joint_offsets.append(
                np.pi * 2 * np.round((-s_joint + c_joint) / (2 * np.pi)) * self._joint_signs[idx]
                + joint_offset
            )
        if self.gripper_open_close is not None:
            new_joint_offsets.append(self._joint_offsets[-1])
        self._joint_offsets = np.array(new_joint_offsets)

    def disconnect(self):
        self._driver.close()
        self._connected = False

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_state(self) -> np.ndarray:
        pos = (self._driver.get_joints() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos

        return pos

    def set_joint_state(self, joint_state: np.ndarray) -> None:
        self._driver.set_joints((joint_state + self._joint_offsets).tolist())
