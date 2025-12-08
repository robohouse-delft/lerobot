#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    BiKeyboardEndEffectorTeleopConfig,
    KeyboardEndEffectorTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


def str_to_key(key_str):
    """
    Convert a string representation of a key to a pynput.keyboard.Key
    enumeration if it exists; otherwise, return the character as-is.
    """
    # Try to get from special keys (e.g. 'up', 'down', 'ctrl', etc.)
    try:
        return getattr(keyboard.Key, key_str)
    except AttributeError:
        # Not a special key, assume it’s a single character key
        if len(key_str) == 1:
            return keyboard.KeyCode.from_char(key_str)
        else:
            raise ValueError(f"Unknown key: {key_str}") from None


def is_reserved_key(key_pair: tuple[str, str]):
    return "s" in key_pair or "q" in key_pair or "r" in key_pair


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        self.event_queue.put((key, True))

    def _on_release(self, key):
        self.event_queue.put((key, False))

        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        if (
            is_reserved_key(self.config.delta_x_keys)
            or is_reserved_key(self.config.delta_y_keys)
            or is_reserved_key(self.config.delta_z_keys)
            or is_reserved_key(self.config.gripper_keys)
        ):
            raise ValueError("Cannot listen for control action on reserved keys: ('s', 'q', 'r')")

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == str_to_key(self.config.delta_y_keys[0]):
                delta_y = -int(val)
            elif key == str_to_key(self.config.delta_y_keys[1]):
                delta_y = int(val)
            elif key == str_to_key(self.config.delta_x_keys[0]):
                delta_x = int(val)
            elif key == str_to_key(self.config.delta_x_keys[1]):
                delta_x = -int(val)
            elif key == str_to_key(self.config.delta_z_keys[0]):
                delta_z = -int(val)
            elif key == str_to_key(self.config.delta_z_keys[1]):
                delta_z = int(val)
            elif key == str_to_key(self.config.gripper_keys[1]):
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == str_to_key(self.config.gripper_keys[0]):
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            str_to_key(self.config.delta_y_keys[0]),
            str_to_key(self.config.delta_y_keys[1]),
            str_to_key(self.config.delta_x_keys[0]),
            str_to_key(self.config.delta_x_keys[1]),
            str_to_key(self.config.delta_z_keys[0]),
            str_to_key(self.config.delta_z_keys[1]),
            str_to_key(self.config.gripper_keys[0]),
            str_to_key(self.config.gripper_keys[0]),
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == str_to_key("s"):
                success = True
            elif key == str_to_key("r"):
                terminate_episode = True
                rerecord_episode = True
            elif key == str_to_key("q"):
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class BiKeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control of a dual arm manipulator.
    """

    config_class = BiKeyboardEndEffectorTeleopConfig
    name = "bi_keyboard_ee"

    def __init__(self, config: BiKeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        if (
            is_reserved_key(self.config.left_delta_x_keys)
            or is_reserved_key(self.config.left_delta_y_keys)
            or is_reserved_key(self.config.left_delta_z_keys)
            or is_reserved_key(self.config.left_gripper_keys)
        ):
            raise ValueError("Cannot listen for control action on reserved keys: ('s', 'q', 'r')")
        if (
            is_reserved_key(self.config.right_delta_x_keys)
            or is_reserved_key(self.config.right_delta_y_keys)
            or is_reserved_key(self.config.right_delta_z_keys)
            or is_reserved_key(self.config.right_gripper_keys)
        ):
            raise ValueError("Cannot listen for control action on reserved keys: ('s', 'q', 'r')")

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (8,),
                "names": {
                    "left_delta_x": 0,
                    "left_delta_y": 1,
                    "left_delta_z": 2,
                    "left_gripper": 3,
                    "right_delta_x": 4,
                    "right_delta_y": 5,
                    "right_delta_z": 6,
                    "right_gripper": 7,
                },
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {
                    "left_delta_x": 0,
                    "left_delta_y": 1,
                    "left_delta_z": 2,
                    "right_delta_x": 3,
                    "right_delta_y": 4,
                    "right_delta_z": 5,
                },
            }

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        left_delta_x = 0.0
        left_delta_y = 0.0
        left_delta_z = 0.0
        left_gripper_action = 1.0
        right_delta_x = 0.0
        right_delta_y = 0.0
        right_delta_z = 0.0
        right_gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == str_to_key(self.config.left_delta_y_keys[0]):
                left_delta_y = -int(val)
            elif key == str_to_key(self.config.left_delta_y_keys[1]):
                left_delta_y = int(val)
            if key == str_to_key(self.config.right_delta_y_keys[0]):
                right_delta_y = -int(val)
            elif key == str_to_key(self.config.right_delta_y_keys[1]):
                right_delta_y = int(val)
            elif key == str_to_key(self.config.left_delta_x_keys[0]):
                left_delta_x = int(val)
            elif key == str_to_key(self.config.left_delta_x_keys[1]):
                left_delta_x = -int(val)
            elif key == str_to_key(self.config.right_delta_x_keys[0]):
                right_delta_x = int(val)
            elif key == str_to_key(self.config.right_delta_x_keys[1]):
                right_delta_x = -int(val)
            elif key == str_to_key(self.config.left_delta_z_keys[0]):
                left_delta_z = -int(val)
            elif key == str_to_key(self.config.left_delta_z_keys[1]):
                left_delta_z = int(val)
            elif key == str_to_key(self.config.right_delta_z_keys[0]):
                right_delta_z = -int(val)
            elif key == str_to_key(self.config.right_delta_z_keys[1]):
                right_delta_z = int(val)
            elif key == str_to_key(self.config.left_gripper_keys[1]):
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                left_gripper_action = int(val) + 1
            elif key == str_to_key(self.config.left_gripper_keys[0]):
                left_gripper_action = int(val) - 1
            elif key == str_to_key(self.config.right_gripper_keys[1]):
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                right_gripper_action = int(val) + 1
            elif key == str_to_key(self.config.right_gripper_keys[0]):
                right_gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "left_delta_x": left_delta_x,
            "left_delta_y": left_delta_y,
            "left_delta_z": left_delta_z,
            "right_delta_x": right_delta_x,
            "right_delta_y": right_delta_y,
            "right_delta_z": right_delta_z,
        }

        if self.config.use_gripper:
            action_dict["left_gripper"] = left_gripper_action
            action_dict["right_gripper"] = right_gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            str_to_key(self.config.left_delta_y_keys[0]),
            str_to_key(self.config.left_delta_y_keys[1]),
            str_to_key(self.config.right_delta_y_keys[0]),
            str_to_key(self.config.right_delta_y_keys[1]),
            str_to_key(self.config.left_delta_x_keys[0]),
            str_to_key(self.config.left_delta_x_keys[1]),
            str_to_key(self.config.right_delta_x_keys[0]),
            str_to_key(self.config.right_delta_x_keys[1]),
            str_to_key(self.config.left_delta_z_keys[0]),
            str_to_key(self.config.left_delta_z_keys[1]),
            str_to_key(self.config.right_delta_z_keys[0]),
            str_to_key(self.config.right_delta_z_keys[1]),
            str_to_key(self.config.left_gripper_keys[0]),
            str_to_key(self.config.left_gripper_keys[0]),
            str_to_key(self.config.right_gripper_keys[0]),
            str_to_key(self.config.right_gripper_keys[0]),
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == str_to_key("s"):
                success = True
            elif key == str_to_key("r"):
                terminate_episode = True
                rerecord_episode = True
            elif key == str_to_key("q"):
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear.vel": float,
            "angular.vel": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    def get_action(self) -> dict[str, Any]:
        """
        Get the current action based on pressed keys.

        Returns:
            dict with 'linear.vel' and 'angular.vel' keys
        """
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardRoverTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear.vel": linear_velocity,
            "angular.vel": angular_velocity,
        }
