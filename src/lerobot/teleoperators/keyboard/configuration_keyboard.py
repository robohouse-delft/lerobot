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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""

    # TODO(Steven): Consider setting in here the keys that we want to capture/listen


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True
    delta_x_keys: tuple[str, str] = ("up", "down")
    delta_y_keys: tuple[str, str] = ("left", "right")
    delta_z_keys: tuple[str, str] = ("shift", "shift_r")
    gripper_keys: tuple[str, str] = ("ctrl_l", "ctrl_r")

@TeleoperatorConfig.register_subclass("bi_keyboard_ee")
@dataclass
class BiKeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True
    # Left arm controls
    left_delta_x_keys: tuple[str, str] = ("up", "down")
    left_delta_y_keys: tuple[str, str] = ("left", "right")
    left_delta_z_keys: tuple[str, str] = ("shift", "shift_r")
    left_gripper_keys: tuple[str, str] = ("ctrl_l", "ctrl_r")
    # Right arm controls
    right_delta_x_keys: tuple[str, str] = ("y", "h")
    right_delta_y_keys: tuple[str, str] = ("g", "j")
    right_delta_z_keys: tuple[str, str] = ("t", "u")
    right_gripper_keys: tuple[str, str] = ("z", "x")