# Good to know

This is a document that provides a list of useful tips and troubleshooting for using this LeRobot fork.

- If you have the following error when trying to run a command:
```
Traceback (most recent call last):
  File "/home/zico/dev/lerobot/src/lerobot/scripts/lerobot_record.py", line 92, in <module>
    from lerobot.robots import (  # noqa: F401
  File "/home/zico/dev/lerobot/src/lerobot/robots/abb/__init__.py", line 1, in <module>
    from .abb import ABB
  File "/home/zico/dev/lerobot/src/lerobot/robots/abb/abb.py", line 5, in <module>
    from ABBRobotEGM import EGM
ModuleNotFoundError: No module named 'ABBRobotEGM'
```
Then it is most likely that you need to install an optional dependency, like `abb`. It is better to install all dependencies to be safe, like `uv sync --extra all`.

- Command to replay: `.venv/bin/python src/lerobot/scripts/lerobot_replay.py --robot.type=ur --robot.ip_address=192.168.0.3 --robot.state_feedback_hz=30 --robot.max_gripper_force=100 --robot.gripper_hostname=192.168.0.3 --robot.gripper_port=63352 --dataset.repo_id=lerobot/test_ur5e --dataset.root=/home/zico/.cache/huggingface/lerobot/test_ur5e --dataset.episode=0 --dataset.fps=30`. Note that if the iniital pose is not close to the start pose in the dataset, the robot potentially moves very fast! So make sure to adjust the initial pose to close to the start location for your dataset. This way you can reduce this jump.

- Command to record: `sudo PYNPUT_BACKEND_KEYBOARD=uinput .venv/bin/python src/lerobot/scripts/lerobot_record.py --robot.type=ur --robot.ip_address=192.168.0.3 --robot.state_feedback_hz=30 --robot.max_gripper_force=100 --robot.gripper_hostname=192.168.0.3 --robot.gripper_port=63352 --teleop.type=gello --teleop.port=/dev/ttyUSB0 --teleop.joint_offsets="[0.000, 3.142, 4.712, 7.854, 3.142, 6.283]" --teleop.gripper_config="[7, 195, 153]" --display_data=true --dataset.fps=30 --dataset.repo_id=lerobot/test_ur5e --dataset.root=/home/zico/.cache/huggingface/lerobot/test_ur5e --dataset.push_to_hub=False --dataset.single_task="Do something"`. Note that on Arch Linux with Wayland, you need to run as root and use `uinput` to receive the keyboard events.
