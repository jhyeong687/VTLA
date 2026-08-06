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

from __future__ import annotations

########################################################################################
# Utilities
########################################################################################
import logging
import select
import sys
import termios
import threading
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.types import PolicyAction


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        logging.debug("pynput is unavailable; treating this process as headless for GUI-only checks.")
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = _make_control_events()

    try:
        from pynput import keyboard
    except Exception:
        listener = _TerminalKeyboardListener(events)
        if listener.start():
            logging.info("Using terminal keyboard controls: S=start, D=end episode, ESC=stop.")
            return listener, events

        logging.warning(
            "Keyboard listener is unavailable. Install pynput or run from an interactive terminal "
            "to use S/D/ESC controls."
        )
        return None, events

    def on_press(key):
        try:
            key_char = getattr(key, "char", None)
            if isinstance(key_char, str):
                _handle_control_key(key_char, events)
            elif key == keyboard.Key.right:
                _handle_control_key("right", events)
            elif key == keyboard.Key.left:
                _handle_control_key("left", events)
            elif key == keyboard.Key.esc:
                _handle_control_key("esc", events)
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def _make_control_events() -> dict[str, bool]:
    return {
        "start_episode": False,
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }


def _handle_control_key(key: str, events: dict[str, bool]) -> None:
    key = key.lower()
    if key == "s":
        print("S key pressed. Starting episode...")
        events["start_episode"] = True
    elif key == "d" or key == "right":
        print("D/right key pressed. Ending current loop...")
        events["exit_early"] = True
    elif key == "left":
        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif key == "esc":
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True


class _TerminalKeyboardListener:
    """Small stdin fallback for WSL/headless terminals where pynput is unavailable."""

    def __init__(self, events: dict[str, bool]):
        self.events = events
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_termios = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False

        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            import tty

            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            self._restore_terminal()
            return False

        self._thread = threading.Thread(target=self._run, name="terminal-keyboard-listener", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None
        self._restore_terminal()

    def _restore_terminal(self) -> None:
        if self._old_termios is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except Exception:
                logging.debug("Failed to restore terminal settings.", exc_info=True)
            self._old_termios = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue
                char = sys.stdin.read(1)
                key = self._read_escape_sequence() if char == "\x1b" else char
                _handle_control_key(key, self.events)
            except Exception as exc:
                logging.debug("Terminal keyboard listener failed: %s", exc)
                break

    @staticmethod
    def _read_escape_sequence() -> str:
        readable, _, _ = select.select([sys.stdin], [], [], 0.02)
        if not readable:
            return "esc"
        second = sys.stdin.read(1)
        readable, _, _ = select.select([sys.stdin], [], [], 0.02)
        if not readable:
            return "esc"
        third = sys.stdin.read(1)
        if second == "[" and third == "C":
            return "right"
        if second == "[" and third == "D":
            return "left"
        return "esc"


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    require_package("deepdiff", extra="deepdiff-dep")

    from lerobot.utils.constants import DEFAULT_FEATURES

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
