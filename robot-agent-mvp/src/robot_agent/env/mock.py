"""Mock robot environment for development and testing."""

from typing import Any

import numpy as np

from robot_agent.env.base import RobotEnv


class MockEnv(RobotEnv):
    """Stateful mock environment. Tracks EE position, gripper, and objects."""

    GRASP_DISTANCE = 0.05  # meters

    def __init__(self, objects: dict[str, list[float]] | None = None):
        self._objects = objects or {
            "alphabet_soup": [0.2, -0.1, 0.10],
            "tomato_sauce": [-0.1, 0.15, 0.10],
            "basket": [0.6, 0.0, 0.05],
        }
        self._ee_pos = np.array([0.0, 0.0, 0.3])
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._gripper_open = True
        self._holding: str | None = None
        self._step_count = 0

    @property
    def action_dim(self) -> int:
        return 7

    def reset(self) -> dict[str, Any]:
        self._ee_pos = np.array([0.0, 0.0, 0.3])
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._gripper_open = True
        self._holding = None
        self._step_count = 0
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        # Delta position (first 3 dims)
        self._ee_pos = self._ee_pos + action[:3]
        # Gripper (last dim): < 0 = close, >= 0 = open
        gripper_cmd = action[6]
        was_open = self._gripper_open
        self._gripper_open = gripper_cmd >= 0

        # Grasp logic: closing gripper near an object picks it up
        if was_open and not self._gripper_open and self._holding is None:
            for name, pos in self._objects.items():
                dist = np.linalg.norm(self._ee_pos - np.array(pos))
                if dist < self.GRASP_DISTANCE:
                    self._holding = name
                    break

        # Release logic: opening gripper drops the held object
        if not was_open and self._gripper_open and self._holding is not None:
            self._objects[self._holding] = self._ee_pos.tolist()
            self._holding = None

        # Move held object with EE
        if self._holding is not None:
            self._objects[self._holding] = self._ee_pos.tolist()

        self._step_count += 1
        obs = self.get_observation()
        return obs, 0.0, False, {"step": self._step_count}

    def get_observation(self) -> dict[str, Any]:
        return {
            "agentview_image": self.render(),
            "robot0_eef_pos": self._ee_pos.copy(),
            "robot0_eef_quat": self._ee_quat.copy(),
            "robot0_gripper_qpos": np.array([1.0 if self._gripper_open else 0.0]),
            "objects": {k: list(v) for k, v in self._objects.items()},
            "holding": self._holding,
        }

    def render(self) -> np.ndarray:
        return np.zeros((256, 256, 3), dtype=np.uint8)
