"""Mock VLA adapter for development and testing."""

import time
from typing import Any

import numpy as np

from robot_agent.vla.base import VLAAdapter


class MockVLAAdapter(VLAAdapter):
    """Returns small random actions. No GPU required."""

    def __init__(self, action_dim: int = 7, action_horizon: int = 4):
        self._action_dim = action_dim
        self._action_horizon = action_horizon

    async def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        actions = np.random.uniform(-0.01, 0.01, size=(self._action_horizon, self._action_dim))
        # Set gripper based on instruction keywords
        instruction_lower = instruction.lower()
        if "close" in instruction_lower or "grasp" in instruction_lower or "pick" in instruction_lower:
            actions[:, -1] = -1.0
        elif "open" in instruction_lower or "release" in instruction_lower or "place" in instruction_lower:
            actions[:, -1] = 1.0
        else:
            actions[:, -1] = 0.0
        return actions

    def reset(self) -> None:
        pass

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "latency_ms": 0.1, "backend": "mock"}

    def get_action_horizon(self) -> int:
        return self._action_horizon
