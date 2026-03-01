"""Abstract base class for robot environments."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RobotEnv(ABC):
    """Abstract robot environment interface (Gymnasium-style)."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action dimensionality (7 for LIBERO: xyz + rxryrz + gripper)."""

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset environment and return initial observation."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute action. Returns (observation, reward, done, info)."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Get current observation without stepping."""

    @abstractmethod
    def render(self) -> np.ndarray:
        """Get camera image as numpy array (H, W, 3)."""

    def close(self) -> None:
        """Clean up resources."""
