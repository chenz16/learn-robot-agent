"""Abstract base class for VLA adapters."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VLAAdapter(ABC):
    """Vision-Language-Action model adapter interface."""

    @abstractmethod
    async def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Predict actions from observation and instruction.

        Returns array of shape (action_horizon, action_dim).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new episode."""

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check service health. Returns {"status": "ok"|"error", "latency_ms": float}."""

    @abstractmethod
    def get_action_horizon(self) -> int:
        """Number of action steps per prediction."""
