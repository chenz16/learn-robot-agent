"""Termination strategies for VLA control loops."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Terminator(ABC):
    """Abstract termination strategy."""

    @abstractmethod
    def should_terminate(self, step_count: int, observation: dict[str, Any], **kwargs) -> tuple[bool, str]:
        """Check if loop should stop. Returns (should_stop, reason)."""


class StepLimitTerminator(Terminator):
    """Terminate after max_steps."""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    def should_terminate(self, step_count: int, observation: dict[str, Any], **kwargs) -> tuple[bool, str]:
        if step_count >= self.max_steps:
            return True, f"Step limit reached ({self.max_steps})"
        return False, ""


class PositionThresholdTerminator(Terminator):
    """Terminate when end-effector is close enough to target."""

    def __init__(self, target_position: list[float], threshold: float):
        self.target = np.array(target_position)
        self.threshold = threshold

    def should_terminate(self, step_count: int, observation: dict[str, Any], **kwargs) -> tuple[bool, str]:
        ee_pos = observation.get("robot0_eef_pos")
        if ee_pos is None:
            return False, ""
        dist = np.linalg.norm(np.array(ee_pos) - self.target)
        if dist < self.threshold:
            return True, f"Position threshold reached (dist={dist:.4f} < {self.threshold})"
        return False, ""


class CompositeTerminator(Terminator):
    """Combine multiple terminators with any/all logic."""

    def __init__(self, terminators: list[Terminator], mode: str = "any"):
        self.terminators = terminators
        self.mode = mode  # "any" = OR, "all" = AND

    def should_terminate(self, step_count: int, observation: dict[str, Any], **kwargs) -> tuple[bool, str]:
        results = [t.should_terminate(step_count, observation, **kwargs) for t in self.terminators]
        if self.mode == "any":
            for should_stop, reason in results:
                if should_stop:
                    return True, reason
            return False, ""
        else:  # "all"
            reasons = [r for s, r in results if s]
            if len(reasons) == len(self.terminators):
                return True, "; ".join(reasons)
            return False, ""
