"""Safety manager: E-Stop and velocity limiting."""

from loguru import logger
import numpy as np


class SafetyManager:
    """Two safety layers: E-Stop and velocity clamp."""

    def __init__(self, max_velocity: float = 0.5):
        self.max_velocity = max_velocity
        self.estop_active = False
        self._events: list[dict] = []

    def trigger_estop(self) -> None:
        """Activate emergency stop. All subsequent actions become zero."""
        self.estop_active = True
        self._log_event("E-STOP triggered")

    def reset_estop(self) -> None:
        """Manually reset emergency stop."""
        self.estop_active = False
        self._log_event("E-STOP reset")

    def check_action(self, action: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Check and clamp action. Returns (safe_action, warnings)."""
        warnings = []
        action = np.array(action, dtype=np.float64)

        if self.estop_active:
            warnings.append("E-STOP ACTIVE: all motion zeroed")
            self._log_event("Action blocked by E-STOP")
            return np.zeros_like(action), warnings

        # Clamp velocity (first 3 dims = position delta)
        velocity = action[:3]
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            action[:3] = velocity * scale
            warnings.append(f"Velocity clamped: {speed:.3f} -> {self.max_velocity:.3f}")
            self._log_event(f"Velocity clamped from {speed:.3f}")

        return action, warnings

    def get_events(self) -> list[dict]:
        """Return logged safety events."""
        return list(self._events)

    def _log_event(self, message: str) -> None:
        import time
        event = {"time": time.time(), "message": message}
        self._events.append(event)
        logger.warning("Safety: {}", message)
