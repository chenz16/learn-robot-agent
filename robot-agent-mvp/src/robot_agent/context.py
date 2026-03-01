"""Shared state for all robot tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_agent.env.base import RobotEnv
    from robot_agent.loop import LoopManager
    from robot_agent.routing import RouteConfig
    from robot_agent.safety import SafetyManager
    from robot_agent.vla.base import VLAAdapter


@dataclass
class RobotContext:
    """Injected into every robot tool. Holds all shared state."""
    env: RobotEnv
    vla: VLAAdapter
    loop_manager: LoopManager
    safety: SafetyManager
    route_config: RouteConfig
    vlm_url: str | None = None
