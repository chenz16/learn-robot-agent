"""EmergencyStopTool: immediately halt all robot motion."""

from typing import Any

from nanobot.agent.tools.base import Tool
from robot_agent.context import RobotContext


class EmergencyStopTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "emergency_stop"

    @property
    def description(self) -> str:
        return "EMERGENCY STOP: immediately halt all robot motion. Use when safety is at risk. Requires manual reset."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        self._ctx.safety.trigger_estop()
        cancelled = self._ctx.loop_manager.stop_all()
        return f"EMERGENCY STOP activated. {cancelled} loop(s) cancelled. Manual reset required."
