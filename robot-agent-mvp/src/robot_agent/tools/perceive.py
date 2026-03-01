"""PerceiveTool: deep perception analysis via multi-step observation."""

from typing import Any

from nanobot.agent.tools.base import Tool
from robot_agent.context import RobotContext
from robot_agent.tools.look import LookTool


class PerceiveTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx
        self._look = LookTool(ctx)

    @property
    def name(self) -> str:
        return "perceive"

    @property
    def description(self) -> str:
        return "Deep perception analysis. Captures scene and provides detailed observation relevant to a specific goal."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The perception goal - what to analyze in detail",
                },
            },
            "required": ["goal"],
        }

    async def execute(self, **kwargs: Any) -> str:
        goal = kwargs["goal"]
        # Use look tool for scene capture
        scene = await self._look.execute(question=f"Describe the scene in detail, focusing on: {goal}")
        return f"[Perception analysis for goal: {goal}]\n{scene}"
