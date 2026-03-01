"""Subtask tools: start_subtask, check_loops, wait_subtask."""

import json
from typing import Any

from nanobot.agent.tools.base import Tool
from robot_agent.context import RobotContext


class StartSubtaskTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "start_subtask"

    @property
    def description(self) -> str:
        return (
            "Launch a VLA control loop for a subtask. The loop runs asynchronously: "
            "observe → VLA predict → safety check → execute → check termination. "
            "Use wait_subtask to wait for completion."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural language instruction for the VLA (e.g., 'pick up the red cup')",
                },
                "target": {
                    "type": "object",
                    "description": "Target info: {object: str, position: [x,y,z]}",
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "hard"],
                    "description": "Task difficulty: 'easy' for simple single-step actions (pick, place, move), 'hard' for multi-step or precise actions (open drawer then place inside). Determines VLA control loop parameters.",
                },
            },
            "required": ["instruction", "target"],
        }

    async def execute(self, **kwargs: Any) -> str:
        instruction = kwargs["instruction"]
        target = kwargs.get("target", {})
        difficulty = kwargs.get("difficulty", "easy")

        route = self._ctx.route_config.get_route(difficulty)

        subtask_id = await self._ctx.loop_manager.start_subtask(
            instruction=instruction,
            target=target,
            route_level=route,
            env=self._ctx.env,
            vla=self._ctx.vla,
            safety=self._ctx.safety,
        )
        return json.dumps({
            "subtask_id": subtask_id,
            "status": "started",
            "difficulty": difficulty,
            "action_hz": route.action_hz,
            "max_steps": route.max_steps,
        })


class CheckLoopsTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "check_loops"

    @property
    def description(self) -> str:
        return "Query the status of all running and completed VLA control loops."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        status = self._ctx.loop_manager.get_status()
        return json.dumps(status, indent=2, default=str)


class WaitSubtaskTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "wait_subtask"

    @property
    def description(self) -> str:
        return "Wait for a running subtask to complete. Returns the final status and result."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subtask_id": {
                    "type": "string",
                    "description": "The subtask ID returned by start_subtask",
                },
                "timeout": {
                    "type": "number",
                    "description": "Maximum seconds to wait (default: 60)",
                },
            },
            "required": ["subtask_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        subtask_id = kwargs["subtask_id"]
        timeout = kwargs.get("timeout", 60.0)
        result = await self._ctx.loop_manager.wait_for_completion(subtask_id, timeout)
        return json.dumps({
            "subtask_id": result.subtask_id,
            "status": result.status,
            "steps_executed": result.steps_executed,
            "reason": result.reason,
            "total_reward": result.total_reward,
        })
