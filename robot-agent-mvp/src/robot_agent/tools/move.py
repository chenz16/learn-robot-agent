"""MoveTool: move end-effector to target position."""

from typing import Any

import numpy as np

from nanobot.agent.tools.base import Tool
from robot_agent.context import RobotContext
from robot_agent.env.mock import MockEnv


class MoveTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "move"

    @property
    def description(self) -> str:
        return "Move the robot end-effector to a target position."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name of the target object or location",
                },
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Target [x, y, z] position",
                },
            },
            "required": ["target", "position"],
        }

    async def execute(self, **kwargs: Any) -> str:
        target = kwargs["target"]
        position = kwargs["position"]

        if isinstance(self._ctx.env, MockEnv):
            return self._mock_move(target, position)

        # LIBERO mode: delegate to LoopManager
        route = self._ctx.route_config.get_route("easy")
        subtask_id = await self._ctx.loop_manager.start_subtask(
            instruction=f"move to {target}",
            target={"object": target, "position": position},
            route_level=route,
            env=self._ctx.env,
            vla=self._ctx.vla,
            safety=self._ctx.safety,
        )
        result = await self._ctx.loop_manager.wait_for_completion(subtask_id, timeout=30.0)
        return f"Move to {target}: {result.status} ({result.steps_executed} steps, {result.reason})"

    def _mock_move(self, target: str, position: list[float]) -> str:
        env = self._ctx.env
        assert isinstance(env, MockEnv)
        env._ee_pos = np.array(position, dtype=np.float64)
        if env._holding is not None:
            env._objects[env._holding] = position
        return f"Moved to {target} at position {position}. Gripper: {'open' if env._gripper_open else 'closed'}."
