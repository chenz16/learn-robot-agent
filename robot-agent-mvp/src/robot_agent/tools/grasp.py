"""GraspTool: control gripper open/close."""

from typing import Any

import numpy as np

from nanobot.agent.tools.base import Tool
from robot_agent.context import RobotContext
from robot_agent.env.mock import MockEnv


class GraspTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "grasp"

    @property
    def description(self) -> str:
        return "Control the robot gripper. Use 'close' to grasp an object, 'open' to release."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open", "close"],
                    "description": "Gripper action",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]

        if isinstance(self._ctx.env, MockEnv):
            return self._mock_grasp(action)

        # LIBERO mode: delegate to LoopManager
        route = self._ctx.route_config.get_route("easy")
        subtask_id = await self._ctx.loop_manager.start_subtask(
            instruction=f"{action} the gripper",
            target={"gripper_action": action},
            route_level=route,
            env=self._ctx.env,
            vla=self._ctx.vla,
            safety=self._ctx.safety,
        )
        result = await self._ctx.loop_manager.wait_for_completion(subtask_id, timeout=10.0)
        return f"Gripper {action}: {result.status}"

    def _mock_grasp(self, action: str) -> str:
        env = self._ctx.env
        assert isinstance(env, MockEnv)
        if action == "close":
            env._gripper_open = False
            # Check if near any object
            for name, pos in env._objects.items():
                dist = np.linalg.norm(env._ee_pos - np.array(pos))
                if dist < MockEnv.GRASP_DISTANCE and env._holding is None:
                    env._holding = name
                    return f"Gripper closed. Grasped {name}."
            return "Gripper closed. No object in range."
        else:
            env._gripper_open = True
            released = env._holding
            if released:
                env._objects[released] = env._ee_pos.tolist()
                env._holding = None
                return f"Gripper opened. Released {released} at position {env._ee_pos.tolist()}."
            return "Gripper opened."
