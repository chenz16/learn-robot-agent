"""Robot tools for the agent."""

from nanobot.agent.tools.registry import ToolRegistry
from robot_agent.context import RobotContext
from robot_agent.tools.grasp import GraspTool
from robot_agent.tools.look import LookTool
from robot_agent.tools.model_mgmt import ModelEnsureTool, ModelHealthTool
from robot_agent.tools.move import MoveTool
from robot_agent.tools.perceive import PerceiveTool
from robot_agent.tools.safety import EmergencyStopTool
from robot_agent.tools.subtask import CheckLoopsTool, StartSubtaskTool, WaitSubtaskTool


def register_robot_tools(registry: ToolRegistry, ctx: RobotContext) -> None:
    """Register all 10 robot tools into the agent's tool registry."""
    for tool_cls in [
        LookTool,
        MoveTool,
        GraspTool,
        PerceiveTool,
        StartSubtaskTool,
        CheckLoopsTool,
        WaitSubtaskTool,
        EmergencyStopTool,
        ModelHealthTool,
        ModelEnsureTool,
    ]:
        registry.register(tool_cls(ctx))
