"""Tests for robot tools."""

import pytest


@pytest.mark.asyncio
async def test_look_mock(robot_ctx):
    from robot_agent.tools.look import LookTool

    robot_ctx.env.reset()
    tool = LookTool(robot_ctx)
    result = await tool.execute(question="what objects are on the table?")
    assert "alphabet_soup" in result
    assert "basket" in result


@pytest.mark.asyncio
async def test_move_mock(robot_ctx):
    from robot_agent.tools.move import MoveTool

    robot_ctx.env.reset()
    tool = MoveTool(robot_ctx)
    result = await tool.execute(target="basket", position=[0.6, 0.0, 0.05])
    assert "Moved" in result


@pytest.mark.asyncio
async def test_grasp_close_near_object(robot_ctx):
    from robot_agent.tools.grasp import GraspTool
    from robot_agent.tools.move import MoveTool

    robot_ctx.env.reset()
    move = MoveTool(robot_ctx)
    await move.execute(target="alphabet_soup", position=[0.2, -0.1, 0.10])
    grasp = GraspTool(robot_ctx)
    result = await grasp.execute(action="close")
    assert "Grasped alphabet_soup" in result


@pytest.mark.asyncio
async def test_grasp_open_releases(robot_ctx):
    from robot_agent.tools.grasp import GraspTool
    from robot_agent.tools.move import MoveTool

    robot_ctx.env.reset()
    move = MoveTool(robot_ctx)
    await move.execute(target="alphabet_soup", position=[0.2, -0.1, 0.10])
    grasp = GraspTool(robot_ctx)
    await grasp.execute(action="close")
    result = await grasp.execute(action="open")
    assert "Released" in result


@pytest.mark.asyncio
async def test_perceive(robot_ctx):
    from robot_agent.tools.perceive import PerceiveTool

    robot_ctx.env.reset()
    tool = PerceiveTool(robot_ctx)
    result = await tool.execute(goal="find the tomato sauce")
    assert "tomato_sauce" in result


@pytest.mark.asyncio
async def test_emergency_stop(robot_ctx):
    from robot_agent.tools.safety import EmergencyStopTool

    tool = EmergencyStopTool(robot_ctx)
    result = await tool.execute()
    assert "EMERGENCY STOP" in result
    assert robot_ctx.safety.estop_active


@pytest.mark.asyncio
async def test_model_health(robot_ctx):
    from robot_agent.tools.model_mgmt import ModelHealthTool

    robot_ctx.env.reset()
    tool = ModelHealthTool(robot_ctx)
    result = await tool.execute()
    assert "ok" in result


@pytest.mark.asyncio
async def test_model_ensure_ready(robot_ctx):
    from robot_agent.tools.model_mgmt import ModelEnsureTool

    robot_ctx.env.reset()
    tool = ModelEnsureTool(robot_ctx)
    result = await tool.execute()
    assert "ready" in result.lower()


@pytest.mark.asyncio
async def test_model_ensure_blocked_by_estop(robot_ctx):
    from robot_agent.tools.model_mgmt import ModelEnsureTool

    robot_ctx.env.reset()
    robot_ctx.safety.trigger_estop()
    tool = ModelEnsureTool(robot_ctx)
    result = await tool.execute()
    assert "NOT READY" in result


@pytest.mark.asyncio
async def test_start_subtask(robot_ctx):
    from robot_agent.tools.subtask import StartSubtaskTool

    robot_ctx.env.reset()
    tool = StartSubtaskTool(robot_ctx)
    result = await tool.execute(
        instruction="pick up the cup",
        target={"object": "cup", "position": [0.2, 0.0, 0.1]},
    )
    assert "subtask_id" in result
    assert "started" in result


@pytest.mark.asyncio
async def test_check_loops(robot_ctx):
    from robot_agent.tools.subtask import CheckLoopsTool

    tool = CheckLoopsTool(robot_ctx)
    result = await tool.execute()
    assert "active_count" in result
