"""Tests for LoopManager."""

import pytest

from robot_agent.routing import RouteLevel


@pytest.mark.asyncio
async def test_start_and_wait(loop_manager, mock_env, mock_vla, safety_mgr):
    mock_env.reset()
    route = RouteLevel(action_hz=10, max_steps=10, position_threshold=0.05)
    subtask_id = await loop_manager.start_subtask(
        instruction="pick up the cup",
        target={"object": "cup", "position": [0.2, 0.0, 0.1]},
        route_level=route,
        env=mock_env,
        vla=mock_vla,
        safety=safety_mgr,
    )
    assert subtask_id.startswith("sub_")
    result = await loop_manager.wait_for_completion(subtask_id, timeout=10.0)
    assert result.status in ("completed", "failed", "stopped")
    assert result.steps_executed > 0


@pytest.mark.asyncio
async def test_step_limit_terminates(loop_manager, mock_env, mock_vla, safety_mgr):
    mock_env.reset()
    route = RouteLevel(action_hz=100, max_steps=5, position_threshold=0.001)
    subtask_id = await loop_manager.start_subtask(
        instruction="test",
        target={},
        route_level=route,
        env=mock_env,
        vla=mock_vla,
        safety=safety_mgr,
    )
    result = await loop_manager.wait_for_completion(subtask_id, timeout=10.0)
    assert "Step limit" in result.reason


@pytest.mark.asyncio
async def test_stop_all(loop_manager, mock_env, mock_vla, safety_mgr):
    mock_env.reset()
    route = RouteLevel(action_hz=10, max_steps=10000)
    await loop_manager.start_subtask(
        instruction="long task",
        target={},
        route_level=route,
        env=mock_env,
        vla=mock_vla,
        safety=safety_mgr,
    )
    cancelled = loop_manager.stop_all()
    assert cancelled >= 1


@pytest.mark.asyncio
async def test_get_status(loop_manager, mock_env, mock_vla, safety_mgr):
    mock_env.reset()
    route = RouteLevel(action_hz=100, max_steps=5)
    subtask_id = await loop_manager.start_subtask(
        instruction="test",
        target={},
        route_level=route,
        env=mock_env,
        vla=mock_vla,
        safety=safety_mgr,
    )
    await loop_manager.wait_for_completion(subtask_id, timeout=10.0)
    status = loop_manager.get_status()
    assert status["completed_count"] >= 1
