"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path so robot_agent is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_env():
    from robot_agent.env.mock import MockEnv
    return MockEnv()


@pytest.fixture
def mock_vla():
    from robot_agent.vla.mock import MockVLAAdapter
    return MockVLAAdapter(action_dim=7)


@pytest.fixture
def safety_mgr():
    from robot_agent.safety import SafetyManager
    return SafetyManager(max_velocity=0.5)


@pytest.fixture
def route_config():
    from robot_agent.routing import RouteConfig
    return RouteConfig.default()


@pytest.fixture
def loop_manager():
    from robot_agent.loop import LoopManager
    return LoopManager()


@pytest.fixture
def robot_ctx(mock_env, mock_vla, safety_mgr, route_config, loop_manager):
    from robot_agent.context import RobotContext
    return RobotContext(
        env=mock_env,
        vla=mock_vla,
        loop_manager=loop_manager,
        safety=safety_mgr,
        route_config=route_config,
        vlm_url=None,
    )
