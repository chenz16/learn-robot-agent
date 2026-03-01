"""End-to-end test: full PRAE loop with mock env + mock VLA."""

import pytest
import numpy as np

from robot_agent.context import RobotContext
from robot_agent.env.mock import MockEnv
from robot_agent.loop import LoopManager
from robot_agent.routing import RouteConfig
from robot_agent.safety import SafetyManager
from robot_agent.vla.mock import MockVLAAdapter
from robot_agent.tools.look import LookTool
from robot_agent.tools.move import MoveTool
from robot_agent.tools.grasp import GraspTool
from robot_agent.tools.model_mgmt import ModelEnsureTool


@pytest.fixture
def full_ctx():
    env = MockEnv(objects={
        "alphabet_soup": [0.2, -0.1, 0.10],
        "tomato_sauce": [-0.1, 0.15, 0.10],
        "basket": [0.6, 0.0, 0.05],
    })
    vla = MockVLAAdapter(action_dim=7)
    safety = SafetyManager()
    loop_manager = LoopManager()
    route_config = RouteConfig.default()
    return RobotContext(
        env=env, vla=vla, loop_manager=loop_manager,
        safety=safety, route_config=route_config, vlm_url=None,
    )


@pytest.mark.asyncio
async def test_prae_pick_and_place(full_ctx):
    """Simulate a simplified PRAE loop: perceive → move → grasp → move → release."""
    ctx = full_ctx
    ctx.env.reset()

    # PREPARE: ensure services ready
    ensure = ModelEnsureTool(ctx)
    result = await ensure.execute()
    assert "ready" in result.lower()

    # PERCEIVE: look at scene
    look = LookTool(ctx)
    scene = await look.execute(question="describe all objects")
    assert "alphabet_soup" in scene

    # ACT: pick up alphabet_soup
    move = MoveTool(ctx)
    grasp = GraspTool(ctx)

    await move.execute(target="alphabet_soup", position=[0.2, -0.1, 0.10])
    result = await grasp.execute(action="close")
    assert "Grasped" in result

    # ACT: place in basket
    await move.execute(target="basket", position=[0.6, 0.0, 0.05])
    result = await grasp.execute(action="open")
    assert "Released" in result

    # EVALUATE: verify placement
    scene = await look.execute(question="is alphabet_soup in basket?")
    obs = ctx.env.get_observation()
    soup_pos = obs["objects"]["alphabet_soup"]
    basket_pos = obs["objects"]["basket"]
    dist = np.linalg.norm(np.array(soup_pos) - np.array(basket_pos))
    assert dist < 0.01, f"Soup not in basket: dist={dist}"


@pytest.mark.asyncio
async def test_estop_during_operation(full_ctx):
    """Verify E-Stop interrupts operations."""
    ctx = full_ctx
    ctx.env.reset()

    # Trigger E-Stop
    from robot_agent.tools.safety import EmergencyStopTool
    estop = EmergencyStopTool(ctx)
    await estop.execute()

    # Verify safety blocks actions
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    safe, warnings = ctx.safety.check_action(action)
    assert np.all(safe == 0)

    # model_ensure should report not ready
    ensure = ModelEnsureTool(ctx)
    result = await ensure.execute()
    assert "NOT READY" in result
