"""Tests for MockVLAAdapter."""

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_predict_shape(mock_vla):
    obs = {"agentview_image": np.zeros((256, 256, 3))}
    actions = await mock_vla.predict(obs, "pick up the cup")
    assert actions.shape == (4, 7)


@pytest.mark.asyncio
async def test_predict_gripper_close(mock_vla):
    obs = {"agentview_image": np.zeros((256, 256, 3))}
    actions = await mock_vla.predict(obs, "close the gripper")
    assert all(actions[:, -1] == -1.0)


@pytest.mark.asyncio
async def test_predict_gripper_open(mock_vla):
    obs = {"agentview_image": np.zeros((256, 256, 3))}
    actions = await mock_vla.predict(obs, "open the gripper and release")
    assert all(actions[:, -1] == 1.0)


@pytest.mark.asyncio
async def test_health_check(mock_vla):
    result = await mock_vla.health_check()
    assert result["status"] == "ok"


def test_action_horizon(mock_vla):
    assert mock_vla.get_action_horizon() == 4
