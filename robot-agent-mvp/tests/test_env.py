"""Tests for MockEnv."""

import numpy as np


def test_reset_returns_observation(mock_env):
    obs = mock_env.reset()
    assert "agentview_image" in obs
    assert "robot0_eef_pos" in obs
    assert "robot0_eef_quat" in obs
    assert "robot0_gripper_qpos" in obs
    assert "objects" in obs


def test_image_shape(mock_env):
    mock_env.reset()
    img = mock_env.render()
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8


def test_action_dim(mock_env):
    assert mock_env.action_dim == 7


def test_step_updates_position(mock_env):
    mock_env.reset()
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    obs, reward, done, info = mock_env.step(action)
    assert obs["robot0_eef_pos"][0] == pytest.approx(0.1, abs=1e-6)


def test_grasp_near_object(mock_env):
    mock_env.reset()
    # Move to alphabet_soup position
    mock_env._ee_pos = np.array([0.2, -0.1, 0.10])
    # Close gripper
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    obs, _, _, _ = mock_env.step(action)
    assert obs["holding"] == "alphabet_soup"


def test_release_object(mock_env):
    mock_env.reset()
    mock_env._ee_pos = np.array([0.2, -0.1, 0.10])
    mock_env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
    assert mock_env._holding == "alphabet_soup"
    # Open gripper
    mock_env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert mock_env._holding is None


import pytest
