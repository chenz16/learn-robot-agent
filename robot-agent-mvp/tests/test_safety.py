"""Tests for SafetyManager."""

import numpy as np


def test_estop_zeros_action(safety_mgr):
    safety_mgr.trigger_estop()
    action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0])
    safe, warnings = safety_mgr.check_action(action)
    assert np.all(safe == 0)
    assert any("E-STOP" in w for w in warnings)


def test_estop_reset(safety_mgr):
    safety_mgr.trigger_estop()
    assert safety_mgr.estop_active
    safety_mgr.reset_estop()
    assert not safety_mgr.estop_active


def test_velocity_clamp(safety_mgr):
    action = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    safe, warnings = safety_mgr.check_action(action)
    speed = np.linalg.norm(safe[:3])
    assert speed <= safety_mgr.max_velocity + 1e-6
    assert any("clamped" in w.lower() for w in warnings)


def test_normal_action_passes(safety_mgr):
    action = np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0, -1.0])
    safe, warnings = safety_mgr.check_action(action)
    np.testing.assert_array_almost_equal(safe, action)
    assert len(warnings) == 0
