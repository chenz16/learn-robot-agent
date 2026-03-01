"""LIBERO environment wrapper."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from robot_agent.env.base import RobotEnv

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    _LIBERO_AVAILABLE = True
except ImportError:
    _LIBERO_AVAILABLE = False

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


class LiberoEnv(RobotEnv):
    """Wraps LIBERO's OffScreenRenderEnv for the robot agent.

    Observation keys passed through to VLA adapter:
        agentview_image: uint8 (H, W, 3) — rotated 180°
        robot0_eye_in_hand_image: uint8 (H, W, 3) — rotated 180°
        robot0_eef_pos: float64 (3,)
        robot0_eef_quat: float64 (4,)
        robot0_gripper_qpos: float64 (2,)
    """

    def __init__(
        self,
        task_name: str = "libero_10:0",
        camera_height: int = 256,
        camera_width: int = 256,
        seed: int = 7,
        num_steps_wait: int = 10,
    ):
        if not _LIBERO_AVAILABLE:
            raise ImportError(
                "LIBERO is not installed. Install with: pip install robosuite libero"
            )

        # Parse task_name: "suite_name:task_id"
        parts = task_name.split(":")
        suite_name = parts[0] if len(parts) > 0 else "libero_10"
        task_id = int(parts[1]) if len(parts) > 1 else 0

        self._suite_name = suite_name
        self._task_id = task_id
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._seed = seed
        self._num_steps_wait = num_steps_wait

        # Load task from benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[suite_name]()
        self._task = self._task_suite.get_task(task_id)
        self._init_states = self._task_suite.get_task_init_states(task_id)

        # Build BDDL file path
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file

        # Create environment
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=camera_height,
            camera_widths=camera_width,
        )
        self._env.seed(seed)
        self._obs: dict[str, Any] | None = None
        self._episode_idx = 0

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def task_description(self) -> str:
        return self._task.language

    @property
    def num_episodes(self) -> int:
        return len(self._init_states)

    def reset(self, episode_idx: int = 0) -> dict[str, Any]:
        self._episode_idx = episode_idx
        self._env.reset()
        raw_obs = self._env.set_init_state(self._init_states[episode_idx])
        # Wait for objects to settle
        for _ in range(self._num_steps_wait):
            raw_obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)
        self._obs = self._process_obs(raw_obs)
        return self._obs

    def step(self, action: np.ndarray | list) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = action.tolist()
        raw_obs, reward, done, info = self._env.step(action)
        self._obs = self._process_obs(raw_obs)
        return self._obs, reward, done, info

    def get_observation(self) -> dict[str, Any]:
        if self._obs is None:
            self._obs = self._process_obs(self._env._get_observations())
        return self._obs

    def render(self) -> np.ndarray:
        obs = self.get_observation()
        return obs.get("agentview_image", np.zeros((self._camera_height, self._camera_width, 3), dtype=np.uint8))

    def close(self) -> None:
        self._env.close()

    def check_success(self) -> bool:
        return self._env.check_success()

    def _process_obs(self, raw_obs: dict) -> dict[str, Any]:
        """Process raw LIBERO observation: rotate images 180 degrees."""
        obs = dict(raw_obs)
        # Rotate images 180 degrees (critical for VLA model compatibility)
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            if key in obs:
                obs[key] = np.ascontiguousarray(obs[key][::-1, ::-1])
        return obs
