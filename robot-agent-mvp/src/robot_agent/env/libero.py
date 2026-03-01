"""LIBERO environment wrapper."""

from typing import Any

import numpy as np

from robot_agent.env.base import RobotEnv

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    _LIBERO_AVAILABLE = True
except ImportError:
    _LIBERO_AVAILABLE = False


class LiberoEnv(RobotEnv):
    """Wraps LIBERO's OffScreenRenderEnv for the robot agent."""

    def __init__(
        self,
        task_name: str = "libero_10:0",
        camera_height: int = 256,
        camera_width: int = 256,
        seed: int = 42,
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

        # Load task from benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[suite_name]()
        self._task = self._task_suite.get_task(task_id)
        self._init_states = self._task_suite.get_task_init_states(task_id)

        # Build BDDL file path
        bddl_file = f"{get_libero_path('bddl_files')}/{self._task.problem_folder}/{self._task.bddl_file}"

        # Create environment
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=camera_height,
            camera_widths=camera_width,
            has_renderer=False,
            has_offscreen_renderer=True,
        )
        self._env.seed(seed)
        self._obs: dict[str, Any] | None = None

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def task_description(self) -> str:
        return self._task.language

    def reset(self, episode_idx: int = 0) -> dict[str, Any]:
        self._env.reset()
        raw_obs = self._env.set_init_state(self._init_states[episode_idx])
        # Wait for objects to settle (10 no-op steps)
        for _ in range(10):
            raw_obs, _, _, _ = self._env.step([0, 0, 0, 0, 0, 0, -1])
        self._obs = self._process_obs(raw_obs)
        return self._obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).tolist()
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
