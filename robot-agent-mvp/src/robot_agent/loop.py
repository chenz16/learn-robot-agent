"""LoopManager: asyncio-based VLA control loop execution engine."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from robot_agent.env.base import RobotEnv
from robot_agent.routing import RouteLevel
from robot_agent.safety import SafetyManager
from robot_agent.termination import (
    CompositeTerminator,
    PositionThresholdTerminator,
    StepLimitTerminator,
)
from robot_agent.vla.base import VLAAdapter


@dataclass
class SubtaskResult:
    """Result of a completed subtask."""
    subtask_id: str
    status: str  # "completed" | "failed" | "stopped"
    steps_executed: int = 0
    reason: str = ""
    total_reward: float = 0.0
    final_observation: dict[str, Any] = field(default_factory=dict)


class LoopManager:
    """Manages VLA control loops as asyncio tasks."""

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, SubtaskResult] = {}

    async def start_subtask(
        self,
        instruction: str,
        target: dict[str, Any],
        route_level: RouteLevel,
        env: RobotEnv,
        vla: VLAAdapter,
        safety: SafetyManager,
    ) -> str:
        """Launch a VLA control loop. Returns subtask_id."""
        subtask_id = f"sub_{uuid.uuid4().hex[:8]}"
        task = asyncio.create_task(
            self._control_loop(subtask_id, instruction, target, route_level, env, vla, safety)
        )
        self._tasks[subtask_id] = task
        logger.info("Started subtask {} : {}", subtask_id, instruction[:80])
        return subtask_id

    async def wait_for_completion(self, subtask_id: str, timeout: float = 60.0) -> SubtaskResult:
        """Wait for a subtask to complete."""
        task = self._tasks.get(subtask_id)
        if task is None:
            if subtask_id in self._results:
                return self._results[subtask_id]
            return SubtaskResult(subtask_id=subtask_id, status="failed", reason="Unknown subtask")

        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._results[subtask_id] = SubtaskResult(
                subtask_id=subtask_id, status="failed", reason=f"Timeout after {timeout}s"
            )

        return self._results.get(
            subtask_id,
            SubtaskResult(subtask_id=subtask_id, status="failed", reason="No result recorded"),
        )

    def get_status(self) -> dict[str, Any]:
        """Get status of all loops."""
        active = {sid: "running" for sid, t in self._tasks.items() if not t.done()}
        completed = {
            sid: {"status": r.status, "steps": r.steps_executed, "reason": r.reason}
            for sid, r in self._results.items()
        }
        return {
            "active_count": len(active),
            "completed_count": len(completed),
            "active": active,
            "completed": completed,
        }

    def stop_all(self) -> int:
        """Cancel all running loops. Returns count of cancelled tasks."""
        cancelled = 0
        for sid, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    async def _control_loop(
        self,
        subtask_id: str,
        instruction: str,
        target: dict[str, Any],
        route_level: RouteLevel,
        env: RobotEnv,
        vla: VLAAdapter,
        safety: SafetyManager,
    ) -> None:
        """The inner control loop: observe → predict → safety → step → terminate."""
        # Build terminators
        terminators = [StepLimitTerminator(route_level.max_steps)]
        target_pos = target.get("position")
        if target_pos is not None:
            terminators.append(
                PositionThresholdTerminator(target_pos, route_level.position_threshold)
            )
        terminator = CompositeTerminator(terminators, mode="any")

        vla.reset()
        step_count = 0
        total_reward = 0.0
        interval = 1.0 / route_level.action_hz
        status = "completed"
        reason = ""
        final_obs: dict[str, Any] = {}

        try:
            while True:
                loop_start = time.monotonic()

                obs = env.get_observation()
                actions = await asyncio.to_thread(self._predict_sync, vla, obs, instruction)

                for action in actions:
                    # Safety check
                    safe_action, warnings = safety.check_action(action)
                    if safety.estop_active:
                        status = "stopped"
                        reason = "E-STOP activated"
                        final_obs = obs
                        logger.warning("Subtask {} stopped by E-STOP", subtask_id)
                        self._results[subtask_id] = SubtaskResult(
                            subtask_id=subtask_id,
                            status=status,
                            steps_executed=step_count,
                            reason=reason,
                            total_reward=total_reward,
                            final_observation=final_obs,
                        )
                        return

                    obs, reward, done, info = env.step(safe_action)
                    step_count += 1
                    total_reward += reward

                    # Check termination
                    should_stop, term_reason = terminator.should_terminate(step_count, obs)
                    if should_stop or done:
                        reason = term_reason or "Environment signaled done"
                        final_obs = obs
                        logger.info("Subtask {} completed at step {}: {}", subtask_id, step_count, reason)
                        self._results[subtask_id] = SubtaskResult(
                            subtask_id=subtask_id,
                            status=status,
                            steps_executed=step_count,
                            reason=reason,
                            total_reward=total_reward,
                            final_observation=final_obs,
                        )
                        return

                # Maintain target Hz
                elapsed = time.monotonic() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            self._results[subtask_id] = SubtaskResult(
                subtask_id=subtask_id,
                status="stopped",
                steps_executed=step_count,
                reason="Cancelled",
                total_reward=total_reward,
            )
            raise
        except Exception as e:
            logger.error("Subtask {} failed: {}", subtask_id, e)
            self._results[subtask_id] = SubtaskResult(
                subtask_id=subtask_id,
                status="failed",
                steps_executed=step_count,
                reason=str(e),
                total_reward=total_reward,
            )
        finally:
            self._tasks.pop(subtask_id, None)

    @staticmethod
    def _predict_sync(vla: VLAAdapter, obs: dict, instruction: str) -> np.ndarray:
        """Synchronous wrapper for VLA predict (called via asyncio.to_thread)."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(vla.predict(obs, instruction))
        finally:
            loop.close()
