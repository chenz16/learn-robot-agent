"""Model management tools: health check and readiness confirmation."""

import json
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool
from robot_agent.context import RobotContext


class ModelHealthTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "model_health"

    @property
    def description(self) -> str:
        return "Check health status of all model services (VLA, VLM) and simulation environment."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        results = {}

        # VLA health
        try:
            vla_health = await self._ctx.vla.health_check()
            results["vla"] = vla_health
        except Exception as e:
            results["vla"] = {"status": "error", "error": str(e)}

        # VLM health
        if self._ctx.vlm_url:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self._ctx.vlm_url}/health")
                    results["vlm"] = {"status": "ok" if resp.status_code == 200 else "error"}
            except Exception as e:
                results["vlm"] = {"status": "error", "error": str(e)}
        else:
            results["vlm"] = {"status": "disabled", "note": "No VLM URL configured (mock mode)"}

        # Environment
        try:
            obs = self._ctx.env.get_observation()
            results["sim"] = {
                "status": "ok",
                "has_image": "agentview_image" in obs,
                "has_state": "robot0_eef_pos" in obs,
            }
        except Exception as e:
            results["sim"] = {"status": "error", "error": str(e)}

        # Safety
        results["safety"] = {
            "estop_active": self._ctx.safety.estop_active,
            "max_velocity": self._ctx.safety.max_velocity,
        }

        return json.dumps(results, indent=2, default=str)


class ModelEnsureTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx
        self._health_tool = ModelHealthTool(ctx)

    @property
    def name(self) -> str:
        return "model_ensure"

    @property
    def description(self) -> str:
        return "Confirm all required services are ready before starting a task. Call this before the PRAE loop."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        health_json = await self._health_tool.execute()
        health = json.loads(health_json)

        failures = []
        for service, status in health.items():
            if isinstance(status, dict) and status.get("status") == "error":
                failures.append(f"{service}: {status.get('error', 'unknown error')}")

        if self._ctx.safety.estop_active:
            failures.append("safety: E-STOP is active, reset required before operation")

        if failures:
            return "NOT READY. Failures:\n" + "\n".join(f"  - {f}" for f in failures)

        return "All services ready. You may proceed with the task."
