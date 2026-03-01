"""HTTP VLA adapter: connects to a remote VLA inference service."""

import base64
import time
from io import BytesIO
from typing import Any

import httpx
import numpy as np

from robot_agent.vla.base import VLAAdapter


class HTTPVLAAdapter(VLAAdapter):
    """Connect to a VLA server via HTTP POST."""

    def __init__(self, base_url: str = "http://localhost:8020", timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._action_horizon: int | None = None

    async def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        payload = {
            "instruction": instruction,
            "observation": self._serialize_obs(observation),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/predict", json=payload)
            resp.raise_for_status()
            data = resp.json()
        actions = np.array(data["actions"], dtype=np.float64)
        if self._action_horizon is None and "action_horizon" in data:
            self._action_horizon = data["action_horizon"]
        return actions

    def reset(self) -> None:
        pass

    async def health_check(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                latency = (time.monotonic() - start) * 1000
                if resp.status_code == 200:
                    return {"status": "ok", "latency_ms": latency, "backend": "http"}
                return {"status": "error", "code": resp.status_code, "latency_ms": latency}
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return {"status": "error", "error": str(e), "latency_ms": latency}

    def get_action_horizon(self) -> int:
        return self._action_horizon or 4

    def _serialize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Serialize observation for HTTP transport (images → base64)."""
        result = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 3 and value.shape[2] == 3:
                    # Image: encode as base64 PNG
                    try:
                        from PIL import Image
                        img = Image.fromarray(value.astype(np.uint8))
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        result[key] = {
                            "type": "image",
                            "data": base64.b64encode(buf.getvalue()).decode(),
                            "shape": list(value.shape),
                        }
                    except ImportError:
                        result[key] = {
                            "type": "array",
                            "data": value.tolist(),
                            "shape": list(value.shape),
                        }
                else:
                    result[key] = {"type": "array", "data": value.tolist()}
            else:
                result[key] = value
        return result
