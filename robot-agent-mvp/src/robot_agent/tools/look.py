"""LookTool: capture image + VLM analysis (or formatted observation in mock mode)."""

from typing import Any

import httpx
import numpy as np

from nanobot.agent.tools.base import Tool
from robot_agent.context import RobotContext


class LookTool(Tool):

    def __init__(self, ctx: RobotContext):
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "look"

    @property
    def description(self) -> str:
        return "Capture current scene image and analyze it. Returns scene description with object positions."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to look for or describe about the scene",
                },
            },
            "required": ["question"],
        }

    async def execute(self, **kwargs: Any) -> str:
        question = kwargs["question"]
        obs = self._ctx.env.get_observation()

        # Mock mode: format observation as structured text
        if self._ctx.vlm_url is None:
            return self._format_mock_observation(obs, question)

        # VLM mode: send image to VLM endpoint
        return await self._query_vlm(obs, question)

    def _format_mock_observation(self, obs: dict, question: str) -> str:
        ee_pos = obs.get("robot0_eef_pos", [0, 0, 0])
        gripper = "open" if obs.get("robot0_gripper_qpos", [1])[0] > 0.5 else "closed"
        holding = obs.get("holding", None)

        lines = [f"[Scene observation for: {question}]"]
        lines.append(f"End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        lines.append(f"Gripper: {gripper}")
        if holding:
            lines.append(f"Currently holding: {holding}")

        objects = obs.get("objects", {})
        if objects:
            lines.append("Objects in scene:")
            for name, pos in objects.items():
                lines.append(f"  - {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        return "\n".join(lines)

    async def _query_vlm(self, obs: dict, question: str) -> str:
        import base64

        image = obs.get("agentview_image")
        if image is None:
            return "Error: No image available"

        # Encode image as base64 PNG
        from io import BytesIO
        from PIL import Image
        img = Image.fromarray(np.asarray(image, dtype=np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._ctx.vlm_url}/v1/chat/completions",
                    json={
                        "model": "vlm",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                    {"type": "text", "text": question},
                                ],
                            }
                        ],
                        "max_tokens": 512,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error querying VLM: {e}"
