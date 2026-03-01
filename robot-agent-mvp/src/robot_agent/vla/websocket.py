"""WebSocket VLA adapter: connects to openpi's WebSocket policy server."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from robot_agent.vla.base import VLAAdapter

logger = logging.getLogger(__name__)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle. Copied from robosuite."""
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image with letterbox padding to target size."""
    from PIL import Image as PILImage

    if image.shape[0] == height and image.shape[1] == width:
        return image

    pil_img = PILImage.fromarray(image)
    cur_w, cur_h = pil_img.size
    ratio = max(cur_w / width, cur_h / height)
    new_h = int(cur_h / ratio)
    new_w = int(cur_w / ratio)
    resized = pil_img.resize((new_w, new_h), resample=PILImage.BILINEAR)
    canvas = PILImage.new(resized.mode, (width, height), 0)
    pad_h = max(0, (height - new_h) // 2)
    pad_w = max(0, (width - new_w) // 2)
    canvas.paste(resized, (pad_w, pad_h))
    return np.array(canvas)


class WebSocketVLAAdapter(VLAAdapter):
    """Connect to openpi's WebSocket policy server.

    Protocol: msgpack-numpy over WebSocket.
    Observation format for LIBERO:
        observation/image: uint8 (224, 224, 3) — main camera
        observation/wrist_image: uint8 (224, 224, 3) — wrist camera
        observation/state: float32 (8,) — [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
        prompt: str — task instruction
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        resize_size: int = 224,
        replan_steps: int = 5,
    ):
        self._host = host
        self._port = port
        self._resize_size = resize_size
        self._replan_steps = replan_steps
        self._ws = None
        self._server_metadata: dict = {}
        self._packer = None

    def _ensure_connection(self) -> None:
        """Establish WebSocket connection if not connected."""
        if self._ws is not None:
            return

        import msgpack
        import websockets.sync.client

        # Custom msgpack packer/unpacker with numpy support
        def pack_array(obj):
            if isinstance(obj, np.ndarray):
                return {
                    b"__ndarray__": True,
                    b"data": obj.tobytes(),
                    b"dtype": obj.dtype.str,
                    b"shape": obj.shape,
                }
            if isinstance(obj, np.generic):
                return {
                    b"__npgeneric__": True,
                    b"data": obj.item(),
                    b"dtype": obj.dtype.str,
                }
            return obj

        self._pack_array = pack_array
        self._packer = msgpack.Packer(default=pack_array)

        def unpack_array(obj):
            if b"__ndarray__" in obj:
                return np.ndarray(
                    buffer=obj[b"data"],
                    dtype=np.dtype(obj[b"dtype"]),
                    shape=obj[b"shape"],
                )
            if b"__npgeneric__" in obj:
                return np.dtype(obj[b"dtype"]).type(obj[b"data"])
            return obj

        self._unpack_array = unpack_array

        uri = f"ws://{self._host}:{self._port}"
        logger.info("Connecting to openpi server at %s ...", uri)

        while True:
            try:
                conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    open_timeout=60,
                    ping_interval=None,  # disable keepalive (inference can be slow)
                    close_timeout=60,
                )
                # Server sends metadata as first message
                raw_meta = conn.recv()
                self._server_metadata = msgpack.unpackb(raw_meta, object_hook=unpack_array)
                self._ws = conn
                logger.info("Connected to openpi server. Metadata: %s", self._server_metadata)
                return
            except ConnectionRefusedError:
                logger.info("Waiting for openpi server at %s ...", uri)
                time.sleep(3)

    def _build_obs_dict(self, observation: dict[str, Any], instruction: str) -> dict:
        """Build openpi-compatible observation dict from env observation."""
        sz = self._resize_size

        # Main camera image
        img = observation.get("agentview_image")
        if img is not None:
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = (255 * img).astype(np.uint8)
            img = _resize_with_pad(img, sz, sz)
        else:
            img = np.zeros((sz, sz, 3), dtype=np.uint8)

        # Wrist camera image
        wrist_img = observation.get("robot0_eye_in_hand_image")
        if wrist_img is not None:
            wrist_img = np.ascontiguousarray(wrist_img)
            if wrist_img.dtype != np.uint8:
                wrist_img = (255 * wrist_img).astype(np.uint8)
            wrist_img = _resize_with_pad(wrist_img, sz, sz)
        else:
            wrist_img = np.zeros((sz, sz, 3), dtype=np.uint8)

        # Robot state: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
        eef_pos = observation.get("robot0_eef_pos", np.zeros(3))
        eef_quat = observation.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))
        gripper_qpos = observation.get("robot0_gripper_qpos", np.zeros(2))

        state = np.concatenate([
            np.asarray(eef_pos, dtype=np.float32),
            _quat2axisangle(np.asarray(eef_quat, dtype=np.float64)).astype(np.float32),
            np.asarray(gripper_qpos, dtype=np.float32),
        ])

        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": instruction,
        }

    async def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Send observation to openpi server, receive action chunk."""
        import msgpack

        self._ensure_connection()
        assert self._ws is not None
        assert self._packer is not None

        obs_dict = self._build_obs_dict(observation, instruction)

        # Send observation
        data = self._packer.pack(obs_dict)
        self._ws.send(data)

        # Receive action chunk
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from openpi server:\n{response}")

        result = msgpack.unpackb(response, object_hook=self._unpack_array)
        actions = result["actions"]  # shape: (horizon, action_dim)

        # For LIBERO: take first 7 dims (rest is padding)
        if actions.shape[-1] > 7:
            actions = actions[:, :7]

        return actions

    def reset(self) -> None:
        """Reset for new episode (openpi server is stateless per-request)."""
        pass

    async def health_check(self) -> dict[str, Any]:
        """Check server health via HTTP /healthz endpoint."""
        import httpx

        url = f"http://{self._host}:{self._port}/healthz"
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                latency = (time.monotonic() - start) * 1000
                if resp.status_code == 200:
                    return {"status": "ok", "latency_ms": latency, "backend": "websocket"}
                return {"status": "error", "code": resp.status_code, "latency_ms": latency}
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return {"status": "error", "error": str(e), "latency_ms": latency}

    def get_action_horizon(self) -> int:
        return self._replan_steps

    def close(self) -> None:
        """Close WebSocket connection."""
        if self._ws is not None:
            self._ws.close()
            self._ws = None
