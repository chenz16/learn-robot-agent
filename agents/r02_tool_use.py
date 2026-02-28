#!/usr/bin/env python3
"""
r02_tool_use.py - Robot Tools

The agent loop from r01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  command |      |       |      | {                |
    +----------+      +---+---+      |   look:  percept |
                          ^          |   move:  motion  |
                          |          |   grasp: gripper |
                          +----------+ }                |
                          tool_result+------------------+
                                        |   |   |
                                        v   v   v
                                      VLM  Sim  VLA

Key insight: "The loop didn't change at all. I just added tools."

With three tools, the LLM can now compose a full pick-and-place:
  1. look  -> see apple at [0.45, 0.12, 0.82]
  2. move  -> move end-effector to apple
  3. grasp -> close gripper (pick up)
  4. move  -> move to plate position
  5. grasp -> open gripper (place down)
  6. look  -> verify success
"""

import os
import math
import json
import base64

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ.get("MODEL_ID", "claude-sonnet-4-20250514")

# Robot service endpoints (leave empty for mock mode)
SIM_URL = os.environ.get("SIM_URL", "")   # e.g. http://localhost:8030
VLM_URL = os.environ.get("VLM_URL", "")   # e.g. http://localhost:8010
VLA_URL = os.environ.get("VLA_URL", "")   # e.g. http://localhost:8020


SYSTEM = """\
You are a robot agent controlling a Unitree G1 humanoid robot in a kitchen.

You have three tools:
- look: observe the scene (always look first to understand the environment)
- move: move the robot arm to a target object or position
- grasp: open or close the gripper

To pick and place an object, use this sequence:
  1. look (understand the scene)
  2. move to target object
  3. grasp close (pick it up)
  4. move to destination
  5. grasp open (put it down)
  6. look (verify success)

Act step by step. After each action, check the result before proceeding."""


# ============================================================
# Tool schemas
# ============================================================

TOOLS = [
    {
        "name": "look",
        "description": (
            "Observe the scene through the robot's cameras. "
            "Optionally ask a specific question about what you see."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to focus on, e.g. 'where is the apple?'. Empty = general overview.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "move",
        "description": (
            "Move the robot's end-effector to a target. "
            "Specify either an object name or [x,y,z] coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Object name to move toward, e.g. 'red apple', 'white plate'.",
                },
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] coordinates. Used if target is not given.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "grasp",
        "description": "Control the robot gripper. Use 'close' to grab, 'open' to release.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open", "close"],
                    "description": "'close' to grab an object, 'open' to release.",
                },
            },
            "required": ["action"],
        },
    },
]


# ============================================================
# Mock Environment (stateful — actions change the world)
# ============================================================

GRASP_DISTANCE = 0.08  # meters — close enough to pick up


class MockRobotEnv:
    """Stateful simulated environment. Actions update the world."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.objects = {
            "red apple":   {"pos": [0.45,  0.12, 0.82], "on": "counter"},
            "white plate": {"pos": [0.45, -0.15, 0.80], "on": "counter"},
            "blue mug":    {"pos": [0.50,  0.30, 0.82], "on": "counter"},
        }
        self.ee_pos = [0.30, 0.0, 0.95]
        self.gripper = "open"
        self.holding = None

    def _distance(self, a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _nearest_object(self):
        best, best_d = None, float("inf")
        for name, obj in self.objects.items():
            d = self._distance(self.ee_pos, obj["pos"])
            if d < best_d:
                best, best_d = name, d
            return best, best_d

    def look(self, question: str = "") -> str:
        lines = ["Scene observation:"]
        for name, obj in self.objects.items():
            loc = f"on {obj['on']}"
            if name == self.holding:
                loc = "held by gripper"
            lines.append(f"  - {name}: pos={obj['pos']}, {loc}")
        lines.append(f"Robot: ee={self.ee_pos}, gripper={self.gripper}, holding={self.holding or 'nothing'}")

        if question:
            q = question.lower()
            for name in self.objects:
                if name.split()[-1] in q or name in q:
                    obj = self.objects[name]
                    d = self._distance(self.ee_pos, obj["pos"])
                    loc = "held by gripper" if name == self.holding else f"on {obj['on']}"
                    lines.append(f"=> {name} is {loc}, {d:.2f}m from end-effector.")
                    break
            else:
                if "reach" in q or "close" in q or "near" in q:
                    nearest, d = self._nearest_object()
                    lines.append(f"=> Nearest object: {nearest} at {d:.2f}m.")
                elif "hold" in q or "grip" in q or "carry" in q:
                    lines.append(f"=> Gripper is {self.gripper}, holding: {self.holding or 'nothing'}.")
                elif "success" in q or "done" in q or "plate" in q:
                    on_plate = [n for n, o in self.objects.items()
                                if o["on"] == "white plate" and n != "white plate"]
                    if on_plate:
                        lines.append(f"=> Objects on plate: {on_plate}. Task looks successful!")
                    else:
                        lines.append("=> The plate is empty. Task not yet complete.")

        return "\n".join(lines)

    def move(self, target: str = "", position: list = None) -> str:
        # Resolve target position
        if target:
            target_lower = target.lower()
            matched = None
            for name in self.objects:
                if target_lower in name or name in target_lower:
                    matched = name
                    break
            if not matched:
                return f"Error: unknown target '{target}'. Known objects: {list(self.objects.keys())}"
            dest = list(self.objects[matched]["pos"])
        elif position and len(position) >= 3:
            dest = position[:3]
        else:
            return "Error: provide either 'target' (object name) or 'position' ([x,y,z])."

        # Move end-effector
        old_pos = list(self.ee_pos)
        self.ee_pos = dest
        dist = self._distance(old_pos, dest)

        # If holding something, it moves with the gripper
        if self.holding and self.holding in self.objects:
            self.objects[self.holding]["pos"] = list(dest)

        result = f"Moved end-effector: {old_pos} -> {dest} ({dist:.2f}m)"
        if target:
            result += f", near '{target}'"
        if self.holding:
            result += f". Carrying {self.holding}."
        return result

    def grasp(self, action: str) -> str:
        if action == "close":
            if self.gripper == "close":
                return "Gripper already closed."
            self.gripper = "close"
            # Check if any object is close enough to pick up
            for name, obj in self.objects.items():
                d = self._distance(self.ee_pos, obj["pos"])
                if d < GRASP_DISTANCE and name != self.holding:
                    self.holding = name
                    return f"Gripper closed. Grasped '{name}' (distance was {d:.3f}m)."
            return "Gripper closed. Nothing within grasp range."

        elif action == "open":
            if self.gripper == "open":
                return "Gripper already open."
            self.gripper = "open"
            if self.holding:
                released = self.holding
                # Determine what surface the object lands on
                for name, obj in self.objects.items():
                    if name != released:
                        d = self._distance(self.ee_pos, obj["pos"])
                        if d < 0.10:
                            self.objects[released]["on"] = name
                            self.holding = None
                            return f"Gripper opened. Released '{released}' onto '{name}'."
                self.objects[released]["on"] = "counter"
                self.holding = None
                return f"Gripper opened. Released '{released}' onto counter."
            return "Gripper opened. Was not holding anything."

        return f"Error: action must be 'open' or 'close', got '{action}'."


mock_env = MockRobotEnv()


# ============================================================
# Real mode handlers (HTTP services)
# ============================================================

def real_look(question: str = "") -> str:
    """FULL mode: sim camera -> VLM -> text description."""
    resp = requests.get(f"{SIM_URL}/render", timeout=5)
    img_b64 = base64.b64encode(resp.content).decode()
    prompt = question if question else (
        "Describe the scene. List all visible objects, their positions, and the robot arm state."
    )
    vlm_resp = requests.post(
        f"{VLM_URL}/analyze",
        json={"image": img_b64, "prompt": prompt},
        timeout=30,
    )
    result = vlm_resp.json()
    return f"{result.get('analysis', 'no response')} [{result.get('inference_ms', 0):.0f}ms]"


def real_act(instruction: str, steps: int = 10) -> str:
    """Send instruction to VLA, execute action steps in sim."""
    # Get current observation from sim
    obs_resp = requests.get(f"{SIM_URL}/raw_observation", timeout=5)
    obs = obs_resp.json()

    results = []
    for i in range(steps):
        # VLA predicts action from observation + instruction
        vla_resp = requests.post(
            f"{VLA_URL}/predict",
            json={"observation": obs, "instruction": instruction},
            timeout=30,
        )
        action = vla_resp.json().get("action", {})

        # Step simulation
        sim_resp = requests.post(
            f"{SIM_URL}/step",
            json={"action": action},
            timeout=5,
        )
        step_result = sim_resp.json()

        if step_result.get("terminated") or step_result.get("success"):
            results.append(f"Step {i+1}: terminated (success={step_result.get('success')})")
            break

        # Update observation for next step
        obs_resp = requests.get(f"{SIM_URL}/raw_observation", timeout=5)
        obs = obs_resp.json()

    if not results:
        results.append(f"Executed {steps} steps.")
    reward = step_result.get("total_reward", 0)
    results.append(f"Total reward: {reward:.3f}")
    return "\n".join(results)


# ============================================================
# Tool dispatch map
# ============================================================

def run_look(question: str = "") -> str:
    if SIM_URL and VLM_URL:
        try:
            return real_look(question)
        except Exception as e:
            return f"[VLM error: {e}] fallback:\n{mock_env.look(question)}"
    return mock_env.look(question)


def run_move(target: str = "", position: list = None) -> str:
    if SIM_URL and VLA_URL:
        try:
            instruction = f"move toward {target}" if target else f"move to position {position}"
            return real_act(instruction)
        except Exception as e:
            return f"[VLA error: {e}] fallback:\n{mock_env.move(target, position)}"
    return mock_env.move(target, position)


def run_grasp(action: str) -> str:
    if SIM_URL and VLA_URL:
        try:
            return real_act(f"{'close' if action == 'close' else 'open'} the gripper")
        except Exception as e:
            return f"[VLA error: {e}] fallback:\n{mock_env.grasp(action)}"
    return mock_env.grasp(action)


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "look":  lambda **kw: run_look(kw.get("question", "")),
    "move":  lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp": lambda **kw: run_grasp(kw["action"]),
}


# ============================================================
# The agent loop — UNCHANGED from r01
# ============================================================

def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                print(f"\033[90m{output[:500]}\033[0m")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})


# ============================================================
# REPL
# ============================================================

if __name__ == "__main__":
    real = SIM_URL and VLA_URL
    mode = "REAL" if real else "MOCK"
    print(f"\033[32m[r02] Robot Tool Use  |  {mode}\033[0m")
    print(f"\033[90mTools: look, move, grasp  |  SIM={SIM_URL or '-'} VLM={VLM_URL or '-'} VLA={VLA_URL or '-'}\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip().lower() == "reset":
            mock_env.reset()
            history = []
            print("Environment and history reset.\n")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"):
                    print(block.text)
        print()
