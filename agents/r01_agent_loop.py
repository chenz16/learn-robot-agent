#!/usr/bin/env python3
"""
r01_agent_loop.py - The Robot Agent Loop

The entire secret of a robot agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools=[look])
        execute tools
        append results

    +----------+      +-------+      +------+      +-----------+
    |   User   | ---> |  LLM  | ---> | look | ---> | Sim + VLM |
    |  command |      |       |      |      |      |  servers  |
    +----------+      +---+---+      +--+---+      +-----------+
                          ^             |
                          | scene desc  |
                          +-------------+
                          (loop continues)

One loop, one tool: the robot first needs to see.
Before you can grasp, move, or navigate — you must perceive.

Three modes:
  MOCK  - no servers needed, returns simulated scene (default)
  SIM   - real sim observation, formatted as text
  FULL  - sim camera image -> VLM -> natural language description
"""

import os
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


SYSTEM = """\
You are a robot agent controlling a Unitree G1 humanoid robot in a kitchen.

You can observe the environment using the 'look' tool.
Right now you can ONLY perceive — describe what you see, analyze the scene, \
and answer questions about the environment.

When the user gives you a task, first observe the scene to understand \
the current state. Be specific about object positions, distances, \
and spatial relationships."""


TOOLS = [{
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
                "description": (
                    "What to focus on, e.g. 'where is the apple?', "
                    "'is the gripper open?'. Empty = general overview."
                ),
            },
        },
        "required": [],
    },
}]


# ============================================================
# Mock Environment (no servers needed)
# ============================================================

MOCK_SCENE = {
    "objects": [
        {"name": "red apple",   "pos": [0.45,  0.12, 0.82], "on": "counter"},
        {"name": "white plate", "pos": [0.45, -0.15, 0.80], "on": "counter"},
        {"name": "blue mug",    "pos": [0.50,  0.30, 0.82], "on": "counter"},
    ],
    "robot": {
        "ee_pos": [0.30, 0.0, 0.95],
        "gripper": "open",
        "base_pos": [0.0, 0.0],
    },
}


def mock_look(question: str = "") -> str:
    lines = ["Kitchen counter — simulated environment."]
    lines.append("Objects:")
    for o in MOCK_SCENE["objects"]:
        lines.append(f"  - {o['name']} at {o['pos']}, on {o['on']}")
    r = MOCK_SCENE["robot"]
    lines.append(f"Robot: ee={r['ee_pos']}, gripper={r['gripper']}, base={r['base_pos']}")

    if question:
        q = question.lower()
        if "apple" in q:
            lines.append("=> The red apple is on the counter, ~27cm from the plate.")
        elif "plate" in q:
            lines.append("=> The white plate is on the counter, empty, no items on it.")
        elif "reach" in q or "grasp" in q:
            lines.append("=> The apple is within arm's reach (~20cm from end-effector).")
        elif "mug" in q:
            lines.append("=> The blue mug is to the right side, handle facing left.")
        else:
            lines.append("=> All objects are on the counter within the robot's workspace.")

    return "\n".join(lines)


# ============================================================
# Real Environment (HTTP services)
# ============================================================

def real_look_sim_only() -> str:
    """SIM mode: get observation dict, format as text."""
    resp = requests.get(f"{SIM_URL}/observation", timeout=5)
    obs = resp.json()
    lines = ["Live observation from simulator:"]
    for k, v in obs.items():
        if k in ("ego_view", "tpp_view"):
            lines.append(f"  {k}: [image {len(v)} chars base64]")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def real_look_full(question: str = "") -> str:
    """FULL mode: sim camera -> VLM -> natural language."""
    # 1. Get camera image from sim
    resp = requests.get(f"{SIM_URL}/render", timeout=5)
    img_b64 = base64.b64encode(resp.content).decode()

    # 2. Send to VLM for analysis
    prompt = question if question else (
        "Describe the scene. List all visible objects, "
        "their positions, and the robot arm state."
    )
    vlm_resp = requests.post(
        f"{VLM_URL}/analyze",
        json={"image": img_b64, "prompt": prompt},
        timeout=30,
    )
    result = vlm_resp.json()
    ms = result.get("inference_ms", 0)
    return f"{result.get('analysis', 'no response')} [{ms:.0f}ms]"


# ============================================================
# Tool dispatcher
# ============================================================

def run_look(question: str = "") -> str:
    # FULL mode
    if SIM_URL and VLM_URL:
        try:
            return real_look_full(question)
        except Exception as e:
            return f"[VLM error: {e}] fallback:\n{mock_look(question)}"

    # SIM-only mode
    if SIM_URL:
        try:
            return real_look_sim_only()
        except Exception as e:
            return f"[Sim error: {e}] fallback:\n{mock_look(question)}"

    # MOCK mode
    return mock_look(question)


# ============================================================
# The agent loop — same structure as learn-claude-code s01
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
                q = block.input.get("question", "")
                print(f"\033[33m[look] {q or '(general observation)'}\033[0m")
                output = run_look(q)
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
    if SIM_URL and VLM_URL:
        mode = "FULL (Sim + VLM)"
    elif SIM_URL:
        mode = "SIM-only"
    else:
        mode = "MOCK"

    print(f"\033[32m[r01] Robot Agent Loop  |  {mode}\033[0m")
    print(f"\033[90mSIM={SIM_URL or '-'}  VLM={VLM_URL or '-'}\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print final text response
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"):
                    print(block.text)
        print()
