#!/usr/bin/env python3
"""
r03_task_decomposition.py - Task Decomposition

The model breaks complex goals into subtasks via a TodoManager,
then executes them one by one. A nag reminder forces progress tracking.

    +----------+      +-------+      +-------------------+
    |   User   | ---> |  LLM  | ---> | Tools             |
    | "set the |      |       |      | look, move, grasp |
    |  table"  |      +---+---+      | + todo            |
    +----------+          ^          +-------------------+
                          |                  |
                          +------------------+
                                    |
                       +------------+------------+
                       | TodoManager             |
                       | [ ] 1. look at scene    |
                       | [>] 2. pick up apple    |
                       | [ ] 3. place on plate   |
                       | [x] 4. verify result    |
                       +-------------------------+
                                    |
                       if rounds_since_todo >= 3:
                         inject <reminder>

Key insight: "An agent without a plan drifts. Give it a todo list."

Before r03, the agent just reacts. Now it plans first, then executes
step by step, tracking what's done and what's left.
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

SIM_URL = os.environ.get("SIM_URL", "")
VLM_URL = os.environ.get("VLM_URL", "")
VLA_URL = os.environ.get("VLA_URL", "")


SYSTEM = """\
You are a robot agent controlling a Unitree G1 humanoid robot in a kitchen.

Tools: look, move, grasp, todo.

WORKFLOW — always follow this pattern:
1. Use 'todo' FIRST to decompose the task into subtasks
2. Mark each subtask 'in_progress' before starting it
3. Execute using look/move/grasp
4. Mark subtask 'completed' when done
5. Repeat until all subtasks are complete

Only one subtask should be in_progress at a time.
Keep your todo list updated — it helps you stay on track."""


# ============================================================
# TodoManager (from learn-claude-code s03)
# ============================================================

class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos")
        validated = []
        in_progress_count = 0
        for item in items:
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(len(validated) + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# ============================================================
# Mock Environment (stateful, from r02)
# ============================================================

GRASP_DISTANCE = 0.08


class MockRobotEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.objects = {
            "red apple":   {"pos": [0.45,  0.12, 0.82], "on": "counter"},
            "white plate": {"pos": [0.45, -0.15, 0.80], "on": "counter"},
            "blue mug":    {"pos": [0.50,  0.30, 0.82], "on": "counter"},
            "fork":        {"pos": [0.55, -0.20, 0.81], "on": "counter"},
            "napkin":      {"pos": [0.60,  0.00, 0.81], "on": "counter"},
        }
        self.ee_pos = [0.30, 0.0, 0.95]
        self.gripper = "open"
        self.holding = None

    def _distance(self, a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _find_object(self, name_query: str):
        q = name_query.lower()
        for name in self.objects:
            if q in name or name in q:
                return name
        return None

    def look(self, question: str = "") -> str:
        lines = ["Scene observation:"]
        for name, obj in self.objects.items():
            loc = "held by gripper" if name == self.holding else f"on {obj['on']}"
            lines.append(f"  - {name}: pos={obj['pos']}, {loc}")
        lines.append(f"Robot: ee={self.ee_pos}, gripper={self.gripper}, holding={self.holding or 'nothing'}")

        if question:
            q = question.lower()
            matched = self._find_object(q)
            if matched:
                obj = self.objects[matched]
                d = self._distance(self.ee_pos, obj["pos"])
                loc = "held by gripper" if matched == self.holding else f"on {obj['on']}"
                lines.append(f"=> {matched} is {loc}, {d:.2f}m from end-effector.")
            elif any(w in q for w in ("success", "done", "complete", "plate")):
                on_plate = [n for n, o in self.objects.items()
                            if o["on"] == "white plate" and n != "white plate"]
                if on_plate:
                    lines.append(f"=> Objects on plate: {on_plate}.")
                else:
                    lines.append("=> The plate is empty.")
            elif any(w in q for w in ("hold", "grip", "carry")):
                lines.append(f"=> Gripper is {self.gripper}, holding: {self.holding or 'nothing'}.")
            else:
                lines.append("=> All objects are on the counter within the robot's workspace.")
        return "\n".join(lines)

    def move(self, target: str = "", position: list = None) -> str:
        if target:
            matched = self._find_object(target)
            if not matched:
                return f"Error: unknown target '{target}'. Known: {list(self.objects.keys())}"
            dest = list(self.objects[matched]["pos"])
        elif position and len(position) >= 3:
            dest = position[:3]
        else:
            return "Error: provide 'target' (object name) or 'position' ([x,y,z])."

        old_pos = list(self.ee_pos)
        self.ee_pos = dest
        dist = self._distance(old_pos, dest)

        if self.holding and self.holding in self.objects:
            self.objects[self.holding]["pos"] = list(dest)

        result = f"Moved: {old_pos} -> {dest} ({dist:.2f}m)"
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
            for name, obj in self.objects.items():
                d = self._distance(self.ee_pos, obj["pos"])
                if d < GRASP_DISTANCE and name != self.holding:
                    self.holding = name
                    return f"Gripper closed. Grasped '{name}' ({d:.3f}m)."
            return "Gripper closed. Nothing within grasp range."
        elif action == "open":
            if self.gripper == "open":
                return "Gripper already open."
            self.gripper = "open"
            if self.holding:
                released = self.holding
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
# Real mode handlers (same as r02)
# ============================================================

def real_look(question: str = "") -> str:
    resp = requests.get(f"{SIM_URL}/render", timeout=5)
    img_b64 = base64.b64encode(resp.content).decode()
    prompt = question if question else "Describe the scene. List all objects and robot state."
    vlm_resp = requests.post(f"{VLM_URL}/analyze", json={"image": img_b64, "prompt": prompt}, timeout=30)
    result = vlm_resp.json()
    return f"{result.get('analysis', 'no response')} [{result.get('inference_ms', 0):.0f}ms]"


def real_act(instruction: str, steps: int = 10) -> str:
    obs_resp = requests.get(f"{SIM_URL}/raw_observation", timeout=5)
    obs = obs_resp.json()
    step_result = {}
    for i in range(steps):
        vla_resp = requests.post(f"{VLA_URL}/predict", json={"observation": obs, "instruction": instruction}, timeout=30)
        action = vla_resp.json().get("action", {})
        sim_resp = requests.post(f"{SIM_URL}/step", json={"action": action}, timeout=5)
        step_result = sim_resp.json()
        if step_result.get("terminated") or step_result.get("success"):
            return f"Done after {i+1} steps (success={step_result.get('success')}). Reward: {step_result.get('total_reward', 0):.3f}"
        obs_resp = requests.get(f"{SIM_URL}/raw_observation", timeout=5)
        obs = obs_resp.json()
    return f"Executed {steps} steps. Reward: {step_result.get('total_reward', 0):.3f}"


# ============================================================
# Tool dispatch map (r02 tools + todo)
# ============================================================

def run_look(question=""):
    if SIM_URL and VLM_URL:
        try:
            return real_look(question)
        except Exception as e:
            return f"[VLM error: {e}]\n{mock_env.look(question)}"
    return mock_env.look(question)

def run_move(target="", position=None):
    if SIM_URL and VLA_URL:
        try:
            instr = f"move toward {target}" if target else f"move to {position}"
            return real_act(instr)
        except Exception as e:
            return f"[VLA error: {e}]\n{mock_env.move(target, position)}"
    return mock_env.move(target, position)

def run_grasp(action):
    if SIM_URL and VLA_URL:
        try:
            return real_act(f"{'close' if action == 'close' else 'open'} the gripper")
        except Exception as e:
            return f"[VLA error: {e}]\n{mock_env.grasp(action)}"
    return mock_env.grasp(action)


TOOL_HANDLERS = {
    "look":  lambda **kw: run_look(kw.get("question", "")),
    "move":  lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp": lambda **kw: run_grasp(kw["action"]),
    "todo":  lambda **kw: TODO.update(kw["items"]),
}

TOOLS = [
    {
        "name": "look",
        "description": "Observe the scene. Optionally ask a question.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "move",
        "description": "Move end-effector to a target object or [x,y,z] position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Object name, e.g. 'red apple'."},
                "position": {"type": "array", "items": {"type": "number"}, "description": "[x,y,z]"},
            },
            "required": [],
        },
    },
    {
        "name": "grasp",
        "description": "Control gripper: 'close' to grab, 'open' to release.",
        "input_schema": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["open", "close"]}},
            "required": ["action"],
        },
    },
    {
        "name": "todo",
        "description": "Update the task plan. Break complex goals into subtasks and track progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["id", "text", "status"],
                    },
                },
            },
            "required": ["items"],
        },
    },
]


# ============================================================
# Agent loop — one addition: nag reminder if todo not updated
# ============================================================

def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                # Print tool call
                if block.name == "todo":
                    print(f"\033[35m[todo]\n{output}\033[0m")
                    used_todo = True
                else:
                    print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

        # Nag reminder: nudge the agent to update its plan
        if rounds_since_todo >= 3:
            results.insert(0, {
                "type": "text",
                "text": "<reminder>Update your todo list. Mark completed tasks and set the next one in_progress.</reminder>",
            })

        messages.append({"role": "user", "content": results})


# ============================================================
# REPL
# ============================================================

if __name__ == "__main__":
    mode = "REAL" if (SIM_URL and VLA_URL) else "MOCK"
    print(f"\033[32m[r03] Task Decomposition  |  {mode}\033[0m")
    print(f"\033[90mTools: look, move, grasp, todo\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip().lower() == "reset":
            mock_env.reset()
            TODO.items = []
            history = []
            print("Environment, todos, and history reset.\n")
            continue
        if query.strip().lower() == "todos":
            print(TODO.render())
            print()
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"):
                    print(block.text)
        print()
