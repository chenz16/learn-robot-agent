#!/usr/bin/env python3
"""
r05_robot_skills.py - Robot Skills (SKILL.md)

Two-layer skill injection for manipulation recipes:

    Layer 1 (cheap): skill names in system prompt (~30 tokens/skill)
    Layer 2 (on demand): full SKILL.md body injected via tool_result

    skills/
      top-grasp/
        SKILL.md     <-- "approach from above, descend, close gripper..."
      side-grasp/
        SKILL.md     <-- "approach from side, grasp handle..."
      pour/
        SKILL.md
      precision-place/
        SKILL.md

    System prompt:
    +----------------------------------------------+
    | Skills available:                            |
    |   - top-grasp: for flat objects on surfaces  |  <-- Layer 1
    |   - side-grasp: for tall objects / handles   |
    |   - pour: pour between containers            |
    |   - precision-place: careful placement       |
    +----------------------------------------------+

    When model calls load_skill("side-grasp"):
    +----------------------------------------------+
    | <skill name="side-grasp">                    |
    |   1. Pre-position to the SIDE               |  <-- Layer 2
    |   2. Open gripper                            |
    |   3. Approach horizontally                   |
    |   4. Grasp, lift                             |
    | </skill>                                     |
    +----------------------------------------------+

Key insight: "Don't hardcode manipulation strategies. Load them as skills."
"""

import os
import re
import math
import json
import base64
from pathlib import Path

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

# Skills directory — relative to repo root, not agents/
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


# ============================================================
# SkillLoader (from learn-claude-code s05)
# ============================================================

class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """Layer 1: short descriptions for system prompt."""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: full body loaded on demand."""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)


# ============================================================
# System prompt — Layer 1 skill descriptions injected here
# ============================================================

SYSTEM = f"""\
You are a robot agent controlling a Unitree G1 humanoid robot in a kitchen.

Tools: look, move, grasp, todo, perceive, load_skill.

WORKFLOW:
1. perceive() to understand the scene
2. todo() to decompose the task
3. load_skill() to get the right manipulation recipe for each subtask
4. Execute with look/move/grasp following the loaded skill's procedure
5. perceive() to verify the result

IMPORTANT: Before picking up an object, load the appropriate skill first.
Different objects need different grasp strategies.

Manipulation skills available:
{SKILL_LOADER.get_descriptions()}"""


SUBAGENT_SYSTEM = """\
You are a perception specialist for a Unitree G1 robot.
Analyze the scene thoroughly using multiple look() calls.
Report: objects, positions, spatial relationships, reachability, obstacles."""


# ============================================================
# TodoManager (from r03)
# ============================================================

class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
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
# Mock Environment (from r04)
# ============================================================

GRASP_DISTANCE = 0.08


class MockRobotEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.objects = {
            "red apple":   {"pos": [0.45,  0.12, 0.82], "on": "counter", "shape": "round-flat"},
            "white plate": {"pos": [0.45, -0.15, 0.80], "on": "counter", "shape": "flat"},
            "blue mug":    {"pos": [0.50,  0.30, 0.82], "on": "counter", "shape": "tall-handle"},
            "fork":        {"pos": [0.55, -0.20, 0.81], "on": "counter", "shape": "flat-long"},
            "napkin":      {"pos": [0.60,  0.00, 0.81], "on": "counter", "shape": "flat"},
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
            lines.append(f"  - {name}: pos={obj['pos']}, {loc}, shape={obj['shape']}")
        lines.append(f"Robot: ee={self.ee_pos}, gripper={self.gripper}, holding={self.holding or 'nothing'}")

        if question:
            q = question.lower()
            matched = self._find_object(q)
            if matched:
                obj = self.objects[matched]
                d = self._distance(self.ee_pos, obj["pos"])
                loc = "held by gripper" if matched == self.holding else f"on {obj['on']}"
                lines.append(f"=> {matched}: {loc}, {d:.2f}m away, shape={obj['shape']}.")
                if obj["shape"] in ("tall-handle",):
                    lines.append("   Recommended: side-grasp (has handle)")
                elif obj["shape"] in ("round-flat", "flat", "flat-long"):
                    lines.append("   Recommended: top-grasp (flat/low object)")
            elif any(w in q for w in ("reach", "distance")):
                lines.append("=> Reachability:")
                for name, obj in self.objects.items():
                    d = self._distance(self.ee_pos, obj["pos"])
                    lines.append(f"   {name}: {d:.2f}m — {'reachable' if d < 0.40 else 'too far'}")
            elif any(w in q for w in ("success", "done", "plate")):
                on_plate = [n for n, o in self.objects.items()
                            if o["on"] == "white plate" and n != "white plate"]
                lines.append(f"=> On plate: {on_plate}" if on_plate else "=> Plate is empty.")
            elif any(w in q for w in ("hold", "grip")):
                lines.append(f"=> Gripper: {self.gripper}, holding: {self.holding or 'nothing'}.")
            else:
                lines.append("=> All objects on counter, within workspace.")
        return "\n".join(lines)

    def move(self, target: str = "", position: list = None) -> str:
        if target:
            matched = self._find_object(target)
            if not matched:
                return f"Error: unknown '{target}'. Known: {list(self.objects.keys())}"
            dest = list(self.objects[matched]["pos"])
        elif position and len(position) >= 3:
            dest = position[:3]
        else:
            return "Error: provide 'target' or 'position'."
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
                            return f"Released '{released}' onto '{name}'."
                self.objects[released]["on"] = "counter"
                self.holding = None
                return f"Released '{released}' onto counter."
            return "Gripper opened. Nothing held."
        return f"Error: action must be 'open' or 'close'."


mock_env = MockRobotEnv()


# ============================================================
# Real mode handlers
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
            return f"Done after {i+1} steps (success={step_result.get('success')})."
        obs_resp = requests.get(f"{SIM_URL}/raw_observation", timeout=5)
        obs = obs_resp.json()
    return f"Executed {steps} steps. Reward: {step_result.get('total_reward', 0):.3f}"


# ============================================================
# Tool handlers
# ============================================================

def run_look(question=""):
    if SIM_URL and VLM_URL:
        try:
            return real_look(question)
        except Exception as e:
            return f"[Error: {e}]\n{mock_env.look(question)}"
    return mock_env.look(question)

def run_move(target="", position=None):
    if SIM_URL and VLA_URL:
        try:
            instr = f"move toward {target}" if target else f"move to {position}"
            return real_act(instr)
        except Exception as e:
            return f"[Error: {e}]\n{mock_env.move(target, position)}"
    return mock_env.move(target, position)

def run_grasp(action):
    if SIM_URL and VLA_URL:
        try:
            return real_act(f"{'close' if action == 'close' else 'open'} the gripper")
        except Exception as e:
            return f"[Error: {e}]\n{mock_env.grasp(action)}"
    return mock_env.grasp(action)


# ============================================================
# Perception subagent (from r04)
# ============================================================

CHILD_TOOLS = [{
    "name": "look",
    "description": "Observe the scene. Ask specific questions.",
    "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []},
}]

def run_perceive(goal: str) -> str:
    prompt = f"Analyze the environment. Goal: {goal}\n\nUse look() multiple times, then summarize."
    sub_messages = [{"role": "user", "content": prompt}]
    print(f"\033[34m  [subagent] spawned — {goal}\033[0m")
    for turn in range(15):
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=4096,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                q = block.input.get("question", "")
                output = run_look(q)
                print(f"\033[34m  [subagent] look: {q or '(overview)'}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        sub_messages.append({"role": "user", "content": results})
    summary = "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"
    print(f"\033[34m  [subagent] done ({turn + 1} turns)\033[0m")
    return summary


# ============================================================
# Dispatch map (r04 tools + load_skill)
# ============================================================

TOOL_HANDLERS = {
    "look":       lambda **kw: run_look(kw.get("question", "")),
    "move":       lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp":      lambda **kw: run_grasp(kw["action"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
    "perceive":   lambda **kw: run_perceive(kw["goal"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}

TOOLS = [
    {"name": "look", "description": "Quick observation.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "move", "description": "Move end-effector to target object or [x,y,z].",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "Control gripper: 'close' to grab, 'open' to release.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "todo", "description": "Update task plan. Break goals into subtasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
    {"name": "perceive", "description": "Spawn perception specialist for thorough scene analysis.",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
    {"name": "load_skill", "description": "Load a manipulation skill by name. Use before executing a grasp or placement.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name, e.g. 'top-grasp', 'side-grasp', 'pour', 'precision-place'."}}, "required": ["name"]}},
]


# ============================================================
# Agent loop (with nag reminder)
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
                # Print
                if block.name == "todo":
                    print(f"\033[35m[todo]\n{output}\033[0m")
                    used_todo = True
                elif block.name == "load_skill":
                    print(f"\033[36m[load_skill] {block.input.get('name', '?')}\033[0m")
                    print(f"\033[90m{str(output)[:400]}...\033[0m")
                elif block.name == "perceive":
                    print(f"\033[34m[perceive] {block.input.get('goal', '')}\033[0m")
                    print(f"\033[90m{str(output)[:800]}\033[0m")
                else:
                    print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.insert(0, {"type": "text", "text": "<reminder>Update your todo list.</reminder>"})
        messages.append({"role": "user", "content": results})


# ============================================================
# REPL
# ============================================================

if __name__ == "__main__":
    mode = "REAL" if (SIM_URL and VLA_URL) else "MOCK"
    print(f"\033[32m[r05] Robot Skills  |  {mode}\033[0m")
    print(f"\033[90mTools: look, move, grasp, todo, perceive, load_skill\033[0m")
    print(f"\033[90mSkills: {', '.join(SKILL_LOADER.skills.keys()) or 'none'}\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip().lower() == "reset":
            mock_env.reset()
            TODO.items = []
            history = []
            print("Reset.\n")
            continue
        if query.strip().lower() == "todos":
            print(TODO.render())
            print()
            continue
        if query.strip().lower() == "skills":
            print(SKILL_LOADER.get_descriptions())
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
