#!/usr/bin/env python3
"""
r07_persistent_tasks.py - Persistent Task Board

Tasks persist as JSON files in .tasks/ — they survive context compression
and even agent restarts. Each task has a dependency graph.

    .tasks/
      task_1.json  {"id":1, "subject":"perceive scene", "status":"completed"}
      task_2.json  {"id":2, "subject":"pick apple", "blockedBy":[1], "status":"pending"}
      task_3.json  {"id":3, "subject":"place on plate", "blockedBy":[2]}

    Dependency resolution:
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- completing task 1 removes it from task 2's blockedBy

Key insight: "State outside the conversation survives compression."

r03 had TodoManager (in-memory) — lost on compression.
r07 has TaskManager (on-disk) — permanent until explicitly completed.
After auto_compact, agent calls task_list to reload exact progress.
"""

import os
import re
import math
import json
import time
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

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
TASKS_DIR = Path(__file__).resolve().parent.parent / ".tasks"
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / ".transcripts"

THRESHOLD = 50000
KEEP_RECENT = 3


# ============================================================
# TaskManager — CRUD with dependency graph, persisted as JSON
# ============================================================

class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id, "subject": subject, "description": description,
            "status": "pending", "blockedBy": [], "blocks": [], "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass
        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        done = sum(1 for t in tasks if t["status"] == "completed")
        lines.append(f"\n({done}/{len(tasks)} completed)")
        return "\n".join(lines)

    def clear_all(self):
        for f in self.dir.glob("task_*.json"):
            f.unlink()
        self._next_id = 1


TASKS = TaskManager(TASKS_DIR)


# ============================================================
# SkillLoader (from r05)
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
            self.skills[name] = {"meta": meta, "body": body}

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
        if not self.skills:
            return "(no skills)"
        return "\n".join(f"  - {n}: {s['meta'].get('description', '')}" for n, s in self.skills.items())

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)


# ============================================================
# Compression (from r06)
# ============================================================

def estimate_tokens(messages: list) -> int:
    return len(str(messages)) // 4

def micro_compact(messages: list) -> list:
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    for _, _, result in tool_results[:-KEEP_RECENT]:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
    return messages

def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"\033[90m[transcript saved: {path.name}]\033[0m")

    robot_state = f"ee={mock_env.ee_pos}, gripper={mock_env.gripper}, holding={mock_env.holding or 'nothing'}"
    task_state = TASKS.list_all()
    conversation_text = json.dumps(messages, default=str)[:80000]

    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this robot agent conversation. Include:\n"
            "1) Original user task\n"
            "2) Key decisions and strategies used\n"
            "3) Any failures or retries\n"
            "Be concise.\n\n"
            f"Robot: {robot_state}\n"
            f"Tasks:\n{task_state}\n\n"
            f"Conversation:\n{conversation_text}"}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    return [
        {"role": "user", "content":
            f"[Context compressed. Transcript: {path.name}]\n\n{summary}\n\n"
            f"Robot: {robot_state}\n\n"
            f"IMPORTANT: Call task_list to see current progress. Tasks persist on disk."},
        {"role": "assistant", "content":
            "Understood. Let me check the task board for current progress."},
    ]


# ============================================================
# Mock Environment (from r05)
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

    def _find_object(self, q: str):
        q = q.lower()
        for name in self.objects:
            if q in name or name in q:
                return name
        return None

    def look(self, question=""):
        lines = ["Scene:"]
        for name, obj in self.objects.items():
            loc = "held" if name == self.holding else f"on {obj['on']}"
            lines.append(f"  - {name}: {obj['pos']}, {loc}, {obj['shape']}")
        lines.append(f"Robot: ee={self.ee_pos}, gripper={self.gripper}, holding={self.holding or 'nothing'}")
        if question:
            q = question.lower()
            m = self._find_object(q)
            if m:
                obj = self.objects[m]
                d = self._distance(self.ee_pos, obj["pos"])
                lines.append(f"=> {m}: {'held' if m == self.holding else f'on {obj[\"on\"]}'}, {d:.2f}m, {obj['shape']}")
            elif any(w in q for w in ("success", "done", "plate")):
                on_plate = [n for n, o in self.objects.items() if o["on"] == "white plate" and n != "white plate"]
                lines.append(f"=> On plate: {on_plate}" if on_plate else "=> Plate empty.")
            elif any(w in q for w in ("hold", "grip")):
                lines.append(f"=> Gripper: {self.gripper}, holding: {self.holding or 'nothing'}")
        return "\n".join(lines)

    def move(self, target="", position=None):
        if target:
            m = self._find_object(target)
            if not m:
                return f"Error: unknown '{target}'. Known: {list(self.objects.keys())}"
            dest = list(self.objects[m]["pos"])
        elif position and len(position) >= 3:
            dest = position[:3]
        else:
            return "Error: provide 'target' or 'position'."
        old = list(self.ee_pos)
        self.ee_pos = dest
        d = self._distance(old, dest)
        if self.holding and self.holding in self.objects:
            self.objects[self.holding]["pos"] = list(dest)
        r = f"Moved: {old} -> {dest} ({d:.2f}m)"
        if target: r += f", near '{target}'"
        if self.holding: r += f". Carrying {self.holding}."
        return r

    def grasp(self, action):
        if action == "close":
            if self.gripper == "close": return "Already closed."
            self.gripper = "close"
            for name, obj in self.objects.items():
                d = self._distance(self.ee_pos, obj["pos"])
                if d < GRASP_DISTANCE and name != self.holding:
                    self.holding = name
                    return f"Grasped '{name}' ({d:.3f}m)."
            return "Nothing in range."
        elif action == "open":
            if self.gripper == "open": return "Already open."
            self.gripper = "open"
            if self.holding:
                released = self.holding
                for name, obj in self.objects.items():
                    if name != released and self._distance(self.ee_pos, obj["pos"]) < 0.10:
                        self.objects[released]["on"] = name
                        self.holding = None
                        return f"Released '{released}' onto '{name}'."
                self.objects[released]["on"] = "counter"
                self.holding = None
                return f"Released '{released}' onto counter."
            return "Nothing held."
        return f"Error: 'open' or 'close', got '{action}'."


mock_env = MockRobotEnv()


# ============================================================
# Real mode + tool handlers
# ============================================================

def real_look(question=""):
    resp = requests.get(f"{SIM_URL}/render", timeout=5)
    img_b64 = base64.b64encode(resp.content).decode()
    prompt = question if question else "Describe the scene."
    r = requests.post(f"{VLM_URL}/analyze", json={"image": img_b64, "prompt": prompt}, timeout=30).json()
    return f"{r.get('analysis', '')} [{r.get('inference_ms', 0):.0f}ms]"

def real_act(instruction, steps=10):
    obs = requests.get(f"{SIM_URL}/raw_observation", timeout=5).json()
    sr = {}
    for i in range(steps):
        a = requests.post(f"{VLA_URL}/predict", json={"observation": obs, "instruction": instruction}, timeout=30).json().get("action", {})
        sr = requests.post(f"{SIM_URL}/step", json={"action": a}, timeout=5).json()
        if sr.get("terminated") or sr.get("success"): return f"Done after {i+1} steps."
        obs = requests.get(f"{SIM_URL}/raw_observation", timeout=5).json()
    return f"Executed {steps} steps."

def run_look(q=""):
    if SIM_URL and VLM_URL:
        try: return real_look(q)
        except Exception as e: return f"[Error: {e}]\n{mock_env.look(q)}"
    return mock_env.look(q)

def run_move(target="", position=None):
    if SIM_URL and VLA_URL:
        try: return real_act(f"move toward {target}" if target else f"move to {position}")
        except Exception as e: return f"[Error: {e}]\n{mock_env.move(target, position)}"
    return mock_env.move(target, position)

def run_grasp(action):
    if SIM_URL and VLA_URL:
        try: return real_act(f"{'close' if action == 'close' else 'open'} the gripper")
        except Exception as e: return f"[Error: {e}]\n{mock_env.grasp(action)}"
    return mock_env.grasp(action)


# Perception subagent
SUBAGENT_SYSTEM = "You are a perception specialist. Multiple look() calls, then summarize."
CHILD_TOOLS = [{"name": "look", "description": "Observe scene.", "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}}]

def run_perceive(goal):
    sub = [{"role": "user", "content": f"Analyze: {goal}\nUse look() multiple times, then summarize."}]
    print(f"\033[34m  [subagent] {goal}\033[0m")
    for turn in range(15):
        resp = client.messages.create(model=MODEL, system=SUBAGENT_SYSTEM, messages=sub, tools=CHILD_TOOLS, max_tokens=4096)
        sub.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use": break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                out = run_look(b.input.get("question", ""))
                print(f"\033[34m  [subagent] look: {b.input.get('question', '(overview)')}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        sub.append({"role": "user", "content": results})
    return "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"


# ============================================================
# System prompt
# ============================================================

SYSTEM = f"""\
You are a robot agent controlling a Unitree G1 humanoid robot.

Tools: look, move, grasp, perceive, load_skill, compact,
       task_create, task_update, task_list, task_get.

WORKFLOW:
1. perceive() the scene
2. task_create() for each subtask (set dependencies with addBlocks)
3. task_update(id, status="in_progress") before starting each
4. Execute with look/move/grasp + load_skill
5. task_update(id, status="completed") when done
6. task_list() to check what's next

Tasks persist on disk — after context compression, call task_list to reload.

Skills: {SKILL_LOADER.get_descriptions()}"""


# ============================================================
# Dispatch map
# ============================================================

TOOL_HANDLERS = {
    "look":        lambda **kw: run_look(kw.get("question", "")),
    "move":        lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp":       lambda **kw: run_grasp(kw["action"]),
    "perceive":    lambda **kw: run_perceive(kw["goal"]),
    "load_skill":  lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":     lambda **kw: "Compressing...",
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
}

TOOLS = [
    {"name": "look", "description": "Quick observation.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "move", "description": "Move end-effector to target or [x,y,z].",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "'close' to grab, 'open' to release.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "perceive", "description": "Thorough scene analysis via subagent.",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
    {"name": "load_skill", "description": "Load a manipulation skill.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compact", "description": "Compress conversation context.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update task status or dependencies.",
     "input_schema": {"type": "object", "properties": {
         "task_id": {"type": "integer"},
         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
         "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
         "addBlocks": {"type": "array", "items": {"type": "integer"}},
     }, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status and dependencies.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full task details by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


# ============================================================
# Agent loop — compression + persistent tasks
# ============================================================

def agent_loop(messages: list):
    while True:
        micro_compact(messages)
        est = estimate_tokens(messages)
        if est > THRESHOLD:
            print(f"\033[91m[auto_compact] ~{est // 4} tokens\033[0m")
            messages[:] = auto_compact(messages)

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                # Print by category
                if block.name in ("task_create", "task_update", "task_list", "task_get"):
                    print(f"\033[35m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                elif block.name == "load_skill":
                    print(f"\033[36m[load_skill] {block.input.get('name', '?')}\033[0m")
                elif block.name == "compact":
                    print(f"\033[91m[compact]\033[0m")
                elif block.name == "perceive":
                    print(f"\033[34m[perceive] {block.input.get('goal', '')}\033[0m")
                    print(f"\033[90m{str(output)[:600]}\033[0m")
                else:
                    print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        messages.append({"role": "user", "content": results})

        if manual_compact:
            print(f"\033[91m[manual compact]\033[0m")
            messages[:] = auto_compact(messages)


# ============================================================
# REPL
# ============================================================

if __name__ == "__main__":
    mode = "REAL" if (SIM_URL and VLA_URL) else "MOCK"
    print(f"\033[32m[r07] Persistent Task Board  |  {mode}\033[0m")
    print(f"\033[90mTools: look, move, grasp, perceive, load_skill, compact, task_*\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = query.strip().lower()
        if cmd in ("q", "exit", ""):
            break
        if cmd == "reset":
            mock_env.reset()
            TASKS.clear_all()
            history = []
            print("Reset.\n")
            continue
        if cmd == "tasks":
            print(TASKS.list_all())
            print()
            continue
        if cmd == "skills":
            print(SKILL_LOADER.get_descriptions())
            print()
            continue
        if cmd == "/compact":
            history[:] = auto_compact(history)
            print("Compressed.\n")
            continue
        if cmd == "tokens":
            est = estimate_tokens(history)
            print(f"~{est // 4} tokens, {len(history)} messages\n")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"):
                    print(block.text)
        print()
