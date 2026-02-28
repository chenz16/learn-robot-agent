#!/usr/bin/env python3
"""
r06_context_compact.py - Context Compression

Three-layer compression so the robot agent can handle long-horizon tasks
without running out of context window.

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]          (silent, every turn)
      Replace old tool_results (except last 3)
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > THRESHOLD?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  LLM summarizes: task progress + robot state.
                  Replace all messages with summary.
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.

Key insight: "A robot that forgets strategically can work forever."

Why this matters for robotics:
- "Set the table" = 30+ tool calls (look, move, grasp each add tokens)
- Each look() returns scene descriptions (~100 tokens)
- Skills loaded add ~300 tokens each
- Without compression, context fills up after ~15 subtasks
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
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / ".transcripts"

# Compression settings
THRESHOLD = 50000      # chars (~12500 tokens) triggers auto_compact
KEEP_RECENT = 3        # keep last N tool results intact in micro_compact


# ============================================================
# Token estimation
# ============================================================

def estimate_tokens(messages: list) -> int:
    return len(str(messages)) // 4


# ============================================================
# Layer 1: micro_compact — replace old tool results with placeholders
# ============================================================

def micro_compact(messages: list) -> list:
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Build tool_use_id -> tool_name map
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # Replace old results
    for _, _, result in tool_results[:-KEEP_RECENT]:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
    return messages


# ============================================================
# Layer 2: auto_compact — save transcript, summarize, replace
# ============================================================

def auto_compact(messages: list) -> list:
    # Save full transcript
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"\033[90m[transcript saved: {transcript_path.name}]\033[0m")

    # Summarize with robot-specific context
    conversation_text = json.dumps(messages, default=str)[:80000]

    # Include current robot state and todo state
    robot_state = f"Robot: ee={mock_env.ee_pos}, gripper={mock_env.gripper}, holding={mock_env.holding or 'nothing'}"
    todo_state = TODO.render()

    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this robot agent conversation for continuity. Include:\n"
            "1) Original user task\n"
            "2) What subtasks have been completed\n"
            "3) What subtasks remain\n"
            "4) Current robot state\n"
            "5) Any loaded skills or strategies in use\n"
            "Be concise but preserve all critical details.\n\n"
            f"Current state: {robot_state}\n"
            f"Todo list:\n{todo_state}\n\n"
            f"Conversation:\n{conversation_text}"}],
        max_tokens=2000,
    )
    summary = response.content[0].text

    return [
        {"role": "user", "content":
            f"[Context compressed. Transcript: {transcript_path.name}]\n\n"
            f"{summary}\n\n"
            f"Current state: {robot_state}\n"
            f"Todos:\n{todo_state}"},
        {"role": "assistant", "content":
            "Understood. I have the context from the summary and current robot state. Continuing with the remaining tasks."},
    ]


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
            elif any(w in q for w in ("success", "done", "plate")):
                on_plate = [n for n, o in self.objects.items()
                            if o["on"] == "white plate" and n != "white plate"]
                lines.append(f"=> On plate: {on_plate}" if on_plate else "=> Plate empty.")
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
            return "Gripper closed. Nothing in range."
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
# Real mode + tool handlers (from r05)
# ============================================================

def real_look(question=""):
    resp = requests.get(f"{SIM_URL}/render", timeout=5)
    img_b64 = base64.b64encode(resp.content).decode()
    prompt = question if question else "Describe the scene."
    vlm_resp = requests.post(f"{VLM_URL}/analyze", json={"image": img_b64, "prompt": prompt}, timeout=30)
    result = vlm_resp.json()
    return f"{result.get('analysis', '')} [{result.get('inference_ms', 0):.0f}ms]"

def real_act(instruction, steps=10):
    obs = requests.get(f"{SIM_URL}/raw_observation", timeout=5).json()
    step_result = {}
    for i in range(steps):
        action = requests.post(f"{VLA_URL}/predict", json={"observation": obs, "instruction": instruction}, timeout=30).json().get("action", {})
        step_result = requests.post(f"{SIM_URL}/step", json={"action": action}, timeout=5).json()
        if step_result.get("terminated") or step_result.get("success"):
            return f"Done after {i+1} steps."
        obs = requests.get(f"{SIM_URL}/raw_observation", timeout=5).json()
    return f"Executed {steps} steps."

def run_look(question=""):
    if SIM_URL and VLM_URL:
        try: return real_look(question)
        except Exception as e: return f"[Error: {e}]\n{mock_env.look(question)}"
    return mock_env.look(question)

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


# ============================================================
# Perception subagent (from r04)
# ============================================================

SUBAGENT_SYSTEM = "You are a perception specialist. Analyze the scene with multiple look() calls, then summarize."
CHILD_TOOLS = [{"name": "look", "description": "Observe the scene.", "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}}]

def run_perceive(goal):
    sub_messages = [{"role": "user", "content": f"Analyze: {goal}\nUse look() multiple times, then summarize."}]
    print(f"\033[34m  [subagent] spawned — {goal}\033[0m")
    for turn in range(15):
        response = client.messages.create(model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages, tools=CHILD_TOOLS, max_tokens=4096)
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_look(block.input.get("question", ""))
                print(f"\033[34m  [subagent] look: {block.input.get('question', '(overview)')}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        sub_messages.append({"role": "user", "content": results})
    summary = "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"
    print(f"\033[34m  [subagent] done ({turn + 1} turns)\033[0m")
    return summary


# ============================================================
# System prompt
# ============================================================

SYSTEM = f"""\
You are a robot agent controlling a Unitree G1 humanoid robot in a kitchen.

Tools: look, move, grasp, todo, perceive, load_skill, compact.

WORKFLOW:
1. perceive() the scene
2. todo() to plan subtasks
3. load_skill() for the right manipulation recipe
4. Execute with look/move/grasp
5. perceive() to verify

Use 'compact' when you feel the conversation is getting long and you want to
compress context to continue working efficiently.

Skills: {SKILL_LOADER.get_descriptions()}"""


# ============================================================
# Dispatch map (r05 tools + compact)
# ============================================================

TOOL_HANDLERS = {
    "look":       lambda **kw: run_look(kw.get("question", "")),
    "move":       lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp":      lambda **kw: run_grasp(kw["action"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
    "perceive":   lambda **kw: run_perceive(kw["goal"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":    lambda **kw: "Compressing...",
}

TOOLS = [
    {"name": "look", "description": "Quick observation.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "move", "description": "Move end-effector to target object or [x,y,z].",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "'close' to grab, 'open' to release.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "todo", "description": "Plan subtasks. Track progress.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
    {"name": "perceive", "description": "Spawn perception specialist.",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
    {"name": "load_skill", "description": "Load a manipulation skill.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compact", "description": "Compress conversation context. Use when context is getting long.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


# ============================================================
# Agent loop — three-layer compression integrated
# ============================================================

def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        # Layer 1: micro_compact (silent, every turn)
        micro_compact(messages)

        # Layer 2: auto_compact (if tokens exceed threshold)
        est = estimate_tokens(messages)
        if est > THRESHOLD:
            print(f"\033[91m[auto_compact] ~{est // 4} tokens > threshold\033[0m")
            messages[:] = auto_compact(messages)

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        used_todo = False
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
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
                elif block.name == "compact":
                    print(f"\033[91m[compact] manual\033[0m")
                elif block.name == "perceive":
                    print(f"\033[34m[perceive] {block.input.get('goal', '')}\033[0m")
                    print(f"\033[90m{str(output)[:600]}\033[0m")
                else:
                    print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.insert(0, {"type": "text", "text": "<reminder>Update your todo list.</reminder>"})

        messages.append({"role": "user", "content": results})

        # Layer 3: manual compact (model called the compact tool)
        if manual_compact:
            print(f"\033[91m[manual compact triggered]\033[0m")
            messages[:] = auto_compact(messages)


# ============================================================
# REPL
# ============================================================

if __name__ == "__main__":
    mode = "REAL" if (SIM_URL and VLA_URL) else "MOCK"
    print(f"\033[32m[r06] Context Compression  |  {mode}\033[0m")
    print(f"\033[90mTools: look, move, grasp, todo, perceive, load_skill, compact\033[0m")
    print(f"\033[90mThreshold: ~{THRESHOLD // 4} tokens  |  /compact to force compression\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr06 >> \033[0m")
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
        if query.strip().lower() == "/compact":
            history[:] = auto_compact(history)
            print("Compressed.\n")
            continue
        if query.strip().lower() == "tokens":
            est = estimate_tokens(history)
            print(f"~{est // 4} tokens ({est} chars), {len(history)} messages\n")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"):
                    print(block.text)
        print()
