#!/usr/bin/env python3
"""
r09_multi_robot_teams.py - Multi-Robot Teams

Persistent named robot agents with file-based JSONL inboxes. Each teammate
runs its own agent loop in a separate thread with robot tools.

    Subagent (r04):  spawn -> execute -> return summary -> destroyed
    Teammate (r09):  spawn -> work -> idle -> work -> ... -> shutdown

    .team/config.json                   .team/inbox/
    +----------------------------+      +------------------+
    | {"team_name": "robot-team",|      | scout.jsonl      |
    |  "members": [              |      | manipulator.jsonl|
    |    {"name":"scout",        |      | lead.jsonl       |
    |     "role":"perception",   |      +------------------+
    |     "status":"idle"}       |
    |  ]}                        |      send_message("scout", "check area"):
    +----------------------------+        open("scout.jsonl", "a").write(msg)

    Thread: scout            Thread: manipulator
    +------------------+     +------------------+
    | agent_loop       |     | agent_loop       |
    | tools: look,     |     | tools: look,     |
    |   move, grasp,   |     |   move, grasp,   |
    |   send_message   |     |   send_message   |
    | status: working  |     | status: idle     |
    +------------------+     +------------------+

Key insight: "Multiple robots, each with its own agent loop, coordinating via messages."
"""

import os
import re
import math
import json
import time
import base64
import threading
import uuid
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
TEAM_DIR = Path(__file__).resolve().parent.parent / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

THRESHOLD = 50000
KEEP_RECENT = 3

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request", "shutdown_response"}


# ============================================================
# MessageBus — JSONL inbox per robot
# ============================================================

class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with self._lock:
            inbox_path = self.dir / f"{to}.jsonl"
            with open(inbox_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        with self._lock:
            inbox_path = self.dir / f"{name}.jsonl"
            if not inbox_path.exists():
                return []
            messages = []
            for line in inbox_path.read_text().strip().splitlines():
                if line:
                    messages.append(json.loads(line))
            inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# ============================================================
# BackgroundManager — threaded execution + notification queue
# ============================================================

class BackgroundManager:
    def __init__(self):
        self.jobs = {}
        self._queue = []
        self._lock = threading.Lock()

    def run(self, job_id: str, func, args: tuple, description: str) -> str:
        self.jobs[job_id] = {"status": "running", "description": description, "result": None}
        thread = threading.Thread(target=self._execute, args=(job_id, func, args), daemon=True)
        thread.start()
        return f"Background job {job_id} started: {description}"

    def _execute(self, job_id: str, func, args: tuple):
        try:
            result = func(*args)
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["result"] = result
        except Exception as e:
            result = f"Error: {e}"
            self.jobs[job_id]["status"] = "error"
            self.jobs[job_id]["result"] = result
        with self._lock:
            self._queue.append({
                "job_id": job_id,
                "status": self.jobs[job_id]["status"],
                "description": self.jobs[job_id]["description"],
                "result": str(result)[:800],
            })

    def check(self, job_id: str = None) -> str:
        if job_id:
            j = self.jobs.get(job_id)
            if not j:
                return f"Unknown job: {job_id}"
            return f"[{j['status']}] {j['description']}\n{j.get('result') or '(running)'}"
        lines = [f"  {jid}: [{j['status']}] {j['description']}" for jid, j in self.jobs.items()]
        return "\n".join(lines) if lines else "No background jobs."

    def drain(self) -> list:
        with self._lock:
            notifs = list(self._queue)
            self._queue.clear()
        return notifs


BG = BackgroundManager()


# ============================================================
# TaskManager (from r07)
# ============================================================

class TaskManager:
    def __init__(self, d):
        self.dir = d; d.mkdir(exist_ok=True); self._next_id = self._max_id() + 1
    def _max_id(self):
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0
    def _load(self, tid):
        p = self.dir / f"task_{tid}.json"
        if not p.exists(): raise ValueError(f"Task {tid} not found")
        return json.loads(p.read_text())
    def _save(self, t): (self.dir / f"task_{t['id']}.json").write_text(json.dumps(t, indent=2))
    def create(self, subject, description=""):
        t = {"id": self._next_id, "subject": subject, "description": description, "status": "pending", "blockedBy": [], "blocks": [], "owner": ""}
        self._save(t); self._next_id += 1; return json.dumps(t, indent=2)
    def get(self, tid): return json.dumps(self._load(tid), indent=2)
    def update(self, tid, status=None, add_blocked_by=None, add_blocks=None):
        t = self._load(tid)
        if status:
            if status not in ("pending", "in_progress", "completed"): raise ValueError(f"Bad status: {status}")
            t["status"] = status
            if status == "completed": self._clear_dep(tid)
        if add_blocked_by: t["blockedBy"] = list(set(t["blockedBy"] + add_blocked_by))
        if add_blocks:
            t["blocks"] = list(set(t["blocks"] + add_blocks))
            for bid in add_blocks:
                try:
                    b = self._load(bid)
                    if tid not in b["blockedBy"]: b["blockedBy"].append(tid); self._save(b)
                except ValueError: pass
        self._save(t); return json.dumps(t, indent=2)
    def _clear_dep(self, cid):
        for f in self.dir.glob("task_*.json"):
            t = json.loads(f.read_text())
            if cid in t.get("blockedBy", []): t["blockedBy"].remove(cid); self._save(t)
    def list_all(self):
        tasks = [json.loads(f.read_text()) for f in sorted(self.dir.glob("task_*.json"))]
        if not tasks: return "No tasks."
        lines = []
        for t in tasks:
            m = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            b = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{m} #{t['id']}: {t['subject']}{b}")
        done = sum(1 for t in tasks if t["status"] == "completed")
        lines.append(f"\n({done}/{len(tasks)} completed)"); return "\n".join(lines)
    def clear_all(self):
        for f in self.dir.glob("task_*.json"): f.unlink()
        self._next_id = 1

TASKS = TaskManager(TASKS_DIR)


# ============================================================
# SkillLoader (from r05)
# ============================================================

class SkillLoader:
    def __init__(self, d):
        self.skills = {}
        if not d.exists(): return
        for f in sorted(d.rglob("SKILL.md")):
            text = f.read_text()
            m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
            if not m: self.skills[f.parent.name] = {"meta": {}, "body": text}; continue
            meta = {}
            for line in m.group(1).strip().splitlines():
                if ":" in line: k, v = line.split(":", 1); meta[k.strip()] = v.strip()
            self.skills[meta.get("name", f.parent.name)] = {"meta": meta, "body": m.group(2).strip()}
    def get_descriptions(self):
        if not self.skills: return "(no skills)"
        return "\n".join(f"  - {n}: {s['meta'].get('description', '')}" for n, s in self.skills.items())
    def get_content(self, name):
        s = self.skills.get(name)
        if not s: return f"Error: Unknown '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{s['body']}\n</skill>"

SKILL_LOADER = SkillLoader(SKILLS_DIR)


# ============================================================
# Compression (from r06)
# ============================================================

def estimate_tokens(msgs): return len(str(msgs)) // 4

def micro_compact(msgs):
    trs = []
    for mi, m in enumerate(msgs):
        if m["role"] == "user" and isinstance(m.get("content"), list):
            for pi, p in enumerate(m["content"]):
                if isinstance(p, dict) and p.get("type") == "tool_result": trs.append((mi, pi, p))
    if len(trs) <= KEEP_RECENT: return msgs
    nm = {}
    for m in msgs:
        if m["role"] == "assistant" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if hasattr(b, "type") and b.type == "tool_use": nm[b.id] = b.name
    for _, _, r in trs[:-KEEP_RECENT]:
        if isinstance(r.get("content"), str) and len(r["content"]) > 100:
            r["content"] = f"[Previous: used {nm.get(r.get('tool_use_id', ''), 'unknown')}]"
    return msgs

def auto_compact(msgs):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    p = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(p, "w") as f:
        for m in msgs: f.write(json.dumps(m, default=str) + "\n")
    rs = f"ee={mock_env.ee_pos}, gripper={mock_env.gripper}, holding={mock_env.holding or 'nothing'}"
    ts = TASKS.list_all()
    conv = json.dumps(msgs, default=str)[:80000]
    resp = client.messages.create(model=MODEL, messages=[{"role": "user", "content":
        f"Summarize robot conversation. Include: original task, progress, strategies.\n\nRobot: {rs}\nTasks:\n{ts}\n\nConversation:\n{conv}"}], max_tokens=2000)
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {p.name}]\n\n{resp.content[0].text}\n\nRobot: {rs}\n\nCall task_list for current progress."},
        {"role": "assistant", "content": "Understood. Checking task board."},
    ]


# ============================================================
# Mock Environment — thread-safe (from r05, lock added for r09)
# ============================================================

GRASP_DISTANCE = 0.08

class MockRobotEnv:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()
    def reset(self):
        with self._lock:
            self.objects = {
                "red apple": {"pos": [0.45, 0.12, 0.82], "on": "counter", "shape": "round-flat"},
                "white plate": {"pos": [0.45, -0.15, 0.80], "on": "counter", "shape": "flat"},
                "blue mug": {"pos": [0.50, 0.30, 0.82], "on": "counter", "shape": "tall-handle"},
                "fork": {"pos": [0.55, -0.20, 0.81], "on": "counter", "shape": "flat-long"},
                "napkin": {"pos": [0.60, 0.00, 0.81], "on": "counter", "shape": "flat"},
            }
            self.ee_pos = [0.30, 0.0, 0.95]; self.gripper = "open"; self.holding = None
    def _dist(self, a, b): return math.sqrt(sum((ai-bi)**2 for ai, bi in zip(a, b)))
    def _find(self, q):
        q = q.lower()
        for n in self.objects:
            if q in n or n in q: return n
        return None
    def look(self, question=""):
        with self._lock:
            lines = ["Scene:"]
            for n, o in self.objects.items():
                loc = "held" if n == self.holding else f"on {o['on']}"
                lines.append(f"  - {n}: {o['pos']}, {loc}, {o['shape']}")
            lines.append(f"Robot: ee={self.ee_pos}, gripper={self.gripper}, holding={self.holding or 'nothing'}")
            if question:
                q = question.lower(); m = self._find(q)
                if m:
                    o = self.objects[m]; d = self._dist(self.ee_pos, o["pos"])
                    on_what = "held" if m == self.holding else f"on {o['on']}"
                    lines.append(f"=> {m}: {on_what}, {d:.2f}m, {o['shape']}")
                elif any(w in q for w in ("success", "done", "plate")):
                    on_p = [n for n, o in self.objects.items() if o["on"] == "white plate" and n != "white plate"]
                    lines.append(f"=> On plate: {on_p}" if on_p else "=> Plate empty.")
                elif any(w in q for w in ("hold", "grip")):
                    lines.append(f"=> Gripper: {self.gripper}, holding: {self.holding or 'nothing'}")
                elif any(w in q for w in ("change", "moved", "different")):
                    lines.append("=> No changes detected since last observation.")
            return "\n".join(lines)
    def move(self, target="", position=None):
        with self._lock:
            if target:
                m = self._find(target)
                if not m: return f"Error: unknown '{target}'."
                dest = list(self.objects[m]["pos"])
            elif position and len(position) >= 3: dest = position[:3]
            else: return "Error: provide 'target' or 'position'."
            old = list(self.ee_pos); self.ee_pos = dest; d = self._dist(old, dest)
            if self.holding and self.holding in self.objects: self.objects[self.holding]["pos"] = list(dest)
            r = f"Moved: {old} -> {dest} ({d:.2f}m)"
            if target: r += f", near '{target}'"
            if self.holding: r += f". Carrying {self.holding}."
            return r
    def grasp(self, action):
        with self._lock:
            if action == "close":
                if self.gripper == "close": return "Already closed."
                self.gripper = "close"
                for n, o in self.objects.items():
                    if self._dist(self.ee_pos, o["pos"]) < GRASP_DISTANCE and n != self.holding:
                        self.holding = n; return f"Grasped '{n}'."
                return "Nothing in range."
            elif action == "open":
                if self.gripper == "open": return "Already open."
                self.gripper = "open"
                if self.holding:
                    rel = self.holding
                    for n, o in self.objects.items():
                        if n != rel and self._dist(self.ee_pos, o["pos"]) < 0.10:
                            self.objects[rel]["on"] = n; self.holding = None; return f"Released '{rel}' onto '{n}'."
                    self.objects[rel]["on"] = "counter"; self.holding = None; return f"Released '{rel}' onto counter."
                return "Nothing held."
            return f"Error: 'open' or 'close'."

mock_env = MockRobotEnv()


# ============================================================
# Real mode functions
# ============================================================

def real_look(q=""):
    resp = requests.get(f"{SIM_URL}/render", timeout=5)
    img_b64 = base64.b64encode(resp.content).decode()
    prompt = q if q else "Describe the scene."
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


# ============================================================
# Robot tool runners
# ============================================================

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
    for turn in range(15):
        resp = client.messages.create(model=MODEL, system=SUBAGENT_SYSTEM, messages=sub, tools=CHILD_TOOLS, max_tokens=4096)
        sub.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use": break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": run_look(b.input.get("question", ""))})
        sub.append({"role": "user", "content": results})
    return "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"


# ============================================================
# Background robot operations
# ============================================================

def run_bg_look(question=""):
    jid = str(uuid.uuid4())[:8]
    desc = f"look: {question or '(monitor)'}"
    def _bg_look():
        time.sleep(1)
        return run_look(question)
    return BG.run(jid, _bg_look, (), desc)

def run_bg_act(instruction=""):
    jid = str(uuid.uuid4())[:8]
    desc = f"act: {instruction[:60]}"
    def _bg_act():
        time.sleep(2)
        if SIM_URL and VLA_URL:
            return real_act(instruction)
        instr = instruction.lower()
        if "move" in instr:
            for obj_name in mock_env.objects:
                if obj_name.split()[-1] in instr or obj_name in instr:
                    return mock_env.move(target=obj_name)
            return "Could not determine move target."
        elif "close" in instr or "grasp" in instr:
            return mock_env.grasp("close")
        elif "open" in instr or "release" in instr:
            return mock_env.grasp("open")
        return f"Executed: {instruction}"
    return BG.run(jid, _bg_act, (), desc)


# ============================================================
# RobotTeamManager — persistent named robot agents
# ============================================================

class RobotTeamManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "robot-team", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        print(f"\033[35m[team] Spawned '{name}' (role: {role})\033[0m")
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """Agent loop for a teammate robot. Runs in its own thread."""
        role_hints = {
            "scout": "You specialize in perception. Use look frequently to observe the scene. Use perceive for thorough analysis.",
            "manipulator": "You specialize in manipulation. Use move and grasp to interact with objects. Load skills when needed.",
            "monitor": "You specialize in safety monitoring. Use look continuously to watch for hazards or changes.",
        }
        hint = role_hints.get(role, f"You perform {role} tasks.")
        sys_prompt = (
            f"You are '{name}', a {role} robot. {hint} "
            f"Use robot tools to complete your task. "
            f"Use send_message to report results to 'lead' or other teammates. "
            f"Use read_inbox to check for new messages."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        for turn in range(50):
            # Drain inbox before each LLM call
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                # Check for shutdown request
                if msg.get("type") == "shutdown_request":
                    BUS.send(name, msg["from"], f"{name} shutting down.", "shutdown_response")
                    print(f"\033[35m[team] {name} received shutdown request, exiting\033[0m")
                    member = self._find_member(name)
                    if member: member["status"] = "shutdown"; self._save_config()
                    return
                messages.append({"role": "user", "content": f"<inbox>{json.dumps(msg)}</inbox>"})
                messages.append({"role": "assistant", "content": "Noted inbox message."})

            try:
                response = client.messages.create(
                    model=MODEL, system=sys_prompt, messages=messages,
                    tools=tools, max_tokens=4096,
                )
            except Exception as e:
                print(f"\033[31m[team] {name} LLM error: {e}\033[0m")
                break
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._teammate_exec(name, block.name, block.input)
                    print(f"\033[32m  [{name}] {block.name}: {str(output)[:120]}\033[0m")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
            messages.append({"role": "user", "content": results})

        # Mark idle when done
        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()
        print(f"\033[35m[team] {name} finished (idle)\033[0m")

    def _teammate_exec(self, sender: str, tool_name: str, args: dict) -> str:
        """Dispatch tool calls for teammate robots."""
        if tool_name == "look":
            return run_look(args.get("question", ""))
        if tool_name == "move":
            return run_move(args.get("target", ""), args.get("position"))
        if tool_name == "grasp":
            return run_grasp(args["action"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """Tool definitions available to teammate robots."""
        return [
            {"name": "look", "description": "Observe the scene. Ask a question to focus.",
             "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
            {"name": "move", "description": "Move end-effector to target object or position.",
             "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
            {"name": "grasp", "description": "Open or close the gripper.",
             "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
            {"name": "send_message", "description": "Send a message to a teammate or to 'lead'.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = RobotTeamManager(TEAM_DIR)


# ============================================================
# System prompt — lead robot agent
# ============================================================

SYSTEM = f"""\
You are the lead robot agent controlling a Unitree G1 humanoid robot team.

You can spawn specialist robot teammates that run in parallel threads:
- "scout" (role: perception) — uses look/perceive heavily to observe the scene
- "manipulator" (role: manipulation) — uses move/grasp/load_skill for actions
- "monitor" (role: safety) — uses look continuously to watch for hazards

Communication:
- send_message to talk to a specific teammate
- broadcast to message all teammates
- read_inbox to check messages from teammates
- Teammates report results via send_message to "lead"

Direct tools (you can also act directly):
- Sync: look, move, grasp, perceive, load_skill, compact
- Async: bg_look, bg_act, check_background
- Planning: task_create, task_update, task_list, task_get

Strategy: Spawn scouts to observe, manipulators to act, monitors for safety.
Coordinate via messages. Check inbox regularly for teammate reports.

Skills: {SKILL_LOADER.get_descriptions()}"""


# ============================================================
# Lead tool dispatch
# ============================================================

TOOL_HANDLERS = {
    # Robot tools (from r08)
    "look":             lambda **kw: run_look(kw.get("question", "")),
    "move":             lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp":            lambda **kw: run_grasp(kw["action"]),
    "perceive":         lambda **kw: run_perceive(kw["goal"]),
    "load_skill":       lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":          lambda **kw: "Compressing...",
    # Task tools
    "task_create":      lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update":      lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":        lambda **kw: TASKS.list_all(),
    "task_get":         lambda **kw: TASKS.get(kw["task_id"]),
    # Background tools
    "bg_look":          lambda **kw: run_bg_look(kw.get("question", "")),
    "bg_act":           lambda **kw: run_bg_act(kw.get("instruction", "")),
    "check_background": lambda **kw: BG.check(kw.get("job_id")),
    # Team tools (new in r09)
    "spawn_robot":      lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_robots":      lambda **kw: TEAM.list_all(),
    "send_message":     lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":       lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":        lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}


# ============================================================
# Lead tool definitions
# ============================================================

TOOLS = [
    # Robot tools
    {"name": "look", "description": "Quick observation (blocking).",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "move", "description": "Move end-effector (blocking).",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "Gripper control (blocking).",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "perceive", "description": "Thorough scene analysis (blocking, spawns subagent).",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
    {"name": "load_skill", "description": "Load manipulation skill.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compact", "description": "Compress context.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
    # Task tools
    {"name": "task_create", "description": "Create task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update task status/deps.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get task details.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    # Background tools
    {"name": "bg_look", "description": "Non-blocking observation. Result delivered next turn.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "bg_act", "description": "Non-blocking action. Runs VLA+sim in background.",
     "input_schema": {"type": "object", "properties": {"instruction": {"type": "string"}}, "required": ["instruction"]}},
    {"name": "check_background", "description": "Check background job status.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}}},
    # Team tools (new in r09)
    {"name": "spawn_robot", "description": "Spawn a persistent robot teammate in its own thread. Roles: scout, manipulator, monitor.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Unique name for the robot."}, "role": {"type": "string", "description": "Role: scout, manipulator, monitor, or custom."}, "prompt": {"type": "string", "description": "Initial task instruction for the robot."}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_robots", "description": "List all robot teammates with name, role, status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a robot teammate's inbox.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all robot teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
]


# ============================================================
# Agent loop — drain inbox + background notifications before each LLM call
# ============================================================

def agent_loop(messages: list):
    while True:
        # Drain background notifications
        notifs = BG.drain()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['job_id']}] {n['status']}: {n['description']}\n  {n['result']}" for n in notifs
            )
            print(f"\033[32m[bg notification] {len(notifs)} job(s) completed\033[0m")
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
            messages.append({"role": "assistant", "content": "Noted background results."})

        # Drain lead inbox
        inbox = BUS.read_inbox("lead")
        if inbox:
            print(f"\033[35m[inbox] {len(inbox)} message(s) from teammates\033[0m")
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages.",
            })

        # Compression
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
                    manual_compact = True; output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try: output = handler(**block.input) if handler else f"Unknown: {block.name}"
                    except Exception as e: output = f"Error: {e}"
                # Color-coded output by category
                if block.name in ("spawn_robot", "list_robots", "send_message", "read_inbox", "broadcast"):
                    print(f"\033[35m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:400]}\033[0m")
                elif block.name.startswith("task_"):
                    print(f"\033[35m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:400]}\033[0m")
                elif block.name.startswith("bg_"):
                    print(f"\033[32m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:200]}\033[0m")
                elif block.name in ("load_skill", "compact"):
                    print(f"\033[36m[{block.name}]\033[0m")
                elif block.name == "perceive":
                    print(f"\033[34m[perceive]\033[0m")
                    print(f"\033[90m{str(output)[:600]}\033[0m")
                else:
                    print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        messages.append({"role": "user", "content": results})
        if manual_compact:
            messages[:] = auto_compact(messages)


# ============================================================
# REPL
# ============================================================

if __name__ == "__main__":
    mode = "REAL" if (SIM_URL and VLA_URL) else "MOCK"
    print(f"\033[32m[r09] Multi-Robot Teams  |  {mode}\033[0m")
    print(f"\033[90mRobot: look, move, grasp | Team: spawn_robot, send_message, broadcast\033[0m")
    print(f"\033[90mAsync: bg_look, bg_act   | Plan: task_create, task_update\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = query.strip().lower()
        if cmd in ("q", "exit", ""): break
        if cmd == "reset":
            mock_env.reset(); TASKS.clear_all(); BG.jobs.clear(); history = []
            # Clean team state
            if TEAM.config_path.exists(): TEAM.config_path.unlink()
            TEAM.config = TEAM._load_config(); TEAM.threads.clear()
            for f in INBOX_DIR.glob("*.jsonl"): f.unlink()
            print("Reset.\n"); continue
        if cmd == "tasks": print(TASKS.list_all()); print(); continue
        if cmd == "bg": print(BG.check()); print(); continue
        if cmd == "skills": print(SKILL_LOADER.get_descriptions()); print(); continue
        if cmd == "team": print(TEAM.list_all()); print(); continue
        if cmd == "inbox":
            msgs = BUS.read_inbox("lead")
            print(json.dumps(msgs, indent=2) if msgs else "(empty)")
            print(); continue
        if cmd == "/compact": history[:] = auto_compact(history); print("Compressed.\n"); continue
        if cmd == "tokens": print(f"~{estimate_tokens(history)//4} tokens, {len(history)} msgs\n"); continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"): print(block.text)
        print()
