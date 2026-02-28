#!/usr/bin/env python3
"""
r11_autonomous_patrol.py - Autonomous Patrol

The robot finds work itself: patrol waypoints, scan for anomalies,
claim tasks from the board, and report to lead.

    Patrol lifecycle:
    +--------+
    | START  |
    +---+----+
        |
        v
    +--------+  waypoint    +--------+
    | PATROL | -----------> | LOOK   |
    +---+----+              +---+----+
        |                       |
        | all done              | anomaly?
        v                       v
    +--------+              +--------+
    | IDLE   | poll 5s      | REPORT |
    +---+----+              +--------+
        |
        +---> check inbox -> message? -> resume PATROL
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> WORK
        |
        +---> timeout (60s) -> start new patrol cycle

Key insight: "The robot finds work itself -- patrol, detect, act."

Why autonomous patrol matters for robotics:
- Factory/warehouse robots must patrol without human supervision
- Anomaly detection is the #1 use case for mobile robots
- The idle-cycle pattern turns a reactive robot into a proactive one
- Task-board integration lets the robot coordinate with other agents

Builds on r08's BackgroundManager, TaskManager, SkillLoader, compression,
MockRobotEnv, and adds: PatrolManager, SafetyManager, auto-claim from
task board, anomaly detection, identity re-injection after compression,
and the autonomous idle-cycle that drives the patrol loop.
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

THRESHOLD = 50000
KEEP_RECENT = 3

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60


# ============================================================
# SafetyManager — enforce workspace limits, velocity caps, etc.
# ============================================================

class SafetyManager:
    """Hardware-level safety layer. Every move/grasp goes through here."""
    def __init__(self):
        self.workspace_min = [0.05, -0.50, 0.60]
        self.workspace_max = [0.90, 0.50, 1.20]
        self.max_velocity = 0.30       # m per move step
        self.max_force = 10.0          # N (simulated)
        self.violations = []
        self._lock = threading.Lock()

    def check_position(self, pos):
        for i, axis in enumerate(["x", "y", "z"]):
            if pos[i] < self.workspace_min[i] or pos[i] > self.workspace_max[i]:
                v = f"Position {axis}={pos[i]:.2f} outside [{self.workspace_min[i]}, {self.workspace_max[i]}]"
                with self._lock:
                    self.violations.append({"type": "workspace", "msg": v, "t": time.time()})
                return False, v
        return True, "OK"

    def check_velocity(self, old_pos, new_pos):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(old_pos, new_pos)))
        if d > self.max_velocity:
            v = f"Move distance {d:.2f}m exceeds max {self.max_velocity}m"
            with self._lock:
                self.violations.append({"type": "velocity", "msg": v, "t": time.time()})
            return False, v
        return True, "OK"

    def status(self):
        with self._lock:
            recent = self.violations[-5:] if self.violations else []
        lines = [f"Safety: {len(self.violations)} total violations"]
        for v in recent:
            lines.append(f"  [{v['type']}] {v['msg']}")
        return "\n".join(lines)


SAFETY = SafetyManager()


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
        self._lock = threading.Lock()
    def _max_id(self):
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0
    def _load(self, tid):
        p = self.dir / f"task_{tid}.json"
        if not p.exists(): raise ValueError(f"Task {tid} not found")
        return json.loads(p.read_text())
    def _save(self, t): (self.dir / f"task_{t['id']}.json").write_text(json.dumps(t, indent=2))
    def create(self, subject, description=""):
        with self._lock:
            t = {"id": self._next_id, "subject": subject, "description": description,
                 "status": "pending", "blockedBy": [], "blocks": [], "owner": ""}
            self._save(t); self._next_id += 1
        return json.dumps(t, indent=2)
    def get(self, tid): return json.dumps(self._load(tid), indent=2)
    def update(self, tid, status=None, add_blocked_by=None, add_blocks=None):
        with self._lock:
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
            self._save(t)
        return json.dumps(t, indent=2)
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
            o = f" @{t['owner']}" if t.get("owner") else ""
            lines.append(f"{m} #{t['id']}: {t['subject']}{o}{b}")
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
# Compression (from r06) + identity re-injection
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
    """Compress conversation, then re-inject patrol identity."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    p = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(p, "w") as f:
        for m in msgs: f.write(json.dumps(m, default=str) + "\n")
    rs = f"ee={mock_env.ee_pos}, gripper={mock_env.gripper}, holding={mock_env.holding or 'nothing'}"
    ts = TASKS.list_all()
    ps = PATROL.status()
    conv = json.dumps(msgs, default=str)[:80000]
    resp = client.messages.create(model=MODEL, messages=[{"role": "user", "content":
        f"Summarize robot conversation. Include: original task, progress, strategies, "
        f"patrol findings.\n\nRobot: {rs}\nTasks:\n{ts}\nPatrol:\n{ps}\n\nConversation:\n{conv}"}],
        max_tokens=2000)
    # Re-inject identity so the agent remembers its patrol role
    identity = _make_patrol_identity()
    return [
        {"role": "user", "content":
            f"{identity}\n\n"
            f"[Compressed. Transcript: {p.name}]\n\n"
            f"{resp.content[0].text}\n\nRobot: {rs}\n\n"
            f"Call patrol_status and task_list for current progress."},
        {"role": "assistant", "content": "Understood. Checking patrol status and task board."},
    ]


def _make_patrol_identity():
    """Build identity block for re-injection after compression."""
    ps = PATROL.status_dict()
    route = ps.get("active_route", "none")
    idx = ps.get("current_waypoint_idx", 0)
    total = ps.get("total_waypoints", 0)
    mode = ps.get("mode", "once")
    anomaly_count = ps.get("anomaly_count", 0)
    return (
        f"<identity>You are an autonomous patrol robot (Unitree G1 humanoid). "
        f"Active route: {route} (waypoint {idx}/{total}, mode={mode}). "
        f"Anomalies detected: {anomaly_count}. "
        f"Continue your patrol. Check task board for new work.</identity>"
    )


# ============================================================
# PatrolManager — waypoint-based patrol system
# ============================================================

ANOMALY_KEYWORDS = [
    "obstacle", "spill", "fallen", "broken", "human", "person",
    "fire", "smoke", "wet", "damage", "blocked", "hazard",
]

DEFAULT_ROUTES = {
    "kitchen_sweep": [
        {"name": "counter_left",   "position": [0.30, -0.20, 0.90],
         "check": "any spills or fallen objects on the left counter?"},
        {"name": "counter_center", "position": [0.45, 0.00, 0.90],
         "check": "any obstacles or misplaced items on the center counter?"},
        {"name": "counter_right",  "position": [0.60, 0.20, 0.90],
         "check": "any hazards or items out of place on the right counter?"},
        {"name": "sink_area",      "position": [0.70, -0.10, 0.85],
         "check": "is the sink area clear? any water spills?"},
    ],
    "safety_check": [
        {"name": "entry_point",  "position": [0.20, 0.00, 0.95],
         "check": "any humans or obstacles near the entry?"},
        {"name": "work_area",    "position": [0.45, 0.00, 0.90],
         "check": "is the work area safe? any humans nearby?"},
        {"name": "exit_point",   "position": [0.70, 0.00, 0.95],
         "check": "any obstacles blocking the exit?"},
    ],
}


class PatrolManager:
    """
    Waypoint-based patrol system.

    Routes are ordered lists of waypoints. Each waypoint has a name,
    position, and a perception query (what to check when arriving).
    Modes: "once" (single pass), "loop" (continuous), "timed" (every N s).
    """

    def __init__(self):
        self.routes = dict(DEFAULT_ROUTES)   # name -> [waypoints]
        self.active_route = None
        self.current_waypoint_idx = 0
        self.mode = "once"                   # once | loop | timed
        self.interval = 60                   # seconds between timed cycles
        self.anomalies = []                  # detected anomalies log
        self.patrol_count = 0                # total completed patrol cycles
        self.last_patrol_end = 0.0           # timestamp of last cycle end
        self._lock = threading.Lock()

    def add_route(self, name, waypoints):
        """Add or overwrite a patrol route."""
        with self._lock:
            self.routes[name] = waypoints
        return f"Route '{name}' added with {len(waypoints)} waypoints."

    def start_patrol(self, route_name, mode="once"):
        """Begin patrolling a named route."""
        with self._lock:
            if route_name not in self.routes:
                return f"Error: Unknown route '{route_name}'. Available: {list(self.routes.keys())}"
            self.active_route = route_name
            self.current_waypoint_idx = 0
            self.mode = mode
        return f"Patrol '{route_name}' started in {mode} mode ({len(self.routes[route_name])} waypoints)."

    def stop_patrol(self):
        """Stop the active patrol."""
        with self._lock:
            prev = self.active_route
            self.active_route = None
            self.current_waypoint_idx = 0
        if prev:
            return f"Patrol '{prev}' stopped."
        return "No active patrol."

    def next_waypoint(self):
        """
        Return the next waypoint dict, or None if the route is complete.
        For loop mode, wraps around. For timed mode, waits for interval.
        """
        with self._lock:
            if not self.active_route:
                return None
            route = self.routes.get(self.active_route, [])
            if not route:
                return None
            if self.current_waypoint_idx >= len(route):
                # Route complete for this cycle
                self.patrol_count += 1
                self.last_patrol_end = time.time()
                if self.mode == "loop":
                    self.current_waypoint_idx = 0
                elif self.mode == "timed":
                    # Check if enough time has passed for next cycle
                    elapsed = time.time() - self.last_patrol_end
                    if elapsed < self.interval:
                        return None   # not yet time
                    self.current_waypoint_idx = 0
                else:
                    # mode == "once" -- done
                    self.active_route = None
                    return None
            if self.current_waypoint_idx >= len(route):
                return None
            wp = dict(route[self.current_waypoint_idx])
            wp["index"] = self.current_waypoint_idx
            wp["total"] = len(route)
            wp["route"] = self.active_route
            self.current_waypoint_idx += 1
            return wp

    def report_anomaly(self, waypoint_name, description):
        """Log an anomaly detected at a waypoint."""
        entry = {
            "waypoint": waypoint_name,
            "description": description,
            "timestamp": time.time(),
        }
        with self._lock:
            self.anomalies.append(entry)
        return f"Anomaly logged at '{waypoint_name}': {description}"

    def status(self):
        """Human-readable patrol status."""
        d = self.status_dict()
        lines = [f"Patrol: {d['state']}"]
        if d["active_route"]:
            lines.append(f"  Route: {d['active_route']} ({d['mode']} mode)")
            lines.append(f"  Waypoint: {d['current_waypoint_idx']}/{d['total_waypoints']}")
        lines.append(f"  Completed cycles: {d['patrol_count']}")
        lines.append(f"  Anomalies: {d['anomaly_count']}")
        if d["recent_anomalies"]:
            for a in d["recent_anomalies"]:
                lines.append(f"    [{a['waypoint']}] {a['description']}")
        return "\n".join(lines)

    def status_dict(self):
        """Structured patrol status for identity re-injection."""
        with self._lock:
            route = self.routes.get(self.active_route, [])
            return {
                "state": "patrolling" if self.active_route else "idle",
                "active_route": self.active_route,
                "mode": self.mode,
                "current_waypoint_idx": self.current_waypoint_idx,
                "total_waypoints": len(route),
                "patrol_count": self.patrol_count,
                "anomaly_count": len(self.anomalies),
                "recent_anomalies": self.anomalies[-3:],
            }

    def list_routes(self):
        """List all available routes."""
        with self._lock:
            if not self.routes:
                return "No routes defined."
            lines = []
            for name, wps in self.routes.items():
                active = " (ACTIVE)" if name == self.active_route else ""
                wp_names = ", ".join(w["name"] for w in wps)
                lines.append(f"  {name}: {len(wps)} waypoints [{wp_names}]{active}")
            return "\n".join(lines)


PATROL = PatrolManager()


# ============================================================
# Task board scanning and auto-claim (from s11 pattern)
# ============================================================

_claim_lock = threading.Lock()


def scan_unclaimed_tasks() -> list:
    """Find pending tasks with no owner and no blockers."""
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    """Atomically claim a task for the given owner."""
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if task.get("owner"):
            return f"Error: Task #{task_id} already claimed by {task['owner']}"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


# ============================================================
# Anomaly detection
# ============================================================

def detect_anomaly(perception_result: str) -> str | None:
    """
    Check perception text for anomaly keywords.
    Returns a description if anomaly found, else None.
    """
    lower = perception_result.lower()
    found = [kw for kw in ANOMALY_KEYWORDS if kw in lower]
    if found:
        # Extract a short description from the perception result
        # Take the first 150 chars as description
        desc = perception_result.strip()[:150]
        return f"Detected: {', '.join(found)}. {desc}"
    return None


# ============================================================
# Mock Environment (from r05, thread-safe)
# ============================================================

GRASP_DISTANCE = 0.08


class MockRobotEnv:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock if hasattr(self, "_lock") else threading.Lock():
            self.objects = {
                "red apple":   {"pos": [0.45, 0.12, 0.82], "on": "counter", "shape": "round-flat"},
                "white plate": {"pos": [0.45, -0.15, 0.80], "on": "counter", "shape": "flat"},
                "blue mug":    {"pos": [0.50, 0.30, 0.82], "on": "counter", "shape": "tall-handle"},
                "fork":        {"pos": [0.55, -0.20, 0.81], "on": "counter", "shape": "flat-long"},
                "napkin":      {"pos": [0.60, 0.00, 0.81], "on": "counter", "shape": "flat"},
            }
            self.ee_pos = [0.30, 0.0, 0.95]; self.gripper = "open"; self.holding = None

    def _dist(self, a, b): return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

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
                    loc = "held" if m == self.holding else f"on {o['on']}"
                    lines.append(f"=> {m}: {loc}, {d:.2f}m, {o['shape']}")
                elif any(w in q for w in ("success", "done", "plate")):
                    on_p = [n for n, o in self.objects.items() if o["on"] == "white plate" and n != "white plate"]
                    lines.append(f"=> On plate: {on_p}" if on_p else "=> Plate empty.")
                elif any(w in q for w in ("hold", "grip")):
                    lines.append(f"=> Gripper: {self.gripper}, holding: {self.holding or 'nothing'}")
                elif any(w in q for w in ("obstacle", "spill", "hazard", "human", "fire", "block")):
                    # Patrol perception: normally nothing found in mock
                    lines.append("=> Area clear. No anomalies detected.")
                elif any(w in q for w in ("change", "moved", "different")):
                    lines.append("=> No changes detected since last observation.")
                else:
                    lines.append(f"=> No specific information for: {question}")
        return "\n".join(lines)

    def move(self, target="", position=None):
        with self._lock:
            if target:
                m = self._find(target)
                if not m: return f"Error: unknown '{target}'."
                dest = list(self.objects[m]["pos"])
            elif position and len(position) >= 3:
                dest = position[:3]
            else:
                return "Error: provide 'target' or 'position'."
            # Safety checks
            ok, msg = SAFETY.check_position(dest)
            if not ok: return f"SAFETY BLOCK: {msg}"
            ok, msg = SAFETY.check_velocity(self.ee_pos, dest)
            if not ok: return f"SAFETY BLOCK: {msg}"
            old = list(self.ee_pos); self.ee_pos = dest; d = self._dist(old, dest)
            if self.holding and self.holding in self.objects:
                self.objects[self.holding]["pos"] = list(dest)
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
                            self.objects[rel]["on"] = n; self.holding = None
                            return f"Released '{rel}' onto '{n}'."
                    self.objects[rel]["on"] = "counter"; self.holding = None
                    return f"Released '{rel}' onto counter."
                return "Nothing held."
        return f"Error: 'open' or 'close'."


mock_env = MockRobotEnv()


# ============================================================
# Tool handlers (sync and background)
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
        a = requests.post(f"{VLA_URL}/predict", json={"observation": obs, "instruction": instruction},
                          timeout=30).json().get("action", {})
        sr = requests.post(f"{SIM_URL}/step", json={"action": a}, timeout=5).json()
        if sr.get("terminated") or sr.get("success"): return f"Done after {i + 1} steps."
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
CHILD_TOOLS = [{"name": "look", "description": "Observe scene.",
                "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}}]


def run_perceive(goal):
    sub = [{"role": "user", "content": f"Analyze: {goal}\nUse look() multiple times, then summarize."}]
    for turn in range(15):
        resp = client.messages.create(model=MODEL, system=SUBAGENT_SYSTEM, messages=sub,
                                      tools=CHILD_TOOLS, max_tokens=4096)
        sub.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use": break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": run_look(b.input.get("question", ""))})
        sub.append({"role": "user", "content": results})
    return "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"


# Background robot operations
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


# Patrol tool handlers
def run_add_route(name, waypoints):
    return PATROL.add_route(name, waypoints)


def run_start_patrol(route_name, mode="once"):
    return PATROL.start_patrol(route_name, mode)


def run_stop_patrol():
    return PATROL.stop_patrol()


def run_patrol_status():
    return PATROL.status()


def run_next_waypoint():
    wp = PATROL.next_waypoint()
    if wp is None:
        return "No next waypoint. Patrol idle or route complete."
    return json.dumps(wp, indent=2)


def run_list_routes():
    return PATROL.list_routes()


def run_report_anomaly(waypoint_name, description):
    result = PATROL.report_anomaly(waypoint_name, description)
    # Auto-create a task for the anomaly
    TASKS.create(
        subject=f"Anomaly at {waypoint_name}: {description[:60]}",
        description=f"Detected during patrol. Waypoint: {waypoint_name}. Details: {description}",
    )
    return result + " (task created)"


# ============================================================
# System prompt
# ============================================================

SYSTEM = f"""\
You are an autonomous patrol robot controlling a Unitree G1 humanoid.
You patrol waypoints, detect anomalies, and claim tasks from the board.

Workflow:
1. Start a patrol route (or use the defaults: kitchen_sweep, safety_check)
2. At each waypoint: move there, look with the check query
3. If anomaly detected: report it (creates a task automatically)
4. When patrol done: check task board for unclaimed work
5. If nothing to do: restart patrol or wait

Anomaly keywords: {', '.join(ANOMALY_KEYWORDS)}
If you see any of these in a look() result, use report_anomaly immediately.

Tools:
- Sync: look, move, grasp, perceive, load_skill, compact
- Async: bg_look, bg_act, check_background
- Planning: task_create, task_update, task_list, task_get, claim_task
- Patrol: add_route, start_patrol, stop_patrol, patrol_status, next_waypoint, list_routes, report_anomaly

After each waypoint inspection, call next_waypoint to advance.
In autonomous mode, waypoint instructions are injected automatically.

Skills: {SKILL_LOADER.get_descriptions()}"""


# ============================================================
# Dispatch map
# ============================================================

TOOL_HANDLERS = {
    "look":             lambda **kw: run_look(kw.get("question", "")),
    "move":             lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp":            lambda **kw: run_grasp(kw["action"]),
    "perceive":         lambda **kw: run_perceive(kw["goal"]),
    "load_skill":       lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":          lambda **kw: "Compressing...",
    "task_create":      lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update":      lambda **kw: TASKS.update(kw["task_id"], kw.get("status"),
                                                   kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":        lambda **kw: TASKS.list_all(),
    "task_get":         lambda **kw: TASKS.get(kw["task_id"]),
    "claim_task":       lambda **kw: claim_task(kw["task_id"], "patrol-robot"),
    "bg_look":          lambda **kw: run_bg_look(kw.get("question", "")),
    "bg_act":           lambda **kw: run_bg_act(kw.get("instruction", "")),
    "check_background": lambda **kw: BG.check(kw.get("job_id")),
    "add_route":        lambda **kw: run_add_route(kw["name"], kw["waypoints"]),
    "start_patrol":     lambda **kw: run_start_patrol(kw["route_name"], kw.get("mode", "once")),
    "stop_patrol":      lambda **kw: run_stop_patrol(),
    "patrol_status":    lambda **kw: run_patrol_status(),
    "next_waypoint":    lambda **kw: run_next_waypoint(),
    "list_routes":      lambda **kw: run_list_routes(),
    "report_anomaly":   lambda **kw: run_report_anomaly(kw["waypoint_name"], kw["description"]),
}

TOOLS = [
    # -- Sync robot tools --
    {"name": "look", "description": "Quick observation (blocking).",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "move", "description": "Move end-effector (blocking). Safety-checked.",
     "input_schema": {"type": "object", "properties": {
         "target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "Gripper control (blocking).",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "perceive", "description": "Thorough scene analysis (blocking, spawns subagent).",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
    {"name": "load_skill", "description": "Load manipulation skill.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compact", "description": "Compress context (re-injects patrol identity).",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
    # -- Async tools --
    {"name": "bg_look", "description": "Non-blocking observation. Result delivered next turn.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "bg_act", "description": "Non-blocking action. Runs VLA+sim in background.",
     "input_schema": {"type": "object", "properties": {
         "instruction": {"type": "string"}}, "required": ["instruction"]}},
    {"name": "check_background", "description": "Check background job status. Omit job_id to list all.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}}},
    # -- Task tools --
    {"name": "task_create", "description": "Create task.",
     "input_schema": {"type": "object", "properties": {
         "subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update task status/deps.",
     "input_schema": {"type": "object", "properties": {
         "task_id": {"type": "integer"},
         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
         "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
         "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get task details.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim an unclaimed task from the board.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    # -- Patrol tools --
    {"name": "add_route", "description": "Add a patrol route. waypoints: [{name, position, check}, ...]",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Route name"},
         "waypoints": {"type": "array", "items": {"type": "object", "properties": {
             "name": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}},
             "check": {"type": "string"}}, "required": ["name", "position", "check"]}}},
         "required": ["name", "waypoints"]}},
    {"name": "start_patrol", "description": "Start patrolling a route. Modes: once, loop, timed.",
     "input_schema": {"type": "object", "properties": {
         "route_name": {"type": "string"},
         "mode": {"type": "string", "enum": ["once", "loop", "timed"]}}, "required": ["route_name"]}},
    {"name": "stop_patrol", "description": "Stop the active patrol.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "patrol_status", "description": "Get current patrol state, waypoint progress, anomalies.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "next_waypoint", "description": "Get the next waypoint to visit. Returns null if route done.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_routes", "description": "List all available patrol routes.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "report_anomaly", "description": "Report an anomaly at a waypoint. Auto-creates a task.",
     "input_schema": {"type": "object", "properties": {
         "waypoint_name": {"type": "string"}, "description": {"type": "string"}},
         "required": ["waypoint_name", "description"]}},
]


# ============================================================
# Agent loop — with autonomous idle-cycle for patrol + task claim
# ============================================================

# autonomous mode flag: when True, the agent auto-injects patrol waypoints
_autonomous = False
_autonomous_lock = threading.Lock()


def set_autonomous(enabled: bool):
    global _autonomous
    with _autonomous_lock:
        _autonomous = enabled


def is_autonomous():
    with _autonomous_lock:
        return _autonomous


def agent_loop(messages: list):
    """
    Core agent loop with idle-cycle extension.

    After the LLM finishes (stop_reason != "tool_use"), instead of
    returning immediately, the loop checks for autonomous work:
      1. Next patrol waypoint -> inject waypoint instruction -> continue
      2. Unclaimed tasks on the board -> claim -> inject -> continue
      3. Nothing found -> return to REPL
    """
    while True:
        # Pre-loop: drain background notifications
        notifs = BG.drain()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['job_id']}] {n['status']}: {n['description']}\n  {n['result']}" for n in notifs
            )
            print(f"\033[32m[bg notification] {len(notifs)} job(s) completed\033[0m")
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
            messages.append({"role": "assistant", "content": "Noted background results."})

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
            # ---- IDLE PHASE: auto-inject next work if autonomous ----
            if is_autonomous():
                injected = _idle_inject(messages)
                if injected:
                    continue  # re-enter loop with injected instruction
            return

        # ---- WORK PHASE: execute tool calls ----
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
                # Print by category
                if block.name.startswith("task_") or block.name == "claim_task":
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
                elif block.name.startswith(("patrol", "start_patrol", "stop_patrol",
                                            "add_route", "next_waypoint", "list_routes",
                                            "report_anomaly")):
                    print(f"\033[96m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:400]}\033[0m")
                else:
                    print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        messages.append({"role": "user", "content": results})
        if manual_compact:
            messages[:] = auto_compact(messages)


def _idle_inject(messages: list) -> bool:
    """
    Idle-cycle injection. Checks for work in priority order:
      1. Next patrol waypoint
      2. Unclaimed tasks on the board
    Returns True if work was injected, False if truly idle.
    """
    # 1. Check for next patrol waypoint
    wp = PATROL.next_waypoint()
    if wp:
        wp_msg = (
            f"<patrol-waypoint>Move to '{wp['name']}' at {wp['position']}. "
            f"Then look: {wp['check']}. "
            f"(waypoint {wp['index'] + 1}/{wp['total']} on route '{wp['route']}')"
            f"</patrol-waypoint>"
        )
        print(f"\033[96m[patrol] -> {wp['name']} ({wp['index'] + 1}/{wp['total']})\033[0m")
        messages.append({"role": "user", "content": wp_msg})
        messages.append({"role": "assistant", "content":
            f"Continuing patrol. Moving to waypoint '{wp['name']}'."})
        return True

    # 2. Check task board for unclaimed work
    unclaimed = scan_unclaimed_tasks()
    if unclaimed:
        task = unclaimed[0]
        result = claim_task(task["id"], "patrol-robot")
        print(f"\033[35m[auto-claim] #{task['id']}: {task['subject']}\033[0m")
        messages.append({"role": "user", "content":
            f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
            f"{task.get('description', '')}</auto-claimed>"})
        messages.append({"role": "assistant", "content":
            f"Claimed task #{task['id']}. Working on it."})
        return True

    # 3. Nothing to do
    print(f"\033[90m[idle] No waypoints or tasks. Returning to REPL.\033[0m")
    return False


# ============================================================
# REPL
# ============================================================

def _print_help():
    print("""\
REPL commands:
  patrol         Show patrol status
  routes         List patrol routes
  tasks          Show task board
  bg             Show background jobs
  skills         List available skills
  safety         Show safety violations
  tokens         Show token estimate
  /compact       Force context compression
  reset          Reset env, tasks, patrol
  patrol_auto    Toggle autonomous patrol mode
  q / exit       Quit
""")


if __name__ == "__main__":
    mode = "REAL" if (SIM_URL and VLA_URL) else "MOCK"
    print(f"\033[32m[r11] Autonomous Patrol  |  {mode}\033[0m")
    print(f"\033[90mSync: look, move, grasp, perceive | Async: bg_look, bg_act\033[0m")
    print(f"\033[90mPatrol: start_patrol, next_waypoint, report_anomaly\033[0m")
    print(f"\033[90mType 'patrol_auto' to toggle autonomous mode. 'help' for commands.\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[96mr11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = query.strip().lower()
        if cmd in ("q", "exit", ""):
            break
        if cmd == "help":
            _print_help(); continue
        if cmd == "reset":
            mock_env.reset(); TASKS.clear_all(); BG.jobs.clear()
            PATROL.__init__(); set_autonomous(False); history = []
            print("Reset (env, tasks, patrol, autonomous=off).\n"); continue
        if cmd == "tasks":
            print(TASKS.list_all()); print(); continue
        if cmd == "bg":
            print(BG.check()); print(); continue
        if cmd == "skills":
            print(SKILL_LOADER.get_descriptions()); print(); continue
        if cmd == "safety":
            print(SAFETY.status()); print(); continue
        if cmd == "patrol":
            print(PATROL.status()); print(); continue
        if cmd == "routes":
            print(PATROL.list_routes()); print(); continue
        if cmd == "/compact":
            history[:] = auto_compact(history); print("Compressed.\n"); continue
        if cmd == "tokens":
            print(f"~{estimate_tokens(history) // 4} tokens, {len(history)} msgs\n"); continue
        if cmd == "patrol_auto":
            new_state = not is_autonomous()
            set_autonomous(new_state)
            state_str = "ON" if new_state else "OFF"
            print(f"\033[96m[autonomous] {state_str}\033[0m")
            if new_state and not PATROL.active_route:
                print(f"\033[90mNo active patrol. Tell the agent to start one first.\033[0m")
            print(); continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"): print(block.text)
        print()
