#!/usr/bin/env python3
"""
r10_safety_protocols.py - Safety Protocols

Four-layer safety for robot agents:
  Layer 1: Hardware limits (simulated) — max velocity, force limits
  Layer 2: Zone enforcement — keep-out and slow-down zones
  Layer 3: Pre-action hooks — check before every move/grasp
  Layer 4: Agent-level E-stop — freeze all actions immediately

    Action request                      Safety pipeline
    +------------------+               +------------------+
    | move(target)     | ------------> | E-stop active?   |
    +------------------+               +--------+---------+
                                                |
                                       +--------v---------+
                                       | Zone check       |
                                       | (forbidden/warn) |
                                       +--------+---------+
                                                |
                                       +--------v---------+
                                       | Velocity limit   |
                                       | (max 0.5 m/step) |
                                       +--------+---------+
                                                |
                                       +--------v---------+
                                       | Execute action   |
                                       | Log to audit     |
                                       +------------------+

Key insight: "Every action passes through the safety pipeline before execution."

Adapts learn-claude-code s10_team_protocols.py (shutdown/plan-approval protocols)
to robot-specific safety. Instead of coding protocols, we add robot safety layers.

Demo:

    r10 >> add a safety zone around the mug
    [add_safety_zone] name="mug_zone", center=[0.50, 0.30, 0.82], radius=0.1, level="forbidden"
    Safety zone 'mug_zone' added.

    r10 >> pick up the red apple
    [move] target="red apple"
    [SAFETY] Move allowed: no zone violations.
    Moved to red apple.

    [move] target="blue mug"
    [SAFETY] Move BLOCKED: enters forbidden zone 'mug_zone' (distance: 0.05m < radius: 0.10m)

    r10 >> emergency stop
    [e_stop]
    EMERGENCY STOP activated. All actions blocked.

    [move] target="red apple"
    [SAFETY] E-STOP active. All actions blocked. Call reset_estop to resume.
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
SAFETY_DIR = Path(__file__).resolve().parent.parent / ".safety"

THRESHOLD = 50000
KEEP_RECENT = 3


# ============================================================
# SafetyManager — central safety enforcement
# ============================================================
#
# Four layers enforced in check_move / check_grasp:
#   1. E-stop gate   — if active, block everything immediately
#   2. Zone check    — forbidden zones block, warning zones slow
#   3. Velocity cap  — max displacement per move
#   4. Audit log     — every check recorded to .safety/audit.jsonl
# ============================================================

class SafetyManager:
    """Central safety enforcement for all robot actions.

    Manages safety zones, e-stop state, velocity limits,
    and an append-only audit log of every safety check.
    """

    def __init__(self, safety_dir: Path):
        self.dir = safety_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.dir / "audit.jsonl"
        self.zones = {}          # name -> {center, radius, level}
        self.estop = False
        self.max_velocity = 0.5  # metres per move step
        self._lock = threading.Lock()

    # -- E-stop -------------------------------------------------------

    def emergency_stop(self) -> str:
        with self._lock:
            self.estop = True
            self._log("e_stop", False, "EMERGENCY STOP activated")
        return "EMERGENCY STOP activated. All actions blocked."

    def reset_estop(self) -> str:
        with self._lock:
            if not self.estop:
                return "E-stop is not active."
            self.estop = False
            self._log("reset_estop", True, "E-stop cleared")
        return "E-stop cleared. Actions resumed."

    # -- Zone management -----------------------------------------------

    def add_zone(self, name: str, center: list, radius: float,
                 level: str = "forbidden") -> str:
        if level not in ("warning", "forbidden"):
            return f"Error: level must be 'warning' or 'forbidden', got '{level}'."
        if radius <= 0:
            return "Error: radius must be positive."
        if not center or len(center) < 3:
            return "Error: center must be [x, y, z]."
        with self._lock:
            self.zones[name] = {
                "center": list(center[:3]),
                "radius": float(radius),
                "level": level,
            }
            self._log("add_zone", True,
                       f"Zone '{name}': center={center[:3]}, r={radius}, level={level}")
        return f"Safety zone '{name}' added ({level}, r={radius:.2f}m)."

    def remove_zone(self, name: str) -> str:
        with self._lock:
            if name not in self.zones:
                return f"Error: zone '{name}' not found."
            del self.zones[name]
            self._log("remove_zone", True, f"Zone '{name}' removed")
        return f"Safety zone '{name}' removed."

    # -- Pre-action checks --------------------------------------------

    def check_move(self, from_pos: list, to_pos: list) -> tuple:
        """Return (allowed: bool, message: str).

        Checks in order: e-stop, zone violations, velocity cap.
        """
        with self._lock:
            # Layer 1: E-stop
            if self.estop:
                reason = "E-STOP active. All actions blocked. Call reset_estop to resume."
                self._log("check_move", False, reason)
                return False, f"[SAFETY] {reason}"

            # Layer 2: Zone check — does the target enter any zone?
            zone_msg = self._check_zones(to_pos)
            if zone_msg is not None:
                blocked, text = zone_msg
                self._log("check_move", not blocked, text)
                if blocked:
                    return False, f"[SAFETY] {text}"
                # warning zone: allow but annotate
                warning_text = text

            else:
                warning_text = None

            # Layer 3: Velocity limit
            dist = self._dist(from_pos, to_pos)
            if dist > self.max_velocity:
                reason = (f"Move distance {dist:.2f}m exceeds max velocity "
                          f"{self.max_velocity}m/step. Move closer first.")
                self._log("check_move", False, reason)
                return False, f"[SAFETY] {reason}"

            # All checks passed
            if warning_text:
                self._log("check_move", True, f"Allowed with warning: {warning_text}")
                return True, f"[SAFETY] WARNING: {warning_text} Proceeding with caution."
            self._log("check_move", True, "Move allowed: no zone violations.")
            return True, "[SAFETY] Move allowed: no zone violations."

    def check_grasp(self) -> tuple:
        """Return (allowed: bool, message: str)."""
        with self._lock:
            if self.estop:
                reason = "E-STOP active. All actions blocked. Call reset_estop to resume."
                self._log("check_grasp", False, reason)
                return False, f"[SAFETY] {reason}"
            self._log("check_grasp", True, "Grasp allowed.")
            return True, "[SAFETY] Grasp allowed."

    def _check_zones(self, pos: list):
        """Check if pos violates any zone. Must be called under _lock.

        Returns None if no zones violated, or (blocked: bool, message: str).
        """
        for name, zone in self.zones.items():
            dist = self._dist(pos, zone["center"])
            if dist < zone["radius"]:
                if zone["level"] == "forbidden":
                    return (True,
                            f"Move BLOCKED: enters forbidden zone '{name}' "
                            f"(distance: {dist:.2f}m < radius: {zone['radius']:.2f}m)")
                else:  # warning
                    return (False,
                            f"Entering warning zone '{name}' "
                            f"(distance: {dist:.2f}m < radius: {zone['radius']:.2f}m).")
        return None

    # -- Audit log -----------------------------------------------------

    def _log(self, action: str, allowed: bool, reason: str):
        """Append an audit entry. Must be called under _lock (or externally safe)."""
        entry = {
            "ts": time.time(),
            "action": action,
            "allowed": allowed,
            "reason": reason,
        }
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # best-effort logging

    def get_log(self, limit: int = 20) -> str:
        """Return last N audit log entries."""
        if not self.audit_path.exists():
            return "No audit log entries."
        lines = self.audit_path.read_text().strip().splitlines()
        if not lines:
            return "No audit log entries."
        recent = lines[-limit:]
        entries = []
        for line in recent:
            try:
                e = json.loads(line)
                ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
                status = "OK" if e["allowed"] else "DENIED"
                entries.append(f"  [{ts}] {status} {e['action']}: {e['reason']}")
            except (json.JSONDecodeError, KeyError):
                entries.append(f"  (malformed entry)")
        return "\n".join(entries)

    # -- Status --------------------------------------------------------

    def status(self) -> str:
        """Return current safety state: e-stop, zones, velocity limit."""
        with self._lock:
            lines = [f"E-stop: {'ACTIVE' if self.estop else 'off'}"]
            lines.append(f"Max velocity: {self.max_velocity} m/step")
            if self.zones:
                lines.append(f"Safety zones ({len(self.zones)}):")
                for name, z in self.zones.items():
                    lines.append(
                        f"  - {name}: center={z['center']}, "
                        f"r={z['radius']:.2f}m, level={z['level']}")
            else:
                lines.append("Safety zones: none")
            # Last 5 audit entries
            log_preview = self.get_log(5)
            if "No audit" not in log_preview:
                lines.append(f"Recent audit:\n{log_preview}")
            return "\n".join(lines)

    # -- Helpers -------------------------------------------------------

    @staticmethod
    def _dist(a: list, b: list) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


SAFETY = SafetyManager(SAFETY_DIR)


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
    ss = SAFETY.status()
    conv = json.dumps(msgs, default=str)[:80000]
    resp = client.messages.create(model=MODEL, messages=[{"role": "user", "content":
        f"Summarize robot conversation. Include: original task, progress, strategies, safety state.\n\nRobot: {rs}\nTasks:\n{ts}\nSafety:\n{ss}\n\nConversation:\n{conv}"}], max_tokens=2000)
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {p.name}]\n\n{resp.content[0].text}\n\nRobot: {rs}\nSafety: {ss}\n\nCall task_list for current progress."},
        {"role": "assistant", "content": "Understood. Checking task board."},
    ]


# ============================================================
# Mock Environment (from r05)
# ============================================================

GRASP_DISTANCE = 0.08

class MockRobotEnv:
    def __init__(self): self.reset()
    def reset(self):
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
        lines = ["Scene:"]
        for n, o in self.objects.items():
            loc = "held" if n == self.holding else f"on {o['on']}"
            lines.append(f"  - {n}: {o['pos']}, {loc}, {o['shape']}")
        lines.append(f"Robot: ee={self.ee_pos}, gripper={self.gripper}, holding={self.holding or 'nothing'}")
        if question:
            q = question.lower(); m = self._find(q)
            if m:
                o = self.objects[m]; d = self._dist(self.ee_pos, o["pos"])
                on_what = 'held' if m == self.holding else f"on {o['on']}"
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
# Base tool handlers (real + mock)
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


# ============================================================
# Safety-wrapped tool handlers
# ============================================================
#
# These replace the raw move/grasp handlers. Every action is
# checked through the SafetyManager pipeline before execution.
# ============================================================

def safe_move(target="", position=None):
    """Move with full safety pipeline: e-stop -> zones -> velocity."""
    # Resolve target position without actually moving
    if target:
        m = mock_env._find(target)
        if not m:
            return f"Error: unknown '{target}'."
        target_pos = list(mock_env.objects[m]["pos"])
    elif position and len(position) >= 3:
        target_pos = list(position[:3])
    else:
        return "Error: provide 'target' or 'position'."

    current_pos = list(mock_env.ee_pos)

    # Run safety pipeline
    allowed, msg = SAFETY.check_move(current_pos, target_pos)
    if not allowed:
        return msg

    # Safety passed — execute the actual move
    result = run_move(target, position)

    # Post-move: scan for human mentions in result
    if "human" in result.lower() or "person" in result.lower():
        offset_pos = [mock_env.ee_pos[0] + 0.2,
                      mock_env.ee_pos[1],
                      mock_env.ee_pos[2]]
        SAFETY.add_zone("human_detected", offset_pos, 0.3, "warning")
        result += "\n[SAFETY] Human detected near move target — warning zone added."

    return f"{msg}\n{result}"


def safe_grasp(action):
    """Grasp with safety check: e-stop gate."""
    allowed, msg = SAFETY.check_grasp()
    if not allowed:
        return msg

    result = run_grasp(action)
    return f"{msg}\n{result}"


# ============================================================
# Perception with safety — human detection post-hook
# ============================================================

def run_look_with_safety(question=""):
    """Look with post-hook: auto-add warning zone on human detection."""
    result = run_look(question)

    # Scan result for human/person mentions
    lower = result.lower()
    if "human" in lower or "person" in lower:
        # Place warning zone offset from current robot position
        offset_pos = [mock_env.ee_pos[0] + 0.3,
                      mock_env.ee_pos[1],
                      mock_env.ee_pos[2]]
        SAFETY.add_zone("human_detected", offset_pos, 0.3, "warning")
        result += "\n[SAFETY] Human detected — warning zone added at estimated position."

    return result


# Perception subagent (with safety-aware look)
SUBAGENT_SYSTEM = "You are a perception specialist. Multiple look() calls, then summarize."
CHILD_TOOLS = [{"name": "look", "description": "Observe scene.",
                "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}}]

def run_perceive(goal):
    sub = [{"role": "user", "content": f"Analyze: {goal}\nUse look() multiple times, then summarize."}]
    for turn in range(15):
        resp = client.messages.create(model=MODEL, system=SUBAGENT_SYSTEM, messages=sub, tools=CHILD_TOOLS, max_tokens=4096)
        sub.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use": break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": run_look_with_safety(b.input.get("question", ""))})
        sub.append({"role": "user", "content": results})
    return "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"


# ============================================================
# Background robot operations (with safety)
# ============================================================

def run_bg_look(question=""):
    """Background perception — non-blocking look with safety hook."""
    jid = str(uuid.uuid4())[:8]
    desc = f"look: {question or '(monitor)'}"
    def _bg_look():
        time.sleep(1)  # simulate VLM latency
        return run_look_with_safety(question)
    return BG.run(jid, _bg_look, (), desc)

def run_bg_act(instruction=""):
    """Background action — non-blocking VLA+sim execution."""
    jid = str(uuid.uuid4())[:8]
    desc = f"act: {instruction[:60]}"
    def _bg_act():
        time.sleep(2)  # simulate VLA+sim latency
        if SIM_URL and VLA_URL:
            return real_act(instruction)
        # Mock: parse simple instructions
        instr = instruction.lower()
        if "move" in instr:
            for obj_name in mock_env.objects:
                if obj_name.split()[-1] in instr or obj_name in instr:
                    return safe_move(target=obj_name)
            return "Could not determine move target."
        elif "close" in instr or "grasp" in instr:
            return safe_grasp("close")
        elif "open" in instr or "release" in instr:
            return safe_grasp("open")
        return f"Executed: {instruction}"
    return BG.run(jid, _bg_act, (), desc)


# ============================================================
# Safety tool handlers — new tools for r10
# ============================================================

def handle_e_stop(**_kw):
    return SAFETY.emergency_stop()

def handle_reset_estop(**_kw):
    return SAFETY.reset_estop()

def handle_add_safety_zone(**kw):
    name = kw.get("name", "")
    center = kw.get("center", [])
    radius = kw.get("radius", 0.1)
    level = kw.get("level", "forbidden")
    if not name:
        return "Error: 'name' is required."
    return SAFETY.add_zone(name, center, radius, level)

def handle_remove_safety_zone(**kw):
    name = kw.get("name", "")
    if not name:
        return "Error: 'name' is required."
    return SAFETY.remove_zone(name)

def handle_safety_status(**_kw):
    return SAFETY.status()

def handle_safety_log(**kw):
    limit = kw.get("limit", 20)
    if not isinstance(limit, int) or limit < 1:
        limit = 20
    return SAFETY.get_log(limit)


# ============================================================
# System prompt
# ============================================================

SYSTEM = f"""\
You are a robot agent controlling a Unitree G1 humanoid robot.

Tools:
- Sync: look, move, grasp, perceive, load_skill, compact
- Async: bg_look, bg_act, check_background
- Planning: task_create, task_update, task_list, task_get
- Safety: e_stop, reset_estop, add_safety_zone, remove_safety_zone, safety_status, safety_log

Safety: All moves and grasps pass through a 4-layer safety pipeline before execution.
  Layer 1: E-stop gate — if active, ALL actions are blocked.
  Layer 2: Zone enforcement — "forbidden" zones block movement, "warning" zones allow with caution.
  Layer 3: Velocity limit — max {SAFETY.max_velocity}m per move step.
  Layer 4: Audit log — every check recorded.

Use e_stop for emergencies. Use add_safety_zone to protect areas or objects.
When look/perceive detects a human, a warning zone is automatically added.

Use bg_look/bg_act when you want non-blocking perception or action.
Results are delivered automatically before your next turn.

Skills: {SKILL_LOADER.get_descriptions()}"""


# ============================================================
# Dispatch map — move and grasp use safety-wrapped handlers
# ============================================================

TOOL_HANDLERS = {
    # Core (safety-wrapped)
    "look":               lambda **kw: run_look_with_safety(kw.get("question", "")),
    "move":               lambda **kw: safe_move(kw.get("target", ""), kw.get("position")),
    "grasp":              lambda **kw: safe_grasp(kw["action"]),
    "perceive":           lambda **kw: run_perceive(kw["goal"]),
    "load_skill":         lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":            lambda **kw: "Compressing...",
    # Tasks
    "task_create":        lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update":        lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":          lambda **kw: TASKS.list_all(),
    "task_get":           lambda **kw: TASKS.get(kw["task_id"]),
    # Background
    "bg_look":            lambda **kw: run_bg_look(kw.get("question", "")),
    "bg_act":             lambda **kw: run_bg_act(kw.get("instruction", "")),
    "check_background":   lambda **kw: BG.check(kw.get("job_id")),
    # Safety (new in r10)
    "e_stop":             lambda **kw: handle_e_stop(**kw),
    "reset_estop":        lambda **kw: handle_reset_estop(**kw),
    "add_safety_zone":    lambda **kw: handle_add_safety_zone(**kw),
    "remove_safety_zone": lambda **kw: handle_remove_safety_zone(**kw),
    "safety_status":      lambda **kw: handle_safety_status(**kw),
    "safety_log":         lambda **kw: handle_safety_log(**kw),
}


# ============================================================
# Tool definitions (schemas for the LLM)
# ============================================================

TOOLS = [
    # --- Core tools (safety-wrapped) ---
    {"name": "look", "description": "Quick observation (blocking). Auto-detects humans and adds warning zones.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "move", "description": "Move end-effector (blocking). Passes through safety pipeline: e-stop, zones, velocity.",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "Gripper control (blocking). Checked against e-stop before execution.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "perceive", "description": "Thorough scene analysis (blocking, spawns subagent). Includes human detection.",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
    {"name": "load_skill", "description": "Load manipulation skill.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compact", "description": "Compress context.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},

    # --- Task tools ---
    {"name": "task_create", "description": "Create task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update task status/deps.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get task details.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},

    # --- Background tools ---
    {"name": "bg_look", "description": "Non-blocking observation. Starts in background, result delivered next turn.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "bg_act", "description": "Non-blocking action. Runs VLA+sim in background. Safety-checked.",
     "input_schema": {"type": "object", "properties": {"instruction": {"type": "string", "description": "Action instruction, e.g. 'move to apple', 'close gripper'."}}, "required": ["instruction"]}},
    {"name": "check_background", "description": "Check background job status. Omit job_id to list all.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}}},

    # --- Safety tools (new in r10) ---
    {"name": "e_stop", "description": "EMERGENCY STOP. Immediately freezes all robot actions. No parameters needed.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "reset_estop", "description": "Clear emergency stop and resume normal operation. No parameters needed.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "add_safety_zone",
     "description": "Add a safety zone. 'forbidden' blocks all movement into the zone. 'warning' allows with caution.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Zone identifier, e.g. 'mug_zone', 'human_area'."},
         "center": {"type": "array", "items": {"type": "number"}, "description": "[x, y, z] center position."},
         "radius": {"type": "number", "description": "Radius in metres."},
         "level": {"type": "string", "enum": ["warning", "forbidden"], "description": "Zone type. Default: forbidden."}
     }, "required": ["name", "center", "radius"]}},
    {"name": "remove_safety_zone", "description": "Remove a named safety zone.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Zone name to remove."}
     }, "required": ["name"]}},
    {"name": "safety_status", "description": "Show current safety state: e-stop, zones, velocity limit, recent audit.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "safety_log", "description": "Show recent safety audit log entries.",
     "input_schema": {"type": "object", "properties": {
         "limit": {"type": "integer", "description": "Number of entries to show (default 20)."}
     }}},
]


# ============================================================
# Agent loop — drain background notifications, run safety checks
# ============================================================

def agent_loop(messages: list):
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
                # Print by category
                if block.name.startswith("task_"):
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
                elif block.name in ("e_stop", "reset_estop", "add_safety_zone",
                                     "remove_safety_zone", "safety_status", "safety_log"):
                    # Safety tools — red/yellow highlight
                    print(f"\033[91m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
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
    print(f"\033[32m[r10] Safety Protocols  |  {mode}\033[0m")
    print(f"\033[90mSync: look, move, grasp, perceive | Async: bg_look, bg_act\033[0m")
    print(f"\033[90mSafety: e_stop, reset_estop, add_safety_zone, remove_safety_zone, safety_status, safety_log\033[0m")
    print(f"\033[90mAll actions pass through 4-layer safety pipeline.\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = query.strip().lower()
        if cmd in ("q", "exit", ""): break
        if cmd == "reset":
            mock_env.reset(); TASKS.clear_all(); BG.jobs.clear()
            SAFETY.estop = False; SAFETY.zones.clear()
            history = []
            print("Reset.\n"); continue
        if cmd == "tasks": print(TASKS.list_all()); print(); continue
        if cmd == "bg": print(BG.check()); print(); continue
        if cmd == "skills": print(SKILL_LOADER.get_descriptions()); print(); continue
        if cmd == "safety": print(SAFETY.status()); print(); continue
        if cmd == "audit": print(SAFETY.get_log(30)); print(); continue
        if cmd == "/compact": history[:] = auto_compact(history); print("Compressed.\n"); continue
        if cmd == "tokens": print(f"~{estimate_tokens(history)//4} tokens, {len(history)} msgs\n"); continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"): print(block.text)
        print()
