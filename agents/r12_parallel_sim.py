#!/usr/bin/env python3
"""
r12_parallel_sim.py - Parallel Simulation Environments

Directory-level isolation for parallel strategy testing. Fork the
environment, try multiple approaches, compare, promote the best.

    Main Environment          Sim "top-grasp"        Sim "side-grasp"
    +----------------+       +----------------+      +----------------+
    | apple on table |  fork | apple on table |      | apple on table |
    | ee=[0.30,...]  | ----> | ee=[0.30,...]  |      | ee=[0.30,...]  |
    +----------------+       +--------+-------+      +--------+-------+
                                      |                       |
                              try top-grasp            try side-grasp
                                      |                       |
                              apple held, 3 moves      apple dropped!
                                      |                       |
                              +-------v-------+               |
                              | PROMOTE       |        DISCARD
                              | copy to main  |
                              +---------------+

    .sims/
      sim_top-grasp.json   {"name":"top-grasp", "status":"active", "task_id":1}
      sim_side-grasp.json  {"name":"side-grasp", "status":"discarded"}

Key insight: "Try it in a sandbox first, promote what works."

Why parallel sim matters for robotics:
- Physical actions are irreversible (dropped object, collision)
- Testing strategies in sim is free, testing on hardware is expensive
- Fork-try-compare-promote lets the agent evaluate before committing
- Multiple grasp approaches can be tested simultaneously
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
SIMS_DIR = Path(__file__).resolve().parent.parent / ".sims"

THRESHOLD = 50000
KEEP_RECENT = 3


# ============================================================
# Mock Environment (thread-safe)
# ============================================================

GRASP_DISTANCE = 0.08

class MockRobotEnv:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.objects = {
            "red apple": {"pos": [0.45, 0.12, 0.82], "on": "counter", "shape": "round-flat"},
            "white plate": {"pos": [0.45, -0.15, 0.80], "on": "counter", "shape": "flat"},
            "blue mug": {"pos": [0.50, 0.30, 0.82], "on": "counter", "shape": "tall-handle"},
            "fork": {"pos": [0.55, -0.20, 0.81], "on": "counter", "shape": "flat-long"},
            "napkin": {"pos": [0.60, 0.00, 0.81], "on": "counter", "shape": "flat"},
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

    def snapshot(self):
        """Return a deep copy of current state for forking."""
        with self._lock:
            return {
                "objects": {k: dict(v) for k, v in self.objects.items()},
                "ee_pos": list(self.ee_pos),
                "gripper": self.gripper,
                "holding": self.holding,
            }

    def restore(self, snap):
        """Restore state from a snapshot dict."""
        with self._lock:
            self.objects = {k: dict(v) for k, v in snap["objects"].items()}
            self.ee_pos = list(snap["ee_pos"])
            self.gripper = snap["gripper"]
            self.holding = snap["holding"]


mock_env = MockRobotEnv()


# ============================================================
# SafetyManager — workspace limits + collision checks
# ============================================================

class SafetyManager:
    """Simple safety layer for the main environment."""
    def __init__(self):
        self.workspace_min = [0.0, -0.6, 0.60]
        self.workspace_max = [0.8, 0.6, 1.20]
        self.max_speed = 0.5       # m per move
        self.violations = []

    def check_move(self, current_pos, target_pos):
        """Check if a move is safe. Returns (ok, reason)."""
        # Workspace bounds
        for i, axis in enumerate(["x", "y", "z"]):
            if target_pos[i] < self.workspace_min[i] or target_pos[i] > self.workspace_max[i]:
                msg = f"Out of workspace: {axis}={target_pos[i]:.2f} (bounds [{self.workspace_min[i]}, {self.workspace_max[i]}])"
                self.violations.append({"type": "workspace", "msg": msg, "ts": time.time()})
                return False, msg
        # Speed limit
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)))
        if dist > self.max_speed:
            msg = f"Move too large: {dist:.2f}m > {self.max_speed}m limit"
            self.violations.append({"type": "speed", "msg": msg, "ts": time.time()})
            return False, msg
        return True, "ok"

    def status(self):
        lines = [f"Safety: workspace=[{self.workspace_min}, {self.workspace_max}], max_speed={self.max_speed}m"]
        if self.violations:
            lines.append(f"  violations: {len(self.violations)}")
            for v in self.violations[-3:]:
                lines.append(f"    [{v['type']}] {v['msg']}")
        else:
            lines.append("  violations: 0")
        return "\n".join(lines)


SAFETY = SafetyManager()


# ============================================================
# SimEnvironment — isolated copy of MockRobotEnv
# ============================================================

class SimEnvironment:
    """An isolated copy of MockRobotEnv with its own state.

    Each sim is a sandbox: actions taken here do NOT affect the main
    environment. The agent can fork, test, compare, and promote.
    """
    def __init__(self, name, source_env):
        self.name = name
        self.env = MockRobotEnv()
        # Deep copy state from source
        snap = source_env.snapshot()
        self.env.restore(snap)
        self.action_log = []       # track actions taken in this sim
        self.created_at = time.time()
        self.status = "active"     # active | promoted | discarded
        self.task_id = None

    def state_summary(self):
        """One-line state summary."""
        return (
            f"ee={self.env.ee_pos}, gripper={self.env.gripper}, "
            f"holding={self.env.holding or 'nothing'}"
        )


# ============================================================
# SimManager — create / run / compare / promote / discard sims
# ============================================================

class SimManager:
    """Manages multiple isolated simulation instances.

    Analogous to WorktreeManager in s12, but instead of git worktrees
    we fork MockRobotEnv state into isolated SimEnvironment instances.
    """
    def __init__(self, sims_dir, main_env):
        self.dir = sims_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.main_env = main_env
        self.sims = {}             # name -> SimEnvironment
        self.events = []           # lifecycle events
        self._lock = threading.Lock()

    def create(self, name, task_id=None):
        """Fork current main env into new isolated sim."""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            return "Error: Invalid sim name. Use 1-40 chars: letters, numbers, ., _, -"
        with self._lock:
            if name in self.sims:
                return f"Error: Sim '{name}' already exists."
            sim = SimEnvironment(name, self.main_env)
            sim.task_id = task_id
            self.sims[name] = sim
            self._save_meta(name, task_id)
            self._log_event("sim.created", name, {"task_id": task_id})
        return f"Sim '{name}' created (forked from main env). State: {sim.state_summary()}"

    def run_action(self, name, tool, args):
        """Execute a robot tool in a specific sim environment."""
        sim = self.sims.get(name)
        if not sim:
            return f"Error: Unknown sim '{name}'. Use list_sims to see available."
        if sim.status != "active":
            return f"Error: Sim '{name}' is {sim.status}, cannot run actions."

        # Execute tool on sim.env instead of main env (no safety checks)
        result = self._dispatch_tool(sim.env, tool, args)
        sim.action_log.append({
            "tool": tool,
            "args": args,
            "result": result[:200],
            "ts": time.time(),
        })
        self._log_event("sim.action", name, {"tool": tool})
        return f"[sim:{name}] {result}"

    def _dispatch_tool(self, env, tool, args):
        """Dispatch a tool call to a specific env instance."""
        if tool == "look":
            return env.look(args.get("question", ""))
        elif tool == "move":
            return env.move(args.get("target", ""), args.get("position"))
        elif tool == "grasp":
            action = args.get("action", "")
            if not action:
                return "Error: grasp requires 'action' (open/close)."
            return env.grasp(action)
        else:
            return f"Error: Tool '{tool}' not supported in sim. Use look/move/grasp."

    def compare(self, name1, name2):
        """Compare state of two sims side-by-side."""
        s1 = self.sims.get(name1)
        s2 = self.sims.get(name2)
        if not s1:
            return f"Error: Unknown sim '{name1}'."
        if not s2:
            return f"Error: Unknown sim '{name2}'."

        lines = [f"=== Comparison: {name1} vs {name2} ==="]

        # Sim 1 state
        lines.append(f"\n{name1} [{s1.status}]:")
        lines.append(f"  ee={s1.env.ee_pos}, gripper={s1.env.gripper}, "
                      f"holding={s1.env.holding or 'nothing'}")
        lines.append(f"  actions: {len(s1.action_log)}")
        for n, o in s1.env.objects.items():
            lines.append(f"  {n}: on {o['on']}")

        # Sim 2 state
        lines.append(f"\n{name2} [{s2.status}]:")
        lines.append(f"  ee={s2.env.ee_pos}, gripper={s2.env.gripper}, "
                      f"holding={s2.env.holding or 'nothing'}")
        lines.append(f"  actions: {len(s2.action_log)}")
        for n, o in s2.env.objects.items():
            lines.append(f"  {n}: on {o['on']}")

        # Differences
        diffs = []
        for n in s1.env.objects:
            on1 = s1.env.objects[n]["on"]
            on2 = s2.env.objects.get(n, {}).get("on", "?")
            if on1 != on2:
                diffs.append(f"  {n}: {on1} vs {on2}")
        if s1.env.holding != s2.env.holding:
            diffs.append(f"  holding: {s1.env.holding or 'nothing'} vs {s2.env.holding or 'nothing'}")
        if s1.env.ee_pos != s2.env.ee_pos:
            diffs.append(f"  ee_pos: {s1.env.ee_pos} vs {s2.env.ee_pos}")
        if s1.env.gripper != s2.env.gripper:
            diffs.append(f"  gripper: {s1.env.gripper} vs {s2.env.gripper}")

        if diffs:
            lines.append(f"\nDifferences ({len(diffs)}):")
            lines.extend(diffs)
        else:
            lines.append("\nNo differences detected.")

        # Action log summary
        lines.append(f"\nAction logs:")
        lines.append(f"  {name1}: {len(s1.action_log)} actions")
        for a in s1.action_log:
            lines.append(f"    {a['tool']}({json.dumps(a['args'], ensure_ascii=False)}) -> {a['result'][:80]}")
        lines.append(f"  {name2}: {len(s2.action_log)} actions")
        for a in s2.action_log:
            lines.append(f"    {a['tool']}({json.dumps(a['args'], ensure_ascii=False)}) -> {a['result'][:80]}")

        return "\n".join(lines)

    def promote(self, name):
        """Copy sim state back to main environment (accept changes)."""
        sim = self.sims.get(name)
        if not sim:
            return f"Error: Unknown sim '{name}'."
        if sim.status != "active":
            return f"Error: Sim '{name}' is already {sim.status}."

        with self._lock:
            snap = sim.env.snapshot()
            self.main_env.restore(snap)
            sim.status = "promoted"
            self._update_meta(name, "promoted")
            self._log_event("sim.promoted", name, {
                "actions_applied": len(sim.action_log),
                "holding": sim.env.holding or "nothing",
            })
        return (
            f"Sim '{name}' promoted to main environment. "
            f"{len(sim.action_log)} actions applied. "
            f"Main env now: ee={self.main_env.ee_pos}, "
            f"holding={self.main_env.holding or 'nothing'}."
        )

    def discard(self, name):
        """Discard a sim without affecting main env (reject changes)."""
        sim = self.sims.get(name)
        if not sim:
            return f"Error: Unknown sim '{name}'."
        if sim.status != "active":
            return f"Error: Sim '{name}' is already {sim.status}."

        sim.status = "discarded"
        self._update_meta(name, "discarded")
        self._log_event("sim.discarded", name, {
            "actions_wasted": len(sim.action_log),
        })
        return f"Sim '{name}' discarded. {len(sim.action_log)} actions thrown away. Main env unchanged."

    def list_all(self):
        """List all sim environments with status."""
        if not self.sims:
            return "No sim environments."
        lines = ["Sim environments:"]
        for n, s in self.sims.items():
            task_str = f" task=#{s.task_id}" if s.task_id else ""
            age = time.time() - s.created_at
            age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.1f}m"
            lines.append(
                f"  [{s.status}] {n}: {s.state_summary()}, "
                f"actions={len(s.action_log)}, age={age_str}{task_str}"
            )
        active = sum(1 for s in self.sims.values() if s.status == "active")
        promoted = sum(1 for s in self.sims.values() if s.status == "promoted")
        discarded = sum(1 for s in self.sims.values() if s.status == "discarded")
        lines.append(f"\n({active} active, {promoted} promoted, {discarded} discarded)")
        return "\n".join(lines)

    def get_events(self, limit=20):
        """Return recent lifecycle events."""
        recent = self.events[-limit:]
        if not recent:
            return "No sim events."
        lines = ["Sim lifecycle events:"]
        for e in recent:
            ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            detail = json.dumps(e.get("detail", {}), ensure_ascii=False)
            lines.append(f"  [{ts}] {e['event']} sim={e['sim']} {detail}")
        return "\n".join(lines)

    def _save_meta(self, name, task_id=None):
        """Persist sim metadata to disk."""
        meta = {
            "name": name,
            "task_id": task_id,
            "status": "active",
            "created_at": time.time(),
        }
        (self.dir / f"sim_{name}.json").write_text(json.dumps(meta, indent=2))

    def _update_meta(self, name, status):
        """Update persisted sim status."""
        p = self.dir / f"sim_{name}.json"
        if p.exists():
            meta = json.loads(p.read_text())
            meta["status"] = status
            meta["updated_at"] = time.time()
            p.write_text(json.dumps(meta, indent=2))

    def _log_event(self, event, sim_name, detail=None):
        """Append a lifecycle event."""
        self.events.append({
            "event": event,
            "sim": sim_name,
            "ts": time.time(),
            "detail": detail or {},
        })

    def cleanup(self):
        """Remove all sims and reset."""
        self.sims.clear()
        self.events.clear()
        for f in self.dir.glob("sim_*.json"):
            f.unlink()


SIMS = SimManager(SIMS_DIR, mock_env)


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
    ss = SIMS.list_all()
    conv = json.dumps(msgs, default=str)[:80000]
    resp = client.messages.create(model=MODEL, messages=[{"role": "user", "content":
        f"Summarize robot conversation. Include: original task, progress, strategies, sim results.\n\nRobot: {rs}\nTasks:\n{ts}\nSims:\n{ss}\n\nConversation:\n{conv}"}], max_tokens=2000)
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {p.name}]\n\n{resp.content[0].text}\n\nRobot: {rs}\nSims: {ss}\n\nCall task_list for current progress."},
        {"role": "assistant", "content": "Understood. Checking task board and sim status."},
    ]


# ============================================================
# Tool handlers — sync, background, and sim management
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
    # Safety check on main env only
    if target:
        m = mock_env._find(target)
        if m:
            dest = list(mock_env.objects[m]["pos"])
            ok, reason = SAFETY.check_move(mock_env.ee_pos, dest)
            if not ok: return f"SAFETY BLOCK: {reason}"
    elif position and len(position) >= 3:
        ok, reason = SAFETY.check_move(mock_env.ee_pos, position[:3])
        if not ok: return f"SAFETY BLOCK: {reason}"
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


# ============================================================
# System prompt
# ============================================================

SYSTEM = f"""\
You are a robot agent controlling a Unitree G1 humanoid robot.

Strategy testing: Use create_sim to fork the environment, try different
approaches in parallel sims, compare_sims to evaluate, promote the best.

Example workflow:
1. create_sim("approach-A") + create_sim("approach-B")
2. sim_action("approach-A", "move", {{"target": "apple"}})
3. sim_action("approach-A", "grasp", {{"action": "close"}})
4. sim_action("approach-B", "move", {{"target": "apple"}})
5. compare_sims("approach-A", "approach-B")
6. promote_sim("approach-A")  -- apply best strategy to main env
7. discard_sim("approach-B")  -- throw away the losing approach

Tools:
- Perception: look, perceive
- Action: move, grasp
- Sim management: create_sim, sim_action, compare_sims, promote_sim, discard_sim, list_sims
- Async: bg_look, bg_act, check_background
- Planning: task_create, task_update, task_list, task_get
- Utility: load_skill, compact

Safety checks apply to main env moves only. Sim actions bypass safety
(that is the point -- sims are sandboxes for experimentation).

Skills: {SKILL_LOADER.get_descriptions()}"""


# ============================================================
# Tool definitions + dispatch map
# ============================================================

TOOL_HANDLERS = {
    # Perception
    "look":             lambda **kw: run_look(kw.get("question", "")),
    "perceive":         lambda **kw: run_perceive(kw["goal"]),
    # Action (main env, safety-checked)
    "move":             lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp":            lambda **kw: run_grasp(kw["action"]),
    # Sim management
    "create_sim":       lambda **kw: SIMS.create(kw["name"], kw.get("task_id")),
    "sim_action":       lambda **kw: SIMS.run_action(kw["sim_name"], kw["tool"], kw.get("args", {})),
    "compare_sims":     lambda **kw: SIMS.compare(kw["sim1"], kw["sim2"]),
    "promote_sim":      lambda **kw: SIMS.promote(kw["name"]),
    "discard_sim":      lambda **kw: SIMS.discard(kw["name"]),
    "list_sims":        lambda **kw: SIMS.list_all(),
    # Background
    "bg_look":          lambda **kw: run_bg_look(kw.get("question", "")),
    "bg_act":           lambda **kw: run_bg_act(kw.get("instruction", "")),
    "check_background": lambda **kw: BG.check(kw.get("job_id")),
    # Tasks
    "task_create":      lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update":      lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":        lambda **kw: TASKS.list_all(),
    "task_get":         lambda **kw: TASKS.get(kw["task_id"]),
    # Utility
    "load_skill":       lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":          lambda **kw: "Compressing...",
}

TOOLS = [
    # -- Perception --
    {"name": "look", "description": "Quick observation (blocking).",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "perceive", "description": "Thorough scene analysis (blocking, spawns subagent).",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},

    # -- Action (main env) --
    {"name": "move", "description": "Move end-effector in MAIN env (blocking, safety-checked).",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "Gripper control in MAIN env (blocking).",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},

    # -- Sim management --
    {"name": "create_sim", "description": "Fork main env into a new isolated sim. Actions in the sim do NOT affect main.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Sim name (1-40 chars, alphanumeric/dash/dot/underscore)."},
         "task_id": {"type": "integer", "description": "Optional task ID to associate with this sim."},
     }, "required": ["name"]}},
    {"name": "sim_action", "description": "Execute a robot tool (look/move/grasp) in a specific sim. No safety checks.",
     "input_schema": {"type": "object", "properties": {
         "sim_name": {"type": "string", "description": "Name of the sim to act in."},
         "tool": {"type": "string", "enum": ["look", "move", "grasp"], "description": "Which tool to run."},
         "args": {"type": "object", "description": "Tool arguments (same as the tool's normal args)."},
     }, "required": ["sim_name", "tool"]}},
    {"name": "compare_sims", "description": "Side-by-side comparison of two sim environments.",
     "input_schema": {"type": "object", "properties": {
         "sim1": {"type": "string"},
         "sim2": {"type": "string"},
     }, "required": ["sim1", "sim2"]}},
    {"name": "promote_sim", "description": "Copy sim state to main env (accept the sim's changes).",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"},
     }, "required": ["name"]}},
    {"name": "discard_sim", "description": "Throw away a sim (reject its changes, main env unaffected).",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"},
     }, "required": ["name"]}},
    {"name": "list_sims", "description": "List all sim environments with status, state, and action counts.",
     "input_schema": {"type": "object", "properties": {}}},

    # -- Background --
    {"name": "bg_look", "description": "Non-blocking observation. Starts in background, result delivered next turn.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "bg_act", "description": "Non-blocking action. Runs VLA+sim in background.",
     "input_schema": {"type": "object", "properties": {"instruction": {"type": "string", "description": "Action instruction."}}, "required": ["instruction"]}},
    {"name": "check_background", "description": "Check background job status. Omit job_id to list all.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}}},

    # -- Tasks --
    {"name": "task_create", "description": "Create task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update task status/deps.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get task details.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},

    # -- Utility --
    {"name": "load_skill", "description": "Load manipulation skill.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compact", "description": "Compress context.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
]


# ============================================================
# Agent loop — drain background, manage compression
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
                elif block.name in ("create_sim", "promote_sim", "discard_sim", "list_sims"):
                    print(f"\033[94m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                elif block.name == "sim_action":
                    print(f"\033[96m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                elif block.name == "compare_sims":
                    print(f"\033[93m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:800]}\033[0m")
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
    print(f"\033[32m[r12] Parallel Simulation Environments  |  {mode}\033[0m")
    print(f"\033[90mSync: look, move, grasp, perceive | Sim: create_sim, sim_action, compare_sims, promote_sim, discard_sim\033[0m")
    print(f"\033[90mAsync: bg_look, bg_act | Tasks: task_create, task_update, task_list\033[0m")
    print(f"\033[90mREPL: reset, tasks, sims, bg, skills, tokens, safety, /compact, q\033[0m\n")

    history = []
    while True:
        try:
            query = input("\033[36mr12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = query.strip().lower()
        if cmd in ("q", "exit", ""): break
        if cmd == "reset":
            mock_env.reset(); TASKS.clear_all(); BG.jobs.clear(); SIMS.cleanup(); history = []
            print("Reset.\n"); continue
        if cmd == "tasks": print(TASKS.list_all()); print(); continue
        if cmd == "sims": print(SIMS.list_all()); print(); continue
        if cmd == "bg": print(BG.check()); print(); continue
        if cmd == "skills": print(SKILL_LOADER.get_descriptions()); print(); continue
        if cmd == "safety": print(SAFETY.status()); print(); continue
        if cmd == "events": print(SIMS.get_events()); print(); continue
        if cmd == "/compact": history[:] = auto_compact(history); print("Compressed.\n"); continue
        if cmd == "tokens": print(f"~{estimate_tokens(history)//4} tokens, {len(history)} msgs\n"); continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"): print(block.text)
        print()
