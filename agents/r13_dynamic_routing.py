#!/usr/bin/env python3
"""
r13_dynamic_routing.py - Multi-Frequency Dynamic Routing

The LLM thinks at 0.1Hz. The robot body runs at 50Hz. Decouple them.

Three nested frequency loops, each running concurrently:

    LLM (0.1Hz, REMOTE)            "pick up the apple"
     |  classify -> route -> start_subtask
     |
     +-- Perception Loop (VLM, 1-5Hz, LOCAL)
     |     activate_service("vlm")
     |     every 1/hz: look() -> update scene -> check VLM terminator
     |
     +-- Control Loop (VLA, 2-50Hz, LOCAL)
     |     activate_service("vla", lora="apple-pick-v2")
     |     every 1/hz: predict() -> step_delta() -> check terminators
     |
     +-- TerminationOracle
           CompositeTerminator([position, gripper, vlm_check, step_limit])
           -> fires termination_signal when done

    <loop-results> injected before next LLM call

Key insight: The agent ORCHESTRATES -- which model, at what frequency,
with what LoRA, and when to stop. VLAs cannot self-terminate, so
external supervision (terminators) decides when to stop.

Why this matters for robotics:
- LLM replanning at 50Hz is wasteful and latency-bound
- VLA needs tight, real-time control loops to be effective
- Different tasks need different control strategies (easy vs hard)
- LoRA adapters let the same VLA specialize per-task
"""

import os, re, math, json, time, base64, threading, uuid
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import yaml
except ImportError:
    yaml = None

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
VLA_BACKEND = os.environ.get("VLA_BACKEND", "mock")

ROUTES_DIR = Path(__file__).resolve().parent.parent / "routes"
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
TASKS_DIR = Path(__file__).resolve().parent.parent / ".tasks"
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / ".transcripts"

THRESHOLD = 50000; KEEP_RECENT = 3


# ============================================================
# RoutingConfig -- YAML-based routing strategy loader
# ============================================================

class RoutingConfig:
    """Loads routing configs from YAML files. Each file defines frequency,
    termination, and LoRA settings for easy/medium/hard difficulty levels."""
    def __init__(self, routes_dir):
        self.routes_dir = Path(routes_dir); self.configs = {}; self._load_all()

    def _load_all(self):
        if not self.routes_dir.exists():
            self.configs["default"] = self._hardcoded_default(); return
        loaded = False
        for f in sorted(self.routes_dir.glob("*.yaml")):
            cfg = self._load_yaml(f)
            if cfg: self.configs[cfg.get("name", f.stem)] = cfg; loaded = True
        if not loaded: self.configs["default"] = self._hardcoded_default()

    def _load_yaml(self, path):
        try: text = path.read_text()
        except Exception: return None
        if yaml is not None:
            try: return yaml.safe_load(text)
            except Exception: return None
        return self._parse_simple_yaml(text)

    def _parse_simple_yaml(self, text):
        """Rough YAML parser for flat/nested dicts when pyyaml unavailable."""
        result = {}; section = None; sub = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"): continue
            indent = len(line) - len(line.lstrip())
            if indent == 0 and ":" in stripped:
                key, _, val = stripped.partition(":"); val = val.strip().strip('"').strip("'")
                if val: result[key.strip()] = self._coerce(val)
                else: section = key.strip(); sub = None; result.setdefault(section, {})
            elif indent == 2 and section and ":" in stripped:
                key, _, val = stripped.partition(":"); val = val.strip().strip('"').strip("'")
                if val: result[section][key.strip()] = self._coerce(val)
                else: sub = key.strip(); result[section].setdefault(sub, {})
            elif indent == 4 and section and sub and ":" in stripped:
                key, _, val = stripped.partition(":"); val = val.strip().strip('"').strip("'")
                if isinstance(result[section].get(sub), dict):
                    result[section][sub][key.strip()] = self._coerce(val)
        return result if result else None

    def _coerce(self, val):
        if val in ("null", "~"): return None
        if val.startswith("[") and val.endswith("]"):
            return [self._coerce(s.strip()) for s in val[1:-1].split(",") if s.strip()]
        try: return int(val)
        except ValueError: pass
        try: return float(val)
        except ValueError: pass
        if val.lower() in ("true", "yes"): return True
        if val.lower() in ("false", "no"): return False
        return val

    def _hardcoded_default(self):
        return {"name": "default", "description": "Built-in default routing",
            "defaults": {"perception_hz": 2, "action_hz": 20, "max_steps": 200,
                         "termination": ["step_limit"], "position_threshold": 0.05},
            "routes": {
                "easy": {"perception_hz": 1, "action_hz": 20, "max_steps": 100,
                         "termination": ["position_threshold", "step_limit"], "position_threshold": 0.05},
                "medium": {"perception_hz": 2, "action_hz": 20, "max_steps": 200,
                           "termination": ["vlm_check", "step_limit"]},
                "hard": {"perception_hz": 5, "action_hz": 50, "max_steps": 500,
                         "termination": ["vlm_check", "gripper_state", "step_limit"]}}}

    def classify_difficulty(self, instruction, objects=None):
        """Keyword heuristic: known object + simple verb -> easy, compound -> hard."""
        instr = instruction.lower()
        known = {"apple", "plate", "mug", "cup", "fork", "napkin", "bottle", "box", "ball"}
        verbs = {"pick", "put", "place", "move", "grab", "grasp", "release", "drop", "lift"}
        compound = {"then", "after", "while", "and then", "followed by", "carefully",
                    "stack", "sort", "arrange", "assemble", "pour"}
        if any(m in instr for m in compound): return "hard"
        found_obj = any(obj in instr for obj in known)
        if objects: found_obj = found_obj or any(o.lower() in instr for o in objects)
        found_verb = any(v in instr for v in verbs)
        if found_obj and found_verb: return "easy"
        if found_obj or found_verb: return "medium"
        return "hard"

    def get_config(self, route_name="default", difficulty="medium"):
        """Merge defaults + difficulty-specific config for a named route."""
        cfg = self.configs.get(route_name, self.configs.get("default", self._hardcoded_default()))
        merged = dict(cfg.get("defaults", {}))
        diff_cfg = dict(cfg.get("routes", {}).get(difficulty, cfg.get("routes", {}).get("medium", {})))
        merged.update({k: v for k, v in diff_cfg.items() if v is not None})
        merged["_name"] = f"{cfg.get('name', route_name)}/{difficulty}"
        merged["_description"] = diff_cfg.get("description", cfg.get("description", ""))
        return merged

    def list_routes(self):
        if not self.configs: return "No routes configured."
        lines = ["Available routes:"]
        for name, cfg in self.configs.items():
            lines.append(f"\n  [{name}] {cfg.get('description', '')}")
            for diff in ("easy", "medium", "hard"):
                r = cfg.get("routes", {}).get(diff)
                if not r: continue
                d = cfg.get("defaults", {})
                ahz = r.get("action_hz", d.get("action_hz", "?"))
                phz = r.get("perception_hz", d.get("perception_hz", "?"))
                term = r.get("termination", d.get("termination", []))
                lora = r.get("vla_lora", "none")
                lines.append(f"    {diff}: action@{ahz}Hz, perception@{phz}Hz, term={term}, lora={lora}")
                rdesc = r.get("description", "")
                if rdesc: lines.append(f"           {rdesc}")
        return "\n".join(lines)


ROUTING = RoutingConfig(ROUTES_DIR)


# ============================================================
# VLA Adapters -- abstraction for different VLA backends
# ============================================================

class BaseVLAAdapter(ABC):
    """Abstract interface for VLA models. predict() returns list of action deltas."""
    @abstractmethod
    def predict(self, observation: dict, instruction: str) -> list: ...
    @abstractmethod
    def reset(self): ...
    def get_action_horizon(self) -> int: return 1
    def activate(self, lora_name=None) -> str: return "activated"
    def deactivate(self) -> str: return "deactivated"
    def health_check(self) -> bool: return True


class MockVLAAdapter(BaseVLAAdapter):
    """Generates synthetic deltas toward target position (action chunking)."""
    def __init__(self):
        self.target_pos = None; self.step_size = 0.01; self.action_horizon = 10
        self._active = False; self._lora = None

    def activate(self, lora_name=None):
        self._active = True; self._lora = lora_name
        return f"MockVLA activated{f' (lora={lora_name})' if lora_name else ''}"

    def deactivate(self):
        self._active = False; self._lora = None; return "MockVLA deactivated (idle)"

    def predict(self, observation, instruction):
        ee = observation.get("ee_pos", [0, 0, 0]); target = self.target_pos or ee
        actions = []; current = list(ee)
        for _ in range(self.action_horizon):
            delta = [(t - c) for t, c in zip(target, current)]
            dist = math.sqrt(sum(d**2 for d in delta))
            if dist < 0.001:
                actions.append([0.0, 0.0, 0.0])
            else:
                scale = min(self.step_size / dist, 1.0)
                step = [d * scale for d in delta]
                actions.append(step)
                current = [c + s for c, s in zip(current, step)]
        return actions

    def reset(self): self.target_pos = None
    def get_action_horizon(self): return self.action_horizon
    def health_check(self): return True


class HTTPVLAAdapter(BaseVLAAdapter):
    """Wraps existing POST VLA_URL/predict pattern. Single-step per call."""
    def __init__(self, url): self.url = url.rstrip("/")

    def activate(self, lora_name=None):
        try: requests.post(f"{self.url}/activate", json={"lora": lora_name}, timeout=10); return f"HTTPVLAAdapter activated (lora={lora_name})"
        except Exception: return "HTTPVLAAdapter activate failed (continuing)"

    def deactivate(self):
        try: requests.post(f"{self.url}/deactivate", timeout=5)
        except Exception: pass
        return "deactivated"

    def predict(self, observation, instruction):
        resp = requests.post(f"{self.url}/predict", json={"observation": observation, "instruction": instruction}, timeout=30)
        action = resp.json().get("action", {})
        if isinstance(action, dict): return [action.get("pos", [0, 0, 0])]
        if isinstance(action, list) and action and isinstance(action[0], list): return action
        return [action if isinstance(action, list) else [0, 0, 0]]

    def reset(self):
        try: requests.post(f"{self.url}/reset", timeout=5)
        except Exception: pass

    def get_action_horizon(self): return 1

    def health_check(self):
        try: return requests.get(f"{self.url}/health", timeout=5).ok
        except Exception: return False


class GR00TStub(BaseVLAAdapter):
    """Stub for NVIDIA GR00T N1.6. Protocol: ZMQ REQ/REP :5555, msgpack,
    action_horizon=16. In demo mode, delegates to MockVLAAdapter."""
    def __init__(self): self._mock = MockVLAAdapter(); self._mock.action_horizon = 16
    def predict(self, observation, instruction): return self._mock.predict(observation, instruction)
    def reset(self): self._mock.reset()
    def get_action_horizon(self): return 16
    def activate(self, lora_name=None):
        self._mock.target_pos = None; return f"GR00TStub activated (lora={lora_name}, zmq://localhost:5555)"
    def deactivate(self): return "GR00TStub deactivated (zmq idle)"


class OpenPIStub(BaseVLAAdapter):
    """Stub for OpenPI (pi0). Protocol: WebSocket :8001, JSON,
    action_horizon=10 @50Hz. In demo mode, delegates to MockVLAAdapter."""
    def __init__(self): self._mock = MockVLAAdapter(); self._mock.action_horizon = 10
    def predict(self, observation, instruction): return self._mock.predict(observation, instruction)
    def reset(self): self._mock.reset()
    def get_action_horizon(self): return 10
    def activate(self, lora_name=None):
        self._mock.target_pos = None; return f"OpenPIStub activated (lora={lora_name}, ws://localhost:8001)"
    def deactivate(self): return "OpenPIStub deactivated (ws idle)"


def create_vla_adapter():
    """Factory: select VLA adapter based on VLA_BACKEND env var."""
    b = VLA_BACKEND.lower()
    if b == "http" and VLA_URL: return HTTPVLAAdapter(VLA_URL)
    if b == "groot": return GR00TStub()
    if b == "openpi": return OpenPIStub()
    return MockVLAAdapter()


# ============================================================
# Termination Strategies -- VLA cannot self-terminate
# ============================================================

class BaseTerminator(ABC):
    """Abstract termination condition. VLAs run open-loop; external
    terminators monitor state and fire when conditions are met."""
    @abstractmethod
    def should_stop(self, state: dict, target: dict) -> tuple: ...
    def reset(self): pass


class StepLimitTerminator(BaseTerminator):
    """Stop after max_steps VLA iterations (safety fallback)."""
    def __init__(self, max_steps=200): self.max_steps = max_steps
    def should_stop(self, state, target):
        steps = state.get("vla_step_count", 0)
        return (True, f"step_limit: {steps} >= {self.max_steps}") if steps >= self.max_steps else (False, "")


class PositionThresholdTerminator(BaseTerminator):
    """Stop when end-effector is within threshold of target position."""
    def __init__(self, threshold=0.05): self.threshold = threshold
    def should_stop(self, state, target):
        ee = state.get("ee_pos", [0, 0, 0]); tgt = target.get("pos", ee)
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(ee, tgt)))
        return (True, f"position_threshold: {dist:.3f}m < {self.threshold}m") if dist < self.threshold else (False, "")


class GripperStateTerminator(BaseTerminator):
    """Stop when gripper matches expected state and grasps/releases."""
    def __init__(self, expected="close"): self.expected = expected
    def should_stop(self, state, target):
        if state.get("gripper") == self.expected:
            held = state.get("holding")
            if self.expected == "close" and held: return True, f"gripper_state: closed, grasped '{held}'"
            if self.expected == "open" and not held: return True, "gripper_state: opened, released object"
        return False, ""


class VLMTerminator(BaseTerminator):
    """Mock: checks scene keywords. Real: would call VLM for completion check."""
    def __init__(self, question="Is the subtask complete?"): self.question = question; self.last_check = ""
    def should_stop(self, state, target):
        scene = state.get("scene", ""); target_name = target.get("name", "")
        if target_name and scene:
            sl = scene.lower()
            if target_name.lower() in sl and any(w in sl for w in ("held", "grasped", "placed", "on plate")):
                self.last_check = f"{target_name} appears done"
                return True, f"vlm_check: {self.last_check}"
        return False, ""


class CompositeTerminator(BaseTerminator):
    """Combine terminators: mode='any' (OR) or 'all' (AND)."""
    def __init__(self, terminators, mode="any"): self.terminators = terminators; self.mode = mode
    def should_stop(self, state, target):
        results = [t.should_stop(state, target) for t in self.terminators]
        if self.mode == "any":
            for stopped, reason in results:
                if stopped: return True, reason
            return False, ""
        if all(r[0] for r in results): return True, " + ".join(r[1] for r in results if r[1])
        return False, ""
    def reset(self):
        for t in self.terminators: t.reset()


TERMINATOR_FACTORIES = {
    "step_limit": lambda cfg: StepLimitTerminator(cfg.get("max_steps", 200)),
    "position_threshold": lambda cfg: PositionThresholdTerminator(cfg.get("position_threshold", 0.05)),
    "gripper_state": lambda cfg: GripperStateTerminator(cfg.get("expected_gripper", "close")),
    "vlm_check": lambda cfg: VLMTerminator(cfg.get("vlm_question", "Is the subtask complete?")),
}

def create_terminators(config: dict) -> CompositeTerminator:
    names = config.get("termination", ["step_limit"])
    if isinstance(names, str): names = [names]
    terms = [TERMINATOR_FACTORIES[n](config) for n in names if n in TERMINATOR_FACTORIES]
    if not terms: terms = [StepLimitTerminator(200)]
    return CompositeTerminator(terms)


# ============================================================
# SharedState -- thread-safe coordination between loops
# ============================================================

class SharedState:
    """Thread-safe state shared by LLM, VLA control, and VLM perception loops."""
    def __init__(self):
        self._lock = threading.Lock()
        self.scene = ""; self.ee_pos = [0.3, 0.0, 0.95]; self.gripper = "open"
        self.holding = None; self.objects = {}; self.target = {}; self.instruction = ""
        self.vla_step_count = 0; self.vlm_check_count = 0
        self.termination_signal = threading.Event(); self.subtask_result = None
        self.active_route = {}

    def get_state_dict(self):
        with self._lock:
            return {"ee_pos": list(self.ee_pos), "gripper": self.gripper,
                    "holding": self.holding, "objects": {k: dict(v) for k, v in self.objects.items()},
                    "scene": self.scene, "vla_step_count": self.vla_step_count}

    def update_from_env(self, env):
        with self._lock:
            self.ee_pos = list(env.ee_pos); self.gripper = env.gripper
            self.holding = env.holding; self.objects = {k: dict(v) for k, v in env.objects.items()}

    def signal_done(self, reason):
        with self._lock: self.subtask_result = reason
        self.termination_signal.set()

    def reset_for_subtask(self, target, instruction):
        with self._lock:
            self.target = target; self.instruction = instruction
            self.vla_step_count = 0; self.vlm_check_count = 0; self.subtask_result = None
        self.termination_signal.clear()


# ============================================================
# LoopManager -- VLA control + VLM perception loops
# ============================================================

class LoopManager:
    """Launches concurrent VLA and VLM loops for subtask execution.
    The LLM calls start_subtask() once, then loops run until a terminator fires."""
    def __init__(self, env, shared, vla_adapter, routing, bg_manager):
        self.env = env; self.shared = shared; self.vla = vla_adapter
        self.routing = routing; self.bg = bg_manager
        self._control_thread = None; self._perception_thread = None
        self.stats = {"subtasks": 0, "total_vla_steps": 0, "total_time": 0.0}

    def start_subtask(self, instruction, target, route_name="default", difficulty="medium"):
        # Wait for any previous subtask
        if self._control_thread and self._control_thread.is_alive():
            self.shared.termination_signal.wait(timeout=5.0)
        # 1. Routing config
        config = self.routing.get_config(route_name, difficulty)
        action_hz = config.get("action_hz", 20); perception_hz = config.get("perception_hz", 2)
        lora = config.get("vla_lora")
        # 2. Reset state
        self.shared.reset_for_subtask(target, instruction)
        self.shared.update_from_env(self.env); self.shared.active_route = config
        # 3. Activate VLA
        self.vla.activate(lora)
        if isinstance(self.vla, MockVLAAdapter): self.vla.target_pos = target.get("pos")
        elif isinstance(self.vla, (GR00TStub, OpenPIStub)): self.vla._mock.target_pos = target.get("pos")
        # 4. Terminators + subtask ID
        terminator = create_terminators(config)
        sid = str(uuid.uuid4())[:8]; start_time = time.time()
        # 5. Launch threads
        self._control_thread = threading.Thread(
            target=self._control_loop, args=(action_hz, terminator, sid, start_time), daemon=True)
        if perception_hz and perception_hz > 0:
            self._perception_thread = threading.Thread(
                target=self._perception_loop, args=(perception_hz, terminator), daemon=True)
            self._perception_thread.start()
        self._control_thread.start()
        route_desc = f"route={route_name}/{difficulty}, action@{action_hz}Hz, perception@{perception_hz}Hz"
        return f"Subtask {sid} started: {instruction} ({route_desc}{f', lora={lora}' if lora else ''})"

    def _control_loop(self, hz, terminator, sid, start_time):
        """VLA control loop: predict -> step_delta -> check terminators at action_hz."""
        interval = 1.0 / max(hz, 1)
        while not self.shared.termination_signal.is_set():
            loop_start = time.time()
            state = self.shared.get_state_dict()
            obs = {"ee_pos": state["ee_pos"], "gripper": state["gripper"],
                   "holding": state["holding"], "objects": state["objects"]}
            actions = self.vla.predict(obs, self.shared.instruction)
            with self.env._lock:
                for action_delta in actions:
                    if self.shared.termination_signal.is_set(): break
                    if isinstance(action_delta, list) and len(action_delta) >= 3:
                        self.env.step_delta(action_delta)
                    with self.shared._lock: self.shared.vla_step_count += 1
                    self.shared.ee_pos = list(self.env.ee_pos)
                    self.shared.gripper = self.env.gripper
                    self.shared.holding = self.env.holding
                    self.shared.objects = {k: dict(v) for k, v in self.env.objects.items()}
                    # Check terminators after each step
                    cs = self.shared.get_state_dict()
                    stopped, reason = terminator.should_stop(cs, self.shared.target)
                    if stopped:
                        elapsed = time.time() - start_time; steps = self.shared.vla_step_count
                        achieved_hz = steps / elapsed if elapsed > 0 else 0
                        result = (f"[subtask:{sid}] completed after {steps} steps ({elapsed:.1f}s)\n"
                                  f"  termination: {reason}\n"
                                  f"  final: ee=[{cs['ee_pos'][0]:.3f}, {cs['ee_pos'][1]:.3f}, "
                                  f"{cs['ee_pos'][2]:.3f}], gripper={cs['gripper']}, "
                                  f"holding={cs['holding'] or 'nothing'}\n"
                                  f"  route: {self.shared.active_route.get('_name', 'default')}, "
                                  f"achieved action_hz={achieved_hz:.1f}")
                        self.shared.signal_done(result)
                        with self.bg._lock:
                            self.bg._queue.append({"job_id": sid, "status": "completed",
                                "description": f"subtask: {self.shared.instruction}", "result": result})
                        self.stats["subtasks"] += 1; self.stats["total_vla_steps"] += steps
                        self.stats["total_time"] += elapsed; self.vla.deactivate(); return
            sleep_t = interval - (time.time() - loop_start)
            if sleep_t > 0: time.sleep(sleep_t)
        self.vla.deactivate()

    def _perception_loop(self, hz, terminator):
        """VLM perception loop: look -> update scene -> check VLM terminators."""
        interval = 1.0 / max(hz, 0.1)
        while not self.shared.termination_signal.is_set():
            scene = self.env.look()
            with self.shared._lock: self.shared.scene = scene; self.shared.vlm_check_count += 1
            state = self.shared.get_state_dict()
            stopped, reason = terminator.should_stop(state, self.shared.target)
            if stopped: self.shared.signal_done(reason); return
            time.sleep(interval)

    def check_status(self):
        state = self.shared.get_state_dict()
        ctrl = "running" if (self._control_thread and self._control_thread.is_alive()) else "idle"
        perc = "running" if (self._perception_thread and self._perception_thread.is_alive()) else "idle"
        lines = [f"Control loop: {ctrl}", f"Perception loop: {perc}",
                 f"VLA steps: {state['vla_step_count']}, VLM checks: {self.shared.vlm_check_count}",
                 f"Robot: ee=[{state['ee_pos'][0]:.3f}, {state['ee_pos'][1]:.3f}, {state['ee_pos'][2]:.3f}], "
                 f"gripper={state['gripper']}, holding={state['holding'] or 'nothing'}",
                 f"Instruction: {self.shared.instruction or '(none)'}"]
        if self.shared.active_route:
            n = self.shared.active_route.get("_name", "?")
            lines.append(f"Route: {n}, action@{self.shared.active_route.get('action_hz', '?')}Hz, "
                         f"perception@{self.shared.active_route.get('perception_hz', '?')}Hz")
        return "\n".join(lines)

    def wait_for_completion(self, timeout=30.0):
        self.shared.termination_signal.wait(timeout=timeout)
        return self.shared.subtask_result or f"timeout after {timeout}s"

    def stop(self, reason="manual"):
        self.shared.signal_done(f"stopped: {reason}")
        if self._control_thread and self._control_thread.is_alive(): self._control_thread.join(timeout=2.0)
        if self._perception_thread and self._perception_thread.is_alive(): self._perception_thread.join(timeout=2.0)
        return f"Loops stopped: {reason}"

    def get_stats(self):
        avg = self.stats["total_vla_steps"] / self.stats["total_time"] if self.stats["total_time"] > 0 else 0
        return (f"Subtasks: {self.stats['subtasks']}, Total VLA steps: {self.stats['total_vla_steps']}, "
                f"Total time: {self.stats['total_time']:.1f}s, Average Hz: {avg:.1f}")


# ============================================================
# Mock Environment (thread-safe)
# ============================================================

GRASP_DISTANCE = 0.08

class MockRobotEnv:
    def __init__(self):
        self._lock = threading.Lock(); self.reset()

    def reset(self):
        self.objects = {
            "red apple": {"pos": [0.45, 0.12, 0.82], "on": "counter", "shape": "round-flat"},
            "white plate": {"pos": [0.45, -0.15, 0.80], "on": "counter", "shape": "flat"},
            "blue mug": {"pos": [0.50, 0.30, 0.82], "on": "counter", "shape": "tall-handle"},
            "fork": {"pos": [0.55, -0.20, 0.81], "on": "counter", "shape": "flat-long"},
            "napkin": {"pos": [0.60, 0.00, 0.81], "on": "counter", "shape": "flat"},
        }
        self.ee_pos = [0.30, 0.0, 0.95]; self.gripper = "open"; self.holding = None

    def _dist(self, a, b): return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

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
            return "Error: 'open' or 'close'."

    def step_delta(self, delta):
        """Apply incremental position change. NOTE: caller must hold self._lock."""
        for i in range(min(len(delta), 3)): self.ee_pos[i] += delta[i]
        self.ee_pos[0] = max(0.0, min(0.8, self.ee_pos[0]))
        self.ee_pos[1] = max(-0.6, min(0.6, self.ee_pos[1]))
        self.ee_pos[2] = max(0.6, min(1.2, self.ee_pos[2]))
        if self.holding and self.holding in self.objects:
            self.objects[self.holding]["pos"] = list(self.ee_pos)
        if self.gripper == "close" and self.holding is None:
            for n, o in self.objects.items():
                if self._dist(self.ee_pos, o["pos"]) < GRASP_DISTANCE:
                    self.holding = n; break

    def get_observation(self):
        with self._lock:
            return {"ee_pos": list(self.ee_pos), "gripper": self.gripper,
                    "holding": self.holding, "objects": {k: dict(v) for k, v in self.objects.items()}}

    def snapshot(self):
        with self._lock:
            return {"objects": {k: dict(v) for k, v in self.objects.items()},
                    "ee_pos": list(self.ee_pos), "gripper": self.gripper, "holding": self.holding}

    def restore(self, snap):
        with self._lock:
            self.objects = {k: dict(v) for k, v in snap["objects"].items()}
            self.ee_pos = list(snap["ee_pos"]); self.gripper = snap["gripper"]; self.holding = snap["holding"]


mock_env = MockRobotEnv()


# ============================================================
# SafetyManager -- workspace limits
# ============================================================

class SafetyManager:
    def __init__(self):
        self.workspace_min = [0.0, -0.6, 0.60]; self.workspace_max = [0.8, 0.6, 1.20]
        self.max_speed = 0.5; self.violations = []

    def check_move(self, current_pos, target_pos):
        for i, axis in enumerate(["x", "y", "z"]):
            if target_pos[i] < self.workspace_min[i] or target_pos[i] > self.workspace_max[i]:
                msg = f"Out of workspace: {axis}={target_pos[i]:.2f} (bounds [{self.workspace_min[i]}, {self.workspace_max[i]}])"
                self.violations.append({"type": "workspace", "msg": msg, "ts": time.time()}); return False, msg
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(current_pos, target_pos)))
        if dist > self.max_speed:
            msg = f"Move too large: {dist:.2f}m > {self.max_speed}m limit"
            self.violations.append({"type": "speed", "msg": msg, "ts": time.time()}); return False, msg
        return True, "ok"

    def status(self):
        lines = [f"Safety: workspace=[{self.workspace_min}, {self.workspace_max}], max_speed={self.max_speed}m"]
        if self.violations:
            lines.append(f"  violations: {len(self.violations)}")
            for v in self.violations[-3:]: lines.append(f"    [{v['type']}] {v['msg']}")
        else: lines.append("  violations: 0")
        return "\n".join(lines)

SAFETY = SafetyManager()


# ============================================================
# BackgroundManager -- threaded execution + notification queue
# ============================================================

class BackgroundManager:
    def __init__(self): self.jobs = {}; self._queue = []; self._lock = threading.Lock()

    def run(self, job_id, func, args, description):
        self.jobs[job_id] = {"status": "running", "description": description, "result": None}
        threading.Thread(target=self._execute, args=(job_id, func, args), daemon=True).start()
        return f"Background job {job_id} started: {description}"

    def _execute(self, job_id, func, args):
        try:
            result = func(*args); self.jobs[job_id]["status"] = "completed"; self.jobs[job_id]["result"] = result
        except Exception as e:
            result = f"Error: {e}"; self.jobs[job_id]["status"] = "error"; self.jobs[job_id]["result"] = result
        with self._lock:
            self._queue.append({"job_id": job_id, "status": self.jobs[job_id]["status"],
                                "description": self.jobs[job_id]["description"], "result": str(result)[:800]})

    def check(self, job_id=None):
        if job_id:
            j = self.jobs.get(job_id)
            return f"[{j['status']}] {j['description']}\n{j.get('result') or '(running)'}" if j else f"Unknown job: {job_id}"
        lines = [f"  {jid}: [{j['status']}] {j['description']}" for jid, j in self.jobs.items()]
        return "\n".join(lines) if lines else "No background jobs."

    def drain(self):
        with self._lock: notifs = list(self._queue); self._queue.clear()
        return notifs

BG = BackgroundManager()


# ============================================================
# TaskManager (from r07)
# ============================================================

class TaskManager:
    def __init__(self, d): self.dir = d; d.mkdir(parents=True, exist_ok=True); self._next_id = self._max_id() + 1
    def _max_id(self):
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0
    def _load(self, tid):
        p = self.dir / f"task_{tid}.json"
        if not p.exists(): raise ValueError(f"Task {tid} not found")
        return json.loads(p.read_text())
    def _save(self, t): (self.dir / f"task_{t['id']}.json").write_text(json.dumps(t, indent=2))
    def create(self, subject, description=""):
        t = {"id": self._next_id, "subject": subject, "description": description,
             "status": "pending", "blockedBy": [], "blocks": [], "owner": ""}
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
            text = f.read_text(); m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
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
# Compression (from r06), with loop stats
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
    ts = TASKS.list_all(); loop_stats = LOOPS.get_stats() if LOOPS else "No loop stats."
    conv = json.dumps(msgs, default=str)[:80000]
    resp = client.messages.create(model=MODEL, messages=[{"role": "user", "content":
        f"Summarize robot conversation. Include: original task, progress, strategies, loop results.\n\n"
        f"Robot: {rs}\nTasks:\n{ts}\nLoop stats: {loop_stats}\n\nConversation:\n{conv}"}], max_tokens=2000)
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {p.name}]\n\n{resp.content[0].text}\n\n"
            f"Robot: {rs}\nLoop stats: {loop_stats}\n\nCall task_list for current progress."},
        {"role": "assistant", "content": "Understood. Checking task board and loop status."},
    ]


# ============================================================
# Real mode helpers
# ============================================================

def real_look(q=""):
    resp = requests.get(f"{SIM_URL}/render", timeout=5)
    img_b64 = base64.b64encode(resp.content).decode()
    r = requests.post(f"{VLM_URL}/analyze", json={"image": img_b64, "prompt": q or "Describe the scene."}, timeout=30).json()
    return f"{r.get('analysis', '')} [{r.get('inference_ms', 0):.0f}ms]"

def run_look(q=""):
    if SIM_URL and VLM_URL:
        try: return real_look(q)
        except Exception as e: return f"[Error: {e}]\n{mock_env.look(q)}"
    return mock_env.look(q)

def run_move(target="", position=None):
    if target:
        m = mock_env._find(target)
        if m:
            ok, reason = SAFETY.check_move(mock_env.ee_pos, list(mock_env.objects[m]["pos"]))
            if not ok: return f"SAFETY BLOCK: {reason}"
    elif position and len(position) >= 3:
        ok, reason = SAFETY.check_move(mock_env.ee_pos, position[:3])
        if not ok: return f"SAFETY BLOCK: {reason}"
    return mock_env.move(target, position)

def run_grasp(action): return mock_env.grasp(action)

# Perception subagent
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
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": run_look(b.input.get("question", ""))})
        sub.append({"role": "user", "content": results})
    return "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"

# Background operations
def run_bg_look(question=""):
    jid = str(uuid.uuid4())[:8]
    def _bg(): time.sleep(1); return run_look(question)
    return BG.run(jid, _bg, (), f"look: {question or '(monitor)'}")

def run_bg_act(instruction=""):
    jid = str(uuid.uuid4())[:8]
    def _bg():
        time.sleep(2); instr = instruction.lower()
        if "move" in instr:
            for obj_name in mock_env.objects:
                if obj_name.split()[-1] in instr or obj_name in instr: return mock_env.move(target=obj_name)
            return "Could not determine move target."
        elif "close" in instr or "grasp" in instr: return mock_env.grasp("close")
        elif "open" in instr or "release" in instr: return mock_env.grasp("open")
        return f"Executed: {instruction}"
    return BG.run(jid, _bg, (), f"act: {instruction[:60]}")


# ============================================================
# Global instances
# ============================================================

SHARED = SharedState()
VLA = create_vla_adapter()
LOOPS = LoopManager(mock_env, SHARED, VLA, ROUTING, BG)


# ============================================================
# System prompt
# ============================================================

SYSTEM = f"""\
You are a robot agent controlling a Unitree G1 humanoid robot.

MULTI-FREQUENCY CONTROL:
You think at 0.1Hz. The VLA control loop runs at 2-50Hz. The VLM
perception loop runs at 1-5Hz. Use start_subtask to launch fast
concurrent loops instead of calling move/grasp directly.

WORKFLOW:
1. classify_task -- determine difficulty (easy/medium/hard)
2. start_subtask -- launch VLA+VLM loops for each sub-action
3. wait_subtask -- wait for automatic termination
4. Repeat for each subtask
5. look -- verify final result

ROUTING:
- easy: VLA only, fast, no LLM replanning needed
- medium: VLA + periodic VLM scene checks
- hard: VLA + VLM + LLM evaluation after completion

FALLBACK: Use move/grasp directly for simple one-shot adjustments.

VLA ADAPTER: {VLA_BACKEND} (supports: mock, http, groot, openpi)

Routes:
{ROUTING.list_routes()}

Skills: {SKILL_LOADER.get_descriptions()}"""


# ============================================================
# Tool definitions + dispatch
# ============================================================

def _set_route(route, difficulty):
    config = ROUTING.get_config(route, difficulty)
    lines = [f"Route: {config.get('_name', '?')}",
             f"  action_hz: {config.get('action_hz', '?')}", f"  perception_hz: {config.get('perception_hz', '?')}",
             f"  max_steps: {config.get('max_steps', '?')}", f"  termination: {config.get('termination', [])}",
             f"  vla_lora: {config.get('vla_lora', 'none')}", f"  position_threshold: {config.get('position_threshold', '?')}"]
    desc = config.get("_description", "")
    if desc: lines.append(f"  description: {desc}")
    return "\n".join(lines)

def _classify_task(instruction):
    difficulty = ROUTING.classify_difficulty(instruction, list(mock_env.objects.keys()))
    config = ROUTING.get_config("default", difficulty)
    lines = [f"Instruction: \"{instruction}\"", f"Difficulty: {difficulty}",
             f"Recommended route: default/{difficulty}",
             f"  action_hz: {config.get('action_hz', '?')}", f"  perception_hz: {config.get('perception_hz', '?')}",
             f"  termination: {config.get('termination', [])}", f"  max_steps: {config.get('max_steps', '?')}"]
    lora = config.get("vla_lora")
    if lora: lines.append(f"  vla_lora: {lora}")
    return "\n".join(lines)

TOOL_HANDLERS = {
    "look":             lambda **kw: run_look(kw.get("question", "")),
    "perceive":         lambda **kw: run_perceive(kw["goal"]),
    "move":             lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp":            lambda **kw: run_grasp(kw["action"]),
    "bg_look":          lambda **kw: run_bg_look(kw.get("question", "")),
    "bg_act":           lambda **kw: run_bg_act(kw.get("instruction", "")),
    "check_background": lambda **kw: BG.check(kw.get("job_id")),
    "task_create":      lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update":      lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":        lambda **kw: TASKS.list_all(),
    "task_get":         lambda **kw: TASKS.get(kw["task_id"]),
    "load_skill":       lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact":          lambda **kw: "Compressing...",
    # Multi-frequency routing (NEW in r13)
    "start_subtask":    lambda **kw: LOOPS.start_subtask(kw["instruction"], kw.get("target", {}), kw.get("route", "default"), kw.get("difficulty", "medium")),
    "check_loops":      lambda **kw: LOOPS.check_status(),
    "wait_subtask":     lambda **kw: LOOPS.wait_for_completion(kw.get("timeout", 30)),
    "stop_loops":       lambda **kw: LOOPS.stop(kw.get("reason", "manual")),
    "set_route":        lambda **kw: _set_route(kw.get("route", "default"), kw.get("difficulty", "medium")),
    "list_routes":      lambda **kw: ROUTING.list_routes(),
    "classify_task":    lambda **kw: _classify_task(kw["instruction"]),
    "loop_stats":       lambda **kw: LOOPS.get_stats(),
}

TOOLS = [
    # Perception
    {"name": "look", "description": "Quick observation (blocking).",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "perceive", "description": "Thorough scene analysis (blocking, spawns subagent).",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
    # Action (fallback -- prefer start_subtask)
    {"name": "move", "description": "Move end-effector directly (blocking, safety-checked). Prefer start_subtask.",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "position": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    {"name": "grasp", "description": "Gripper control (blocking). Prefer start_subtask.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    # Multi-frequency routing (NEW)
    {"name": "start_subtask", "description": "Launch VLA+VLM loops for a subtask. Runs until terminator fires.",
     "input_schema": {"type": "object", "properties": {
         "instruction": {"type": "string", "description": "What the robot should do."},
         "target": {"type": "object", "description": "Target with 'name' and 'pos' [x,y,z].",
                    "properties": {"name": {"type": "string"}, "pos": {"type": "array", "items": {"type": "number"}}}, "required": ["name", "pos"]},
         "route": {"type": "string", "description": "Routing config name."},
         "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
     }, "required": ["instruction", "target"]}},
    {"name": "check_loops", "description": "Check current VLA/VLM loop status and stats.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "wait_subtask", "description": "Block until current subtask completes or times out.",
     "input_schema": {"type": "object", "properties": {"timeout": {"type": "number", "description": "Max wait seconds (default 30)."}}}},
    {"name": "stop_loops", "description": "Force-stop all running loops.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}}},
    {"name": "set_route", "description": "Preview routing config for a route/difficulty combo.",
     "input_schema": {"type": "object", "properties": {"route": {"type": "string"}, "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]}}, "required": ["route", "difficulty"]}},
    {"name": "list_routes", "description": "List all available routing configurations.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "classify_task", "description": "Classify instruction difficulty and suggest routing.",
     "input_schema": {"type": "object", "properties": {"instruction": {"type": "string"}}, "required": ["instruction"]}},
    {"name": "loop_stats", "description": "Performance stats: subtasks completed, Hz achieved, total time.",
     "input_schema": {"type": "object", "properties": {}}},
    # Background
    {"name": "bg_look", "description": "Non-blocking observation. Result delivered next turn.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": []}},
    {"name": "bg_act", "description": "Non-blocking action. Runs in background.",
     "input_schema": {"type": "object", "properties": {"instruction": {"type": "string"}}, "required": ["instruction"]}},
    {"name": "check_background", "description": "Check background job status.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}}},
    # Tasks
    {"name": "task_create", "description": "Create task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update task status/deps.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get task details.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    # Utility
    {"name": "load_skill", "description": "Load manipulation skill.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compact", "description": "Compress context.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
]


# ============================================================
# Agent loop -- drain loop-results + background, compression
# ============================================================

def agent_loop(messages: list):
    while True:
        # Drain background + loop-results notifications
        notifs = BG.drain()
        if notifs and messages:
            loop_notifs = [n for n in notifs if n["description"].startswith("subtask:")]
            bg_notifs = [n for n in notifs if not n["description"].startswith("subtask:")]
            if loop_notifs:
                loop_text = "\n".join(f"[loop:{n['job_id']}] {n['status']}: {n['description']}\n  {n['result']}" for n in loop_notifs)
                print(f"\033[95m[loop-results] {len(loop_notifs)} subtask(s) completed\033[0m")
                messages.append({"role": "user", "content": f"<loop-results>\n{loop_text}\n</loop-results>"})
                messages.append({"role": "assistant", "content": "Noted loop results."})
            if bg_notifs:
                bg_text = "\n".join(f"[bg:{n['job_id']}] {n['status']}: {n['description']}\n  {n['result']}" for n in bg_notifs)
                print(f"\033[32m[bg notification] {len(bg_notifs)} job(s) completed\033[0m")
                messages.append({"role": "user", "content": f"<background-results>\n{bg_text}\n</background-results>"})
                messages.append({"role": "assistant", "content": "Noted background results."})
        # Compression
        micro_compact(messages); est = estimate_tokens(messages)
        if est > THRESHOLD:
            print(f"\033[91m[auto_compact] ~{est // 4} tokens\033[0m"); messages[:] = auto_compact(messages)

        response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=4096)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use": return

        results = []; manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact": manual_compact = True; output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try: output = handler(**block.input) if handler else f"Unknown: {block.name}"
                    except Exception as e: output = f"Error: {e}"
                # Color-coded printing
                if block.name in ("start_subtask", "wait_subtask", "stop_loops"):
                    print(f"\033[95m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:600]}\033[0m")
                elif block.name in ("check_loops", "loop_stats", "classify_task", "set_route", "list_routes"):
                    print(f"\033[94m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                elif block.name.startswith("task_"):
                    print(f"\033[35m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:400]}\033[0m")
                elif block.name.startswith("bg_"):
                    print(f"\033[32m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:200]}\033[0m")
                elif block.name in ("load_skill", "compact"): print(f"\033[36m[{block.name}]\033[0m")
                elif block.name == "perceive":
                    print(f"\033[34m[perceive]\033[0m"); print(f"\033[90m{str(output)[:600]}\033[0m")
                else:
                    print(f"\033[33m[{block.name}] {json.dumps(block.input, ensure_ascii=False)}\033[0m")
                    print(f"\033[90m{str(output)[:500]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
        if manual_compact: messages[:] = auto_compact(messages)


# ============================================================
# REPL
# ============================================================

if __name__ == "__main__":
    mode = "REAL" if (SIM_URL and VLA_URL) else "MOCK"
    print(f"\033[32m[r13] Multi-Frequency Dynamic Routing  |  {mode}  |  VLA={VLA_BACKEND.upper()}\033[0m")
    print(f"\033[90mRouting: start_subtask, wait_subtask, check_loops, stop_loops, classify_task, set_route, list_routes, loop_stats\033[0m")
    print(f"\033[90mFallback: look, move, grasp, perceive | Async: bg_look, bg_act | Tasks: task_create, task_update\033[0m")
    print(f"\033[90mREPL: reset, tasks, bg, skills, loops, routes, stats, tokens, /compact, safety, q\033[0m\n")

    history = []
    while True:
        try: query = input("\033[36mr13 >> \033[0m")
        except (EOFError, KeyboardInterrupt): break
        cmd = query.strip().lower()
        if cmd in ("q", "exit", ""): break
        if cmd == "reset":
            mock_env.reset(); TASKS.clear_all(); BG.jobs.clear(); LOOPS.stop("reset")
            VLA.reset(); SHARED.__init__(); history = []; print("Reset.\n"); continue
        if cmd == "tasks": print(TASKS.list_all()); print(); continue
        if cmd == "bg": print(BG.check()); print(); continue
        if cmd == "skills": print(SKILL_LOADER.get_descriptions()); print(); continue
        if cmd == "loops": print(LOOPS.check_status()); print(); continue
        if cmd == "routes": print(ROUTING.list_routes()); print(); continue
        if cmd == "stats": print(LOOPS.get_stats()); print(); continue
        if cmd == "safety": print(SAFETY.status()); print(); continue
        if cmd == "/compact": history[:] = auto_compact(history); print("Compressed.\n"); continue
        if cmd == "tokens": print(f"~{estimate_tokens(history) // 4} tokens, {len(history)} msgs\n"); continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"): print(block.text)
        print()
