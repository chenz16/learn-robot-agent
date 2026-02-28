# Product Requirements Document: Robot Agent v0.1

## 1. Executive Summary

A production-grade robot agent framework that orchestrates open-source VLA/VLM models on local GPU, optional user-configured remote LLMs via API, and customer-trained LoRA adapters — all through a unified tool-calling agent loop. Built on [nanobot](https://github.com/HKUDS/nanobot) (~4,000 lines, Python, asyncio), it extends the same architecture that powers chat assistants to also drive physical robots. The agent treats `move_arm` and `send_email` as the same abstraction: a tool.

**Target users**: Robotics teams deploying Unitree G1 (or similar) with GR00T/OpenPI VLAs, who need an orchestration layer that handles model lifecycle, multi-frequency control, safety, and external service integration — without writing a custom framework.

---

## 2. Problem Statement

### 2.1 What We Learned (v0.0 — learn-robot-agent)

The 13-lesson tutorial validated core concepts:
- Agent loop + tool dispatch works for robot control (r01-r02)
- Task decomposition, skills, and compression scale the agent's capability (r03-r06)
- Background threads, multi-robot teams, and safety are feasible (r08-r10)
- Multi-frequency VLA/VLM/LLM loops work in mock mode (r13)

### 2.2 What Doesn't Scale

| Problem | Root Cause | Impact |
|---------|-----------|--------|
| GIL contention at 50Hz × 10 robots | `threading.Lock` everywhere | Latency spikes, missed control deadlines |
| 4,000+ lines of duplicated code | Each lesson self-contained (pedagogical) | Unmaintainable for production |
| No model lifecycle management | Models treated as external HTTP endpoints | Can't load/unload/swap models or LoRA |
| No external service integration | No MCP, no channels | Robot can't email, chat, or access config servers |
| Flat tool dispatch | Hardcoded `TOOL_HANDLERS = {}` dict | Can't add/remove tools at runtime |
| File-based messaging | JSONL inboxes on local filesystem | Doesn't work across machines |

### 2.3 What's Actually Needed

1. **Open-source models, locally deployed**: GR00T, OpenPI, Qwen-VL on local GPU
2. **Customer adaptation**: Fine-tuned LoRA adapters as small files (<200MB), hot-swappable
3. **Dynamic model loading**: Base models loaded on-demand (minutes), LoRA swapped per-task (seconds)
4. **Multi-frequency control**: VLA@2-50Hz, VLM@1-5Hz, LLM@0.1Hz — concurrently
5. **External service access**: Email, chat tools, config servers — via MCP
6. **One unified framework**: Robot tools + AI tools + communication — same agent loop

---

## 3. Product Vision

```
v0.0 (done)                    v0.1 (this PRD)                 v1.0 (future)
─────────────                  ───────────────                  ─────────────
learn-robot-agent              robot-agent                      production fleet
13 self-contained .py          Modular package on nanobot       Distributed deployment
threading, mock only           asyncio, mock + real             Kubernetes, multi-machine
Teaching tool                  Lab prototype (1-3 robots)       Production (10+ robots)
```

**v0.1 scope**: Single-machine deployment supporting 1-3 robots in a lab environment, with mock mode for development and real mode for G1 + GR00T/OpenPI hardware.

---

## 4. System Architecture

### 4.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User / Operator                              │
│  Telegram │ Slack │ 飞书 │ CLI │ Email │ Web UI                     │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ (nanobot channels)
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Agent Core (nanobot)                             │
│                                                                     │
│  AgentLoop ──► ToolRegistry ──► Tools                               │
│      │              │                                               │
│      │         ┌────┴─────────────────────────────────────┐         │
│      │         │  Robot Tools    │ Model Tools │ MCP Tools │         │
│      │         │  look, move,   │ load_model, │ email,    │         │
│      │         │  grasp, start_ │ swap_adapter│ slack,    │         │
│      │         │  subtask, ...  │ list_models │ database  │         │
│      │         └────┬───────────┴──────┬──────┴─────┬─────┘         │
│      │              │                  │            │               │
│  ContextBuilder  MemoryStore  SkillsLoader  SubagentManager         │
└──────┬──────────────┼──────────────────┼────────────┼───────────────┘
       │              │                  │            │
       ▼              ▼                  ▼            ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐
│ LLM      │  │ Loop Manager │  │ Model        │  │ MCP        │
│ Provider │  │              │  │ Lifecycle    │  │ Servers    │
│          │  │ VLA@2-50Hz   │  │ Manager      │  │            │
│ Local    │  │ VLM@1-5Hz   │  │              │  │ Email      │
│ vLLM     │  │ Evaluator    │  │ Base models  │  │ Slack      │
│ + Remote │  │ Terminators  │  │ LoRA swap    │  │ Database   │
│ (optional) │ │ (local)      │  │ Health check │  │ Config srv │
└──────────┘  └──────┬───────┘  └──────┬───────┘  └────────────┘
                     │                 │
              ┌──────┴──────┐   ┌──────┴──────┐
              │ VLA Adapter │   │ VLM Adapter │
              │ Mock/HTTP/  │   │ HTTP/vLLM   │
              │ GR00T/OpenPI│   │             │
              └──────┬──────┘   └──────┬──────┘
                     │                 │
              ┌──────┴─────────────────┴──────┐
              │      Robot Hardware / Sim      │
              │  Unitree G1  │  MuJoCo Sim    │
              └───────────────────────────────┘
```

### 4.2 Three-Tier Frequency Model

| Tier | Component | Frequency | Location | Latency Budget | Process |
|------|-----------|-----------|----------|----------------|---------|
| 1 | VLA Control | 2-50 Hz | LOCAL GPU | < 20ms | Model Manager |
| 2 | VLM Perception | 0.5-5 Hz | LOCAL GPU | 200ms-2s | Model Manager |
| 3 | LLM Planning | 0.05-0.2 Hz | LOCAL vLLM (default) / REMOTE API (optional) | 0.5-10s | Agent Core |
| 4 | Evaluator | 0.1-1 Hz | LOCAL/REMOTE | 1-3s | Agent Core |

### 4.3 What's Reused from Nanobot vs What's New

| Component | Source | Lines | Status |
|-----------|--------|-------|--------|
| AgentLoop | nanobot | ~500 | Reuse as-is |
| ToolRegistry | nanobot | ~70 | Reuse as-is |
| ContextBuilder | nanobot | ~150 | Reuse as-is |
| MemoryStore | nanobot | ~150 | Reuse as-is |
| SkillsLoader | nanobot | ~230 | Reuse as-is |
| SubagentManager | nanobot | ~240 | Reuse as-is |
| LLMProvider | nanobot | ~110 | Reuse as-is |
| Channels | nanobot | ~1500 | Reuse as-is (11 channels) |
| MCP integration | nanobot | ~100 | Reuse as-is |
| SessionManager | nanobot | ~200 | Reuse as-is |
| MessageBus | nanobot | ~50 | Reuse as-is |
| **Robot Tools** | **new** | ~300 | look, move, grasp, perceive, etc. |
| **VLA Adapters** | **new** (from r13) | ~200 | Mock/HTTP/GR00T/OpenPI |
| **LoopManager** | **new** (from r13) | ~200 | asyncio control+perception loops |
| **Terminators** | **new** (from r13) | ~150 | 5 strategies |
| **RoutingConfig** | **new** (from r13) | ~100 | YAML routes + difficulty classifier |
| **SafetyManager** | **new** (from r10) | ~200 | E-stop, zones, velocity, audit |
| **Model Lifecycle** | **new** | ~400 | Base model + LoRA management |
| **RobotEnv** | **new** (from r01) | ~200 | Mock + real robot interface |

**Total new code**: ~1,750 lines. Everything else comes from nanobot.

---

## 5. Functional Requirements

### FR-1: Agent Core

**Reuse nanobot's agent core without modification.**

| Component | Interface | Behavior |
|-----------|-----------|----------|
| AgentLoop | `async run()`, `async process_direct(content)` | Consume messages from bus, call LLM, dispatch tool calls, save session |
| ToolRegistry | `register(tool)`, `unregister(name)`, `async execute(name, params)` | Dynamic tool registration, JSON Schema validation, async execution |
| ContextBuilder | `build_messages(history, current_message)` | Assemble system prompt from identity + memory + skills + bootstrap files |
| MemoryStore | `async consolidate(session, provider, model)` | LLM-driven memory compaction when session exceeds `memory_window` |
| SkillsLoader | `load_skill(name)`, `build_skills_summary()` | Load SKILL.md files with YAML frontmatter, availability checking |
| SubagentManager | `async spawn(task, label, origin)` | Background asyncio tasks with reduced tool set, max 15 iterations |

**Configuration** (via nanobot's `config.json`):
```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.robot-agent/workspace",
      "model": "local/qwen2.5-72b-instruct",
      "max_tool_iterations": 40,
      "memory_window": 100,
      "temperature": 0.1
    }
  }
}
```

Remote LLM is user-configurable and optional. If no remote provider is configured, planning/evaluation uses the local vLLM model by default.

```json
{
  "agents": {
    "defaults": {
      "model": "local/qwen2.5-72b-instruct",
      "remote_llm": {
        "enabled": true,
        "provider": "openai_compatible",
        "base_url": "https://api.example.com/v1",
        "model": "custom-remote-model"
      }
    }
  }
}
```

### FR-2: Robot Tools

Robot actions exposed as nanobot `Tool` subclasses. The tool set is split into a **base set** (framework-provided) and an **extension set** (user-defined). Users can register custom tools at runtime without modifying framework code.

#### 2.1 Tool Interface

Every tool — base or user-defined — implements the same nanobot `Tool` ABC:

```python
class Tool(ABC):
    name: str                              # unique identifier
    description: str                       # for LLM context
    parameters: dict[str, Any]             # JSON Schema
    async def execute(**kwargs) -> str      # async execution
```

#### 2.2 Base Tool Set (framework-provided)

Always available. Cover the minimum capabilities for any robot agent.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `look` | `question: str` | Capture image + VLM analysis |
| `move` | `target: str, position: [x,y,z]` | Move end-effector to position |
| `grasp` | `action: "open"\|"close"` | Control gripper |
| `perceive` | `goal: str` | Deep perception via subagent (multi-step observation) |
| `start_subtask` | `instruction: str, target: dict, route: str, difficulty: str` | Launch VLA+VLM control loops |
| `check_loops` | — | Current loop status and statistics |
| `wait_subtask` | `timeout: float` | Block until subtask completes |
| `stop_loops` | `reason: str` | Force-stop all active loops |
| `set_route` | `route: str, difficulty: str` | Change routing configuration |
| `list_routes` | — | Available routing configs |
| `classify_task` | `instruction: str` | Determine task difficulty |
| `loop_stats` | — | Performance: Hz achieved, latency |
| `emergency_stop` | — | Immediate halt of all robot motion |
| `add_safety_zone` | `name, center, radius, level` | Define forbidden/warning zone |
| `safety_status` | — | Current safety state and audit log |

#### 2.3 Extension Tool Set (user-defined)

Users register custom tools for their specific robot, sensors, or workflows. The framework provides discovery and registration mechanisms.

**Registration methods:**

1. **Programmatic** — direct Python registration:
```python
class LaserScanTool(Tool):
    name = "laser_scan"
    description = "Perform 360° LIDAR scan and return point cloud summary"
    parameters = {"type": "object", "properties": {"resolution": {"type": "number"}}}
    async def execute(self, resolution=0.01, **kwargs) -> str:
        return await self.scanner.scan(resolution)

agent.tools.register(LaserScanTool(scanner=my_scanner))
```

2. **Config-driven** — declare in `tools.yaml`, framework auto-discovers:
```yaml
extension_tools:
  - module: my_tools.laser
    class: LaserScanTool
    config:
      device: /dev/ttyUSB0
  - module: my_tools.force_sensor
    class: ForceFeedbackTool
    config:
      threshold: 5.0
```

3. **MCP-based** — tools from external MCP servers appear alongside robot tools:
```json
{
  "mcp_servers": {
    "custom-sensors": {
      "command": "python",
      "args": ["-m", "my_sensors.mcp_server"]
    }
  }
}
```

4. **SKILL.md** — skills can compose multiple base and extension tools into reusable workflows.

**User extension examples:**

| Custom Tool | Use Case |
|-------------|----------|
| `laser_scan` | LIDAR-based obstacle mapping |
| `force_feedback` | Force/torque sensor reading |
| `voice_command` | Speech-to-text for operator commands |
| `navigate` | Autonomous navigation to named locations |
| `manipulate_dual` | Dual-arm coordinated manipulation |
| `inspect_weld` | Weld quality inspection with specialized camera |

The framework guarantees: user tools and base tools are treated identically by the LLM. No special handling — the ToolRegistry is flat.

### FR-3: Model Lifecycle Manager

**The central new component.** A dedicated long-running process that manages all local AI models.

#### 3.1 Model Registry

Tracks all available models and their current state.

```
Model Registry (persistent, YAML)
├── Base Models
│   ├── groot-n1.6        state: loaded     GPU: 0   VRAM: 3.2GB
│   ├── openpi-pi0        state: unloaded   GPU: —   VRAM: —
│   ├── qwen2.5-vl-7b     state: loaded     GPU: 0   VRAM: 7.1GB
│   └── dreamzero-v1      state: available  GPU: —   VRAM: —
│
└── Adapters (LoRA)
    ├── groot/apple-pick-v2.safetensors     size: 48MB    compatible: groot-n1.6
    ├── groot/mug-grasp-v1.safetensors      size: 62MB    compatible: groot-n1.6
    ├── openpi/pour-v1.safetensors          size: 35MB    compatible: openpi-pi0
    └── qwen-vl/scene-detect-v2.safetensors size: 120MB   compatible: qwen2.5-vl-7b
```

**Registry file** (`models/registry.yaml`):
```yaml
models:
  groot-n1.6:
    type: vla
    path: /models/groot-n1.6/
    size_gb: 3.2
    protocol: zmq            # zmq | http | websocket
    action_horizon: 16
    default_hz: 20
    adapters:
      apple-pick-v2:
        path: /adapters/groot/apple-pick-v2.safetensors
        size_mb: 48
        task: "pick up apple from table"
        trained_on: "2026-02-15"
      mug-grasp-v1:
        path: /adapters/groot/mug-grasp-v1.safetensors
        size_mb: 62
        task: "grasp mug by handle"

  openpi-pi0:
    type: vla
    path: /models/openpi-pi0/
    size_gb: 2.8
    protocol: websocket
    action_horizon: 10
    default_hz: 20

  qwen2.5-vl-7b:
    type: vlm
    path: /models/qwen2.5-vl-7b/
    size_gb: 7.1
    protocol: http            # vLLM / TGI compatible
    adapters:
      scene-detect-v2:
        path: /adapters/qwen-vl/scene-detect-v2.safetensors
        size_mb: 120

remote_models:
  user-primary:
    enabled: true
    type: llm
    provider: openai_compatible   # anthropic | openai_compatible | deepseek | ...
    base_url: https://api.example.com/v1
    model: custom-remote-model
    latency_class: high       # 1-10s per call
  deepseek-r1:
    enabled: false
    type: llm
    provider: deepseek
    model: deepseek-reasoner
    latency_class: medium     # 0.5-5s per call
```

#### 3.2 Operations

| Operation | Latency | When Used |
|-----------|---------|-----------|
| `load_base(model_name, device)` | 10-60s | Startup, or when task needs a model not yet loaded |
| `unload(model_name)` | 1-5s | Free GPU VRAM for another model |
| `swap_adapter(model_name, adapter_name)` | 0.1-2s | Per-subtask, when route specifies a LoRA |
| `remove_adapter(model_name)` | 0.1s | Reset to base model weights |
| `health_check(model_name)` | <100ms | Periodic liveness check |
| `list_models()` | <10ms | Agent queries available models |
| `list_adapters(model_name)` | <10ms | Agent queries available LoRA adapters |
| `register_adapter(model_name, adapter_path, metadata)` | <1s | After customer fine-tuning completes |

#### 3.3 Loading Protocol

```
Agent decides subtask needs groot-n1.6 with apple-pick-v2 LoRA:

1. Agent: load_model("groot-n1.6", device="cuda:0")
   Model Manager: check state
   ├── If already loaded → skip (0ms)
   ├── If another model on same GPU → unload first, then load (30-60s)
   └── If GPU free → load (10-30s)
   Return: "groot-n1.6 loaded on cuda:0, VRAM 3.2GB"

2. Agent: swap_adapter("groot-n1.6", "apple-pick-v2")
   Model Manager:
   ├── Verify adapter compatible with base model
   ├── Load safetensors file (48MB → 0.5s)
   └── Apply LoRA weights to loaded model
   Return: "apple-pick-v2 active on groot-n1.6"

3. Agent: start_subtask("pick up red apple", adapter="apple-pick-v2")
   LoopManager: call VLA adapter → predict() uses groot-n1.6 + apple-pick-v2

4. Subtask completes → Agent may swap to next adapter or keep current
```

#### 3.4 Customer Fine-Tuning Workflow

```
Customer trains LoRA on their specific task/scene:

1. Train (external):
   python train_lora.py --base groot-n1.6 --data my_dataset/ --output my-task-v1.safetensors

2. Register:
   Agent: register_adapter("groot-n1.6", "/path/to/my-task-v1.safetensors", {
       "task": "pick up custom object from specific shelf",
       "trained_on": "2026-03-01",
       "dataset_size": 500
   })
   Model Manager: validate file, copy to /adapters/groot/, update registry.yaml

3. Use:
   Agent: swap_adapter("groot-n1.6", "my-task-v1")
   Agent: start_subtask("pick up custom object", adapter="my-task-v1")

4. Iterate:
   Train v2 → register → swap → test → keep or rollback to v1
```

#### 3.5 Model Manager Tools

Exposed to the agent as nanobot Tool subclasses:

| Tool | Parameters | Returns |
|------|-----------|---------|
| `load_model` | `model: str, device: str` | Status message + VRAM usage |
| `unload_model` | `model: str` | Confirmation |
| `swap_adapter` | `model: str, adapter: str` | Confirmation + adapter metadata |
| `remove_adapter` | `model: str` | Confirmation (back to base weights) |
| `list_models` | — | Table of models with state/GPU/VRAM |
| `list_adapters` | `model: str` | Available adapters with metadata |
| `register_adapter` | `model: str, path: str, metadata: dict` | Confirmation |
| `model_health` | `model: str` | Health status + latency |

#### 3.6 Model Manager as Skill or MCP

Beyond direct tool calls, the Model Manager can also be accessed via:

**SKILL.md** — Recipes that include model management steps:
```markdown
---
name: precision-grasp-setup
description: Load and configure models for precision grasping
---
# Precision Grasp Setup

1. Ensure groot-n1.6 is loaded: `load_model("groot-n1.6", "cuda:0")`
2. Swap to precision adapter: `swap_adapter("groot-n1.6", "precision-grasp-v3")`
3. Set cautious routing: `set_route("cautious", "hard")`
4. Verify health: `model_health("groot-n1.6")`
```

**MCP Server** — Model Manager exposes an MCP interface so external tools and other agents can query/control models:
```json
{
  "mcp_servers": {
    "model-manager": {
      "command": "python",
      "args": ["-m", "robot_agent.models.mcp_server"],
      "tool_timeout": 120
    }
  }
}
```

### FR-3B: Model Housekeeping & OTA Updates

Model Lifecycle Manager (FR-3) handles running models. **Housekeeping** is the layer above it — managing the files, versions, and disk/VRAM hygiene over time. This is not a full system OTA flash; it is **incremental file-level maintenance** executed by the agent itself via bash + skills.

#### 3B.1 Why This Is Separate from Model Management

| Model Manager (FR-3) | Housekeeping (FR-3B) |
|---|---|
| Load/unload models in VRAM | Clean up unused files on disk |
| Swap LoRA weights at runtime | Download new LoRA versions from server |
| Health check running models | Verify file integrity (checksums) |
| **Operates on running processes** | **Operates on files and storage** |

#### 3B.2 Housekeeping Operations

All executed by the agent through **existing nanobot tools** (bash, file system) composed into skills. No new framework code needed — this is a **skill-layer concern**.

| Operation | Trigger | How It Runs |
|-----------|---------|-------------|
| **Garbage collection** | Scheduled (cron) or disk pressure | Scan `/adapters/` for orphaned files not in registry → archive or delete |
| **Version cleanup** | After new adapter registered | Keep last N versions of each adapter, archive older ones |
| **Disk usage report** | Scheduled or on-demand | `du -sh /models/ /adapters/` → summarize for operator |
| **Integrity check** | Startup or scheduled | Verify sha256 checksums of all model/adapter files against registry |
| **OTA adapter pull** | Agent detects new version available | `curl` / `wget` new safetensors from config server → `register_adapter()` |
| **OTA base model pull** | Operator command or scheduled | Download new base model version → swap after verification |
| **Cache cleanup** | VRAM pressure | Unload least-recently-used model to free GPU memory |
| **Log rotation** | Scheduled | Compress and archive old safety/audit/loop logs |
| **Config sync** | Startup or scheduled | Pull latest `registry.yaml`, `routes/*.yaml` from config server |

#### 3B.3 OTA Update Flow (Incremental, Not Full Flash)

```
Config server has new adapter: groot/apple-pick-v3.safetensors

Agent (via cron or heartbeat):
  1. skill: check_model_updates
     └── bash: curl https://config.example.com/api/adapters/groot/latest
     └── compare with local registry.yaml
     └── result: "apple-pick-v3 available (v2 → v3, 52MB)"

  2. skill: pull_adapter_update
     └── bash: wget -O /adapters/groot/apple-pick-v3.safetensors <url>
     └── bash: sha256sum -c apple-pick-v3.sha256   # integrity
     └── tool: register_adapter("groot-n1.6", "/adapters/groot/apple-pick-v3.safetensors",
               {"task": "apple pick", "version": "v3", "replaces": "apple-pick-v2"})
     └── result: "apple-pick-v3 registered, ready for use"

  3. (optional) skill: cleanup_old_adapter
     └── bash: mv /adapters/groot/apple-pick-v1.safetensors /adapters/archive/
     └── tool: update registry.yaml (remove v1 entry)
     └── result: "apple-pick-v1 archived, 48MB freed"

No downtime. No full reflash. Just file operations.
```

#### 3B.4 Housekeeping as Skills

All housekeeping runs through SKILL.md recipes, using nanobot's built-in bash and file tools:

```markdown
---
name: model-housekeeping
description: "Clean unused adapters, verify integrity, report disk usage"
always: false
---
# Model Housekeeping

## Step 1: Disk usage
Run `du -sh /models/ /adapters/` and report to operator.

## Step 2: Orphan detection
List files in /adapters/ not referenced in /models/registry.yaml.
For each orphan older than 7 days, move to /adapters/archive/.

## Step 3: Integrity check
For each adapter in registry.yaml that has a sha256 field,
run `sha256sum -c` and report any mismatches.

## Step 4: Version cleanup
For each model, keep the latest 3 adapter versions.
Archive older versions to /adapters/archive/.

## Step 5: VRAM report
Run `nvidia-smi` and report GPU memory usage.
If any GPU above 90% utilization, suggest unloading idle models.
```

```markdown
---
name: ota-check
description: "Check config server for model/adapter updates"
---
# OTA Update Check

## Step 1: Pull manifest
Fetch latest manifest from config server:
`curl -s https://config.example.com/api/manifest.json`

## Step 2: Compare versions
Compare remote manifest with local /models/registry.yaml.
List any models or adapters with newer versions available.

## Step 3: Auto-pull (if policy allows)
For adapters marked `auto_update: true` in registry.yaml:
- Download new version
- Verify checksum
- Register with register_adapter()
- Archive old version

For base models: report only, require operator approval to download.
```

#### 3B.5 Implementation Principle

**No new framework code.** Housekeeping uses only:
- `bash` tool (nanobot built-in) — file operations, downloads, checksums
- `read_file` / `write_file` / `edit_file` (nanobot built-in) — registry updates
- `model_health` / `list_models` / `register_adapter` (FR-3 tools) — model operations
- `cron` (nanobot built-in) — scheduled execution
- `message` (nanobot built-in) — notify operator of results
- SKILL.md — compose the above into repeatable recipes

The agent IS the housekeeping system. It uses its own tools to maintain itself.

### FR-4: VLA Adapter Layer and LoRA Layer

Two distinct extension layers for robot intelligence. They are **not the same thing**:

```
┌─────────────────────────────────────────────────────────────┐
│ VLA Adapter Layer                                           │
│ = WHICH model runs (GR00T vs OpenPI vs DreamZero)           │
│ = Different architectures, protocols, action spaces         │
│ = Adding/replacing the entire VLA inference backend         │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ LoRA Layer                                              │ │
│ │ = WHICH weights the model uses                          │ │
│ │ = Same architecture, different task specialization       │ │
│ │ = Loading a small weight file into a LoRA-capable base  │ │
│ │ = Base model MUST have LoRA architecture built in       │ │
│ │ = Agent only provides the weight file path              │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### 4.1 VLA Adapter Layer (which model)

Abstract interface for current and future VLA models. Each adapter handles its model's specific protocol.

```python
class BaseVLAAdapter(ABC):
    async def predict(self, observation: dict, instruction: str) -> list[list[float]]
        """Return action sequence (action_horizon × action_dim)"""

    async def activate(self) -> str
        """Signal server to start inference"""

    async def deactivate(self) -> str
        """Signal server to idle"""

    async def reset(self) -> None
        """Reset internal state for new episode"""

    def get_action_horizon(self) -> int
        """Number of actions per prediction (chunking size)"""

    async def health_check(self) -> bool
        """Liveness check"""

    def supports_lora(self) -> bool
        """Whether this VLA base model has LoRA architecture"""
```

**Implementations:**

| Adapter | Protocol | Action Horizon | LoRA Support | Status |
|---------|----------|---------------|-------------|--------|
| `MockVLAAdapter` | In-process | 10 | Simulated | Implemented (r13) |
| `HTTPVLAAdapter` | POST /predict | configurable | Via server API | Implemented (r13) |
| `GR00TAdapter` | ZMQ :5555 + msgpack | 16 | Yes (built-in) | Stub (r13), implement for v0.1 |
| `OpenPIAdapter` | WebSocket + msgpack_numpy | 10 | Yes (built-in) | Stub (r13), implement for v0.1 |
| `DreamZeroAdapter` | TBD | TBD | TBD | Document interface only |

**Adapter selection** via config:
```yaml
vla:
  backend: groot          # mock | http | groot | openpi
  endpoint: tcp://localhost:5555
  action_horizon: 16
```

#### 4.2 LoRA Layer (which weights)

LoRA is **not an adapter** in the software sense. It is a weight-loading mechanism built into the base VLA model's architecture. The framework's only responsibility:

1. **Know which weight files are available** (registry)
2. **Tell the VLA server which weight file to load** (API call)
3. **The server handles actual weight loading internally**

```python
class LoRAManager:
    async def load_weights(self, model: str, lora_name: str) -> str
        """Tell VLA server to load specific LoRA weights.

        Prerequisite: base model MUST have LoRA architecture.
        The server receives the weight file path and handles loading.
        Agent does NOT touch model internals.
        """

    async def unload_weights(self, model: str) -> str
        """Revert to base model weights"""

    def list_available(self, model: str) -> list[dict]
        """List compatible LoRA weight files for a model"""

    async def register_weights(self, model: str, path: str, metadata: dict) -> str
        """Register new LoRA weight file (after customer fine-tuning)"""
```

**The distinction matters:**

| | VLA Adapter | LoRA Weights |
|---|---|---|
| What changes | The entire model backend | A small weight file |
| Size | 2-7 GB (full model) | 10-200 MB (delta weights) |
| Swap time | Minutes (cold load) | Seconds (hot swap) |
| Architecture | Different per adapter (ZMQ, WebSocket, HTTP) | Same base architecture |
| Who handles loading | Framework (adapter code) | VLA server (internal) |
| Compatibility | Any VLA model | Only LoRA-capable base models |
| User's role | Choose which VLA to deploy | Train LoRA weights for their task |

**LoRA weight flow:**

```
Customer trains LoRA:
  Base model: GR00T N1.6 (LoRA architecture built in)
  Training data: 500 demos of "pick up custom part from conveyor"
  Output: custom-part-v1.safetensors (48MB)

Agent uses LoRA:
  1. load_model("groot-n1.6")           ← load base model (minutes, once)
  2. load_weights("groot-n1.6",         ← load LoRA weights (seconds)
       "custom-part-v1")
  3. start_subtask("pick up part")      ← VLA runs with specialized weights
  4. load_weights("groot-n1.6",         ← swap to different task (seconds)
       "other-task-v2")
  5. start_subtask("other task")        ← VLA runs with new weights
```

**The base model must be LoRA-capable.** If a VLA model doesn't support LoRA (e.g., an early model with no adapter layers), `supports_lora()` returns False, and `load_weights()` returns an error. The framework does not force LoRA onto incompatible models.

### FR-5: Multi-Frequency Control

LoopManager orchestrates concurrent VLA control and VLM perception loops using asyncio.

```python
class LoopManager:
    async def start_subtask(
        self,
        instruction: str,
        target: dict,
        route: str = "default",
        difficulty: str = "easy",
    ) -> str
        """Launch control + perception loops as asyncio tasks"""

    async def wait_for_completion(self, timeout: float = 30.0) -> str
        """Block until subtask completes or times out"""

    async def stop(self, reason: str = "manual") -> str
        """Force-stop all active loops"""

    def check_status(self) -> str
        """Current loop state, step count, elapsed time"""

    def get_stats(self) -> str
        """Achieved Hz, latency percentiles, termination reason"""
```

**Control loop** (runs at `action_hz`):
```
async def _control_loop(action_hz, terminator):
    while not termination_signal.is_set():
        observation = env.get_observation()
        actions = await vla_adapter.predict(observation, instruction)
        for action in actions:
            if termination_signal.is_set(): break
            await env.step_delta(action)
            if terminator.should_stop(state, target):
                termination_signal.set()
        await asyncio.sleep(1.0 / action_hz)
```

**Perception loop** (runs at `perception_hz`):
```
async def _perception_loop(perception_hz):
    while not termination_signal.is_set():
        image = await env.capture_image()
        scene = await vlm.analyze(image, "describe the scene")
        shared_state.update_scene(scene)
        await asyncio.sleep(1.0 / perception_hz)
```

### FR-6: Termination Strategies

VLAs do not self-terminate. External supervision required.

```python
class BaseTerminator(ABC):
    def should_stop(self, state: dict, target: dict) -> tuple[bool, str]:
        """Return (should_stop, reason)"""
    def reset(self) -> None:
        """Reset for new subtask"""
```

| Strategy | Fires When | Used For |
|----------|-----------|----------|
| `StepLimitTerminator(max_steps)` | Step count >= max_steps | Safety fallback (always included) |
| `PositionThresholdTerminator(threshold_m)` | ee_pos within threshold of target | Move-to-target tasks |
| `GripperStateTerminator(expected_state)` | Gripper matches expected open/closed + object grasped/released | Grasp/release tasks |
| `VLMTerminator(question_template)` | VLM confirms subtask completion | Complex multi-step tasks |
| `CompositeTerminator(terminators, mode)` | Any (OR) / All (AND) of children fire | Combined strategies |

**Assigned by routing config:**
```yaml
routes:
  easy:
    termination: [position_threshold, step_limit]    # fast, no VLM needed
  medium:
    termination: [vlm_check, step_limit]             # periodic VLM check
  hard:
    termination: [vlm_check, gripper_state, step_limit]  # multi-signal
```

### FR-7: Dynamic Routing

Task difficulty determines which models run, at what frequency, and with what LoRA.

**Difficulty classification:**
```
classify_difficulty(instruction, scene) → "easy" | "medium" | "hard"

  easy:   Known object + simple verb     ("pick up red apple")
  medium: Known type + new instance      ("pick up the blue mug")
  hard:   Compound / novel task          ("pour water, then place mug on shelf")
```

**Route config** (`routes/default.yaml`):
```yaml
name: default
description: "Standard pick-and-place routing"

defaults:
  action_hz: 20
  perception_hz: 2
  max_steps: 300
  position_threshold: 0.05

routes:
  easy:
    planner: null                        # no LLM replanning
    perception_hz: 1
    action_hz: 20
    termination: [position_threshold, step_limit]
    max_steps: 100
    vla_lora: null                       # base model
  medium:
    planner: local-vlm                   # VLM as quick planner
    perception_hz: 2
    action_hz: 20
    termination: [vlm_check, step_limit]
    max_steps: 200
    vla_lora: null
  hard:
    planner: remote-llm                  # user-configured remote; fallback local-vllm
    perception_hz: 5
    action_hz: 50
    termination: [vlm_check, gripper_state, step_limit]
    max_steps: 500
    vla_lora: task-specific              # load user's fine-tuned LoRA
    evaluator: remote-llm                # same fallback rule
```

**Three route profiles included:**
- `default` — Balanced pick-and-place (action@20Hz)
- `fast_manipulation` — Max speed for known objects (action@50Hz)
- `cautious` — Safety-first with frequent VLM checks (action@10Hz, perception@5Hz)

### FR-8: Safety

Multi-layer safety pipeline. Every robot action passes through safety checks before execution.

```
Action Request
    │
    ▼
┌─ Layer 1: E-Stop ────────────────────────────┐
│  If e_stop_active: BLOCK all actions          │
└───────────────────────────────┬───────────────┘
                                │
┌─ Layer 2: Zone Enforcement ───┴──────────────┐
│  If target in forbidden zone: BLOCK           │
│  If target in warning zone: LOG + ALLOW       │
└───────────────────────────────┬───────────────┘
                                │
┌─ Layer 3: Velocity Limits ────┴──────────────┐
│  If speed > max_velocity: CLAMP              │
└───────────────────────────────┬───────────────┘
                                │
┌─ Layer 4: Execute ────────────┴──────────────┐
│  Send action to robot / sim                   │
│  Append to audit log                          │
└──────────────────────────────────────────────┘
```

**Safety config** (`safety.yaml`):
```yaml
e_stop:
  persistent: true                  # survives restart

zones:
  - name: operator_area
    center: [0.0, -0.5, 0.0]
    radius: 0.3
    level: forbidden
  - name: table_edge
    center: [0.6, 0.0, 0.8]
    radius: 0.1
    level: warning

velocity:
  max_linear: 0.5                   # m/s
  max_angular: 1.0                  # rad/s

audit:
  log_dir: ./safety_logs/
  rotation: daily
  retention_days: 30
```

### FR-9: Multi-Robot Coordination

Multiple robots coordinated through nanobot's MessageBus with role-specific tool sets.

**Team config** (`team.yaml`):
```yaml
team:
  - name: scout
    role: perception
    system_prompt: "You are a scout robot. Your job is to explore and report."
    tools: [look, perceive, move, send_message, read_inbox]
    model: qwen2.5-vl-7b

  - name: manipulator
    role: manipulation
    system_prompt: "You are a manipulation robot. Execute pick-and-place tasks."
    tools: [look, move, grasp, start_subtask, send_message, read_inbox]
    model: local-qwen2.5-72b
    vla: groot-n1.6

  - name: monitor
    role: safety
    system_prompt: "You monitor the workspace for safety violations."
    tools: [look, perceive, emergency_stop, safety_status, send_message, read_inbox]
    model: qwen2.5-vl-7b
```

**Communication**: via nanobot MessageBus (asyncio queues). Each robot has a named inbox. Messages are typed (status, request, alert, report).

### FR-10: Autonomous Operation

When idle, the robot autonomously performs useful work — or performs choreographed demonstrations.

#### 10.1 Patrol

Waypoint routes defined in YAML, executed in loop/once/timed modes. At each waypoint: look → detect anomalies → report or claim task.

#### 10.2 Task Auto-Claim

Scan persistent task board for unclaimed tasks matching robot's capabilities. Claim and execute autonomously.

#### 10.3 Anomaly Detection

VLM-based scene analysis at each waypoint. Keyword or confidence-threshold triggers anomaly report to operator.

#### 10.4 Performance / Show Actions

Choreographed demonstration sequences for exhibitions, customer demos, or visitor interaction. Defined as skills or route configs.

```yaml
# skills/demo-show/SKILL.md
---
name: demo-show
description: "Exhibition demo: pick, pour, place sequence with pauses for audience"
---
# Demo Show Sequence
1. Wave greeting gesture
2. Pick up apple with dramatic pause (2s hold at peak)
3. Show apple to audience (rotate wrist, hold 3s)
4. Place apple on plate with precision
5. Pour water into glass (slow, visible pour)
6. Take a bow gesture
```

**Performance modes:**

| Mode | Description | Use Case |
|------|-----------|----------|
| `patrol` | Autonomous waypoint traversal | Security, inspection |
| `auto_claim` | Pick up tasks from board | Warehouse, assembly |
| `demo_show` | Choreographed skill sequence | Exhibition, customer demo |
| `interactive` | Wait for commands, respond | Visitor interaction |

**Idle-cycle priority**: When no explicit task:
1. Check task board for unclaimed work → auto_claim
2. If no tasks, execute patrol route → patrol
3. If no patrol route, enter standby with interactive mode → interactive

### FR-11: Simulation

Fork-compare-promote pattern for testing strategies in sandbox before executing on real hardware.

```
main_env (real or primary sim)
    │
    ├── fork("strategy_a") → isolated copy
    │     └── run actions → measure outcome
    │
    ├── fork("strategy_b") → isolated copy
    │     └── run actions → measure outcome
    │
    └── compare("strategy_a", "strategy_b")
          └── promote winner → apply to main_env
```

**No safety checks in sim** — the agent can freely explore strategies.

### FR-12: Communication Channels

**Reuse nanobot's channel system directly.** No custom code needed.

| Channel | Protocol | Use Case |
|---------|----------|----------|
| Telegram | Bot API | Operator commands, status alerts |
| Slack | WebSocket | Team notifications |
| 飞书 (Feishu) | WebSocket | Enterprise integration |
| 钉钉 (DingTalk) | Stream | Enterprise integration |
| Discord | Gateway | Community / demo |
| Email | IMAP/SMTP | Reports, audit summaries |
| CLI | stdin/stdout | Development, debugging |
| Matrix | Sync | Self-hosted alternative |

**MCP for external services:**
```json
{
  "mcp_servers": {
    "config-server": {
      "url": "https://config.example.com/mcp",
      "headers": {"Authorization": "Bearer ${CONFIG_TOKEN}"},
      "tool_timeout": 10
    },
    "model-manager": {
      "command": "python",
      "args": ["-m", "robot_agent.models.mcp_server"]
    }
  }
}
```

### FR-13: Persistence

| Data | Storage | Lifetime |
|------|---------|----------|
| Conversation history | JSONL per session (nanobot SessionManager) | Until `/new` or consolidation |
| Long-term memory | MEMORY.md (nanobot MemoryStore) | Permanent, LLM-maintained |
| Task board | JSON files with dependency graph | Until task completed/archived |
| Safety audit log | JSONL with daily rotation | 30-day retention |
| Model registry | YAML | Permanent, version-controlled |
| Route configs | YAML | Permanent, version-controlled |
| Patrol routes | YAML | Permanent, version-controlled |
| Sim trajectories | JSON per fork | Until promoted/discarded |

---

## 6. Non-Functional Requirements

### NFR-1: Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| VLA control loop latency | < 20ms per step | Time from predict() call to step_delta() completion |
| VLM perception latency | < 2s per frame | Time from capture to scene description available |
| LLM planning latency | < 10s per decision | Time from tool call to response |
| LoRA swap time | < 2s | Time from swap_adapter() call to ready |
| Base model load time | < 60s | Time from load_model() to first predict() |
| End-to-end task | < 30s for simple pick-and-place | From instruction to verification |
| Subtask loop startup | < 500ms | Time from start_subtask() to first VLA prediction |

### NFR-2: Scalability

| Scenario | Target |
|----------|--------|
| Robots per deployment | 1-3 (v0.1), 10+ (v1.0) |
| Tools per agent | 30+ without performance degradation |
| Concurrent subtasks | 3 (one per robot) |
| Session history | 1000+ messages before consolidation needed |
| Adapters per model | 50+ in registry |
| Route configs | 10+ YAML files |

### NFR-3: Concurrency

- **asyncio** for all I/O-bound operations (LLM calls, VLA/VLM inference, file I/O)
- **No threading.Lock** in hot paths (use asyncio.Lock where needed)
- **No GIL contention** — single event loop handles all coroutines
- **CPU-bound exceptions**: VLA predict() may use `asyncio.to_thread()` for blocking inference calls

### NFR-4: Reliability

| Scenario | Expected Behavior |
|----------|------------------|
| VLA server unreachable | Retry 3x, then fall back to lower-frequency route or alert operator |
| LLM API timeout | Retry with exponential backoff, fall back to local model |
| Agent process crash | Session persisted on disk, resume from last checkpoint |
| Model load failure | Log error, remain on previous model, alert operator |
| LoRA incompatible | Reject swap, log warning, continue with base model |
| E-stop triggered | All motion stops immediately, persist state, require manual reset |

### NFR-5: Security

- **Tool validation**: JSON Schema on all tool parameters (nanobot built-in)
- **Command injection**: ExecTool blocks dangerous patterns (nanobot built-in)
- **Path traversal**: `restrict_to_workspace` option (nanobot built-in)
- **API key management**: Config file with env var interpolation, never logged
- **Channel allowlists**: Optional `allow_from` filter per channel (nanobot built-in)
- **LoRA file validation**: Checksum verification on adapter files before loading

### NFR-6: Observability

| Signal | Implementation |
|--------|---------------|
| Structured logging | loguru with JSON format (nanobot built-in) |
| Loop metrics | Steps/s, achieved Hz, termination reasons |
| Model state | Current GPU assignment, VRAM usage, adapter loaded |
| Safety events | Zone violations, e-stop triggers, velocity clamps |
| Session tracking | Session key in all log lines (nanobot built-in) |
| Tool execution | Tool name + args logged (nanobot built-in, first 200 chars) |

---

## 7. Model Lifecycle Requirements (Deep Dive)

### 7.1 Open-Source First

All core control/perception/planning paths must work with open-source or open-weight local models. Proprietary remote models are optional accelerators, not hard dependencies.

| Component | Primary Model | Fallback | License |
|-----------|--------------|----------|---------|
| VLA | GR00T N1.6 | OpenPI pi0 | Apache-2.0 / MIT |
| VLM | Qwen2.5-VL-7B | InternVL2 | Apache-2.0 |
| LLM (local, default) | Qwen2.5-72B (vLLM) | DeepSeek-V3 (vLLM) | Apache-2.0 |
| LLM (remote, optional) | User-configured endpoint | Local LLM (above) | Varies by provider |

### 7.2 Storage Layout

```
/models/                              # Base models (large, 2-7GB each)
├── groot-n1.6/
│   ├── config.json
│   └── model.safetensors
├── openpi-pi0/
│   └── ...
└── qwen2.5-vl-7b/
    └── ...

/adapters/                            # LoRA adapters (small, 10-200MB each)
├── groot/
│   ├── apple-pick-v2.safetensors     # 48MB
│   ├── mug-grasp-v1.safetensors     # 62MB
│   └── custom-task-v1.safetensors   # customer fine-tuned
├── openpi/
│   └── pour-v1.safetensors          # 35MB
└── qwen-vl/
    └── scene-detect-v2.safetensors  # 120MB

/models/registry.yaml                 # Model + adapter metadata
```

### 7.3 Loading Tiers

```
┌───────────────────────────────────────────────────────────────────┐
│ Tier 0: Always Loaded (never unloaded during session)             │
│ • Primary VLM (Qwen2.5-VL-7B) — used for look() every few seconds│
│ • Cost: ~7GB VRAM                                                 │
├───────────────────────────────────────────────────────────────────┤
│ Tier 1: Task-Duration Loading (loaded for task, kept until swap)  │
│ • Primary VLA (GR00T N1.6) — used during subtask execution       │
│ • Loaded at first subtask, kept loaded between subtasks           │
│ • Cost: ~3GB VRAM                                                 │
├───────────────────────────────────────────────────────────────────┤
│ Tier 2: Per-Subtask Swap (hot-swapped, seconds)                  │
│ • LoRA adapters — swapped per subtask based on route config       │
│ • apple-pick-v2 for apple tasks, mug-grasp-v1 for mug tasks     │
│ • Cost: <200MB additional, swap time <2s                          │
├───────────────────────────────────────────────────────────────────┤
│ Tier 3: On-Demand Loading (cold-loaded, minutes)                 │
│ • Secondary VLA (OpenPI pi0) — loaded only if needed             │
│ • May require unloading primary VLA first (VRAM constraint)      │
│ • Cost: time (30-60s to swap base model)                         │
└───────────────────────────────────────────────────────────────────┘
```

### 7.4 Remote LLM Considerations

- **Optionality**: Remote LLM is optional. System runs fully with local vLLM only.
- **Default**: If remote LLM is not configured, use local Qwen/DeepSeek via vLLM for planning/evaluation.
- **Latency**: Remote LLM is typically 1-10s per call. Only used for planning (0.1Hz), not control.
- **Fallback**: If remote LLM is configured but unreachable, fall back to local Qwen/DeepSeek.
- **Cost**: Remote API calls are metered. Route config determines when LLM is used:
  - Easy tasks: no LLM replanning (save cost)
  - Hard tasks: LLM evaluates each subtask (worth the cost)
- **Provider abstraction**: Nanobot's LLMProvider supports 20+ providers. No vendor lock-in.

### 7.5 Model Manager Process Architecture

```
┌─────────────────────────────────────────────────────┐
│                Model Manager Process                 │
│                (always running)                      │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │ Registry        │  │ GPU Manager              │ │
│  │ (YAML on disk)  │  │ • Track VRAM usage       │ │
│  │ • Models        │  │ • Allocate/free devices   │ │
│  │ • Adapters      │  │ • OOM prevention         │ │
│  │ • State         │  │                          │ │
│  └────────┬────────┘  └──────────┬───────────────┘ │
│           │                      │                  │
│  ┌────────┴──────────────────────┴───────────────┐ │
│  │ API Layer                                      │ │
│  │ • load_model(name, device)                    │ │
│  │ • unload_model(name)                          │ │
│  │ • swap_adapter(model, adapter)                │ │
│  │ • health_check(model)                         │ │
│  │ • list_models() / list_adapters()             │ │
│  │ • register_adapter(model, path, metadata)     │ │
│  └───────────────────────────────────────────────┘ │
│           │                                         │
│  Exposed as:                                        │
│  ├── Nanobot Tools (ToolRegistry.register)          │
│  ├── MCP Server (for external access)               │
│  └── SKILL.md (for recipe-based loading)            │
└─────────────────────────────────────────────────────┘
```

---

## 8. Hardware & Software Targets

### 8.1 Target Hardware

| Component | Spec | Notes |
|-----------|------|-------|
| Robot | Unitree G1 humanoid | 30-DOF, walk + manipulate |
| GPU | NVIDIA H200 (80GB) | Local VLA + VLM inference |
| Camera | Robot-mounted RGB | 640×480 or 1280×720 |
| Gripper | Unitree dexterous hand | 2-finger or 5-finger |
| Network | LAN (robot ↔ GPU server) | <1ms latency required |
| Cloud | API access (optional) | For user-configured remote LLM provider |

### 8.2 Software Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| OS | Ubuntu | 22.04+ |
| Python | CPython | 3.11+ |
| Agent framework | nanobot | 0.1.4+ |
| VLA serving | Isaac GR00T | N1.6 |
| VLM serving | vLLM | 0.6+ |
| Sim | MuJoCo | 3.0+ |
| GPU runtime | CUDA | 12.0+ |
| Container | Docker | 24.0+ (optional) |

### 8.3 VRAM Budget (H200 80GB)

```
Component                    VRAM
──────────────────────────   ──────
Qwen2.5-VL-7B (FP16)        ~14GB
GR00T N1.6 (FP16)           ~6GB
LoRA adapter (active)        <1GB
CUDA kernels + overhead      ~4GB
──────────────────────────   ──────
Total                        ~25GB
Available for second VLA     ~55GB    ← room for OpenPI or DreamZero
```

---

## 9. Integration Points

### 9.1 Dependency Map

```
robot-agent (new repo)
├── nanobot-ai (pip)           # Agent core, tools, channels, MCP
├── openai / anthropic (pip, optional)  # User-configured remote LLM providers
├── pyyaml (pip)               # Route + model config
├── requests (pip)             # HTTP VLA/VLM adapters
├── pyzmq (pip, optional)      # GR00T ZMQ adapter
├── websockets (pip, optional) # OpenPI WebSocket adapter
└── mujoco (pip, optional)     # Sim environment
```

### 9.2 Service Endpoints

| Service | Protocol | Port | Process |
|---------|----------|------|---------|
| Agent Core | MessageBus (in-process) | — | Main process |
| Model Manager | Tool calls (in-process) or MCP | — | Same or separate process |
| VLA (GR00T) | ZMQ | :5555 | Dedicated GPU server |
| VLA (HTTP proxy) | HTTP | :8020 | Co-located with VLA |
| VLM (Qwen-VL) | HTTP (vLLM) | :8010 | Dedicated GPU server |
| Sim (MuJoCo) | HTTP | :8030 | CPU server |
| LLM (local planner, default) | HTTP (vLLM OpenAI-compatible) | :8008 (example) | Local GPU server |
| LLM (remote, optional) | HTTPS | :443 | User-configured provider API |
| Channels | Various | Various | nanobot channel processes |

### 9.3 MCP Server Integration

External services accessed via MCP tools:

```json
{
  "mcp_servers": {
    "model-manager": {
      "command": "python",
      "args": ["-m", "robot_agent.models.mcp_server"],
      "tool_timeout": 120
    },
    "config-server": {
      "url": "https://config.internal/mcp",
      "headers": {"Authorization": "Bearer ${CONFIG_TOKEN}"}
    },
    "monitoring": {
      "url": "https://grafana.internal/mcp"
    }
  }
}
```

---

## 10. Milestones

### v0.1-alpha: Core Agent + Robot Tools + Mock Mode

**Goal**: Nanobot + robot tools working in mock mode. No real hardware needed.

**Deliverables**:
- Robot tools registered in nanobot ToolRegistry (look, move, grasp, perceive)
- MockRobotEnv with simulated physics
- SKILL.md robot skills (top-grasp, side-grasp, pour, precision-place)
- CLI REPL for testing
- All existing nanobot features work (memory, sessions, channels)

**Verification**: `python -m robot_agent "pick up the apple and put it on the plate"` works in mock mode.

### v0.1-beta: Model Manager + VLA Adapters + Routing

**Goal**: Model lifecycle management and multi-frequency control loops.

**Deliverables**:
- Model Registry (registry.yaml)
- Model Manager with load/unload/swap/health operations
- MockVLAAdapter + HTTPVLAAdapter
- LoopManager with asyncio control + perception loops
- 5 termination strategies
- RoutingConfig with 3 YAML profiles
- Difficulty classifier

**Verification**: `load_model("mock-vla") → swap_adapter("mock-vla", "test-v1") → start_subtask("pick apple")` completes with position_threshold termination.

### v0.1-rc: Multi-Robot + Safety + Channels

**Goal**: Full feature set working in mock mode.

**Deliverables**:
- SafetyManager with e-stop, zones, velocity limits, audit
- Multi-robot team coordination via MessageBus
- Patrol + idle-cycle auto-claim
- Sim fork/compare/promote
- Channel integration tested (at least CLI + Telegram)
- MCP server for Model Manager

**Verification**: 3-robot team (scout + manipulator + monitor) completes coordinated task in mock mode.

### v0.1: Release

**Goal**: Stable release for lab deployment with G1 + GR00T.

**Deliverables**:
- GR00TAdapter (ZMQ) tested against real GR00T server
- VLM integration with vLLM-served Qwen2.5-VL
- MuJoCo sim integration
- Documentation and examples
- `pip install robot-agent` from PyPI

**Verification**: "Pick up the apple and put it on the plate" works on real G1 hardware or MuJoCo sim.

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **VLA** | Vision-Language-Action model. Takes image + text instruction, outputs robot actions. |
| **VLM** | Vision-Language Model. Takes image + text question, outputs text description. |
| **LLM** | Large Language Model. Text-to-text reasoning and planning. |
| **LoRA** | Low-Rank Adaptation. Small parameter file (<200MB) that specializes a base model for a specific task. |
| **MCP** | Model Context Protocol. Standard for connecting AI agents to external tools and services. |
| **Adapter** | A LoRA weight file that can be hot-swapped onto a base model. |
| **Route** | A YAML configuration specifying which models run at what frequency for a given task difficulty. |
| **Terminator** | A strategy that decides when a VLA control loop should stop. |
| **Action Horizon** | Number of action steps a VLA predicts per inference call (chunking). |
| **SharedState** | Thread/task-safe state object shared between control and perception loops. |
| **Subtask** | A single atomic robot action (e.g., "move to apple") executed by the LoopManager. |
| **Mock Mode** | Development mode using simulated physics, no real hardware or GPU needed. |
| **Nanobot** | Ultra-lightweight Python agent framework (~4,000 LOC) that this product builds on. |
| **Tool** | An async function registered with the agent that the LLM can invoke. Robot actions, model management, and external services are all tools. |

---

## Appendix A: Relationship to learn-robot-agent (v0.0)

| Lesson | Concept | v0.1 Mapping |
|--------|---------|-------------|
| r01 | Agent loop | Nanobot AgentLoop (reused) |
| r02 | Tool use | Nanobot ToolRegistry (reused) |
| r03 | Task decomposition | Nanobot SubagentManager (reused) |
| r04 | Perception subagent | Nanobot spawn tool (reused) |
| r05 | Skills | Nanobot SkillsLoader (reused) |
| r06 | Context compression | Nanobot MemoryStore.consolidate (reused) |
| r07 | Persistent tasks | Nanobot SessionManager + task tools (reused) |
| r08 | Background tasks | Nanobot asyncio tasks (reused) |
| r09 | Multi-robot teams | Nanobot MessageBus + team config (adapted) |
| r10 | Safety protocols | New SafetyManager (from r10, asyncified) |
| r11 | Autonomous patrol | New PatrolManager (from r11, asyncified) |
| r12 | Parallel sim | New SimManager (from r12, asyncified) |
| r13 | Dynamic routing | New LoopManager + VLA adapters + routing (from r13, asyncified) |
| — | **Model lifecycle** | **New (v0.1 original)** |
| — | **Communication channels** | **Nanobot channels (reused)** |
| — | **MCP integration** | **Nanobot MCP (reused)** |

## Appendix B: Key Design Decisions

### B.1 Why Nanobot over OpenClaw

| Factor | OpenClaw | Nanobot | Decision |
|--------|----------|---------|----------|
| Language | TypeScript | Python | Python matches VLA/VLM ecosystem |
| Size | 842k LOC | 4k LOC | Smaller = easier to understand and modify |
| Concurrency | Node.js event loop | asyncio | asyncio is native Python, no interop needed |
| Tool pattern | Plugin SDK | ToolRegistry | Same concept, simpler API |
| Channels | 38 channels | 11 channels | Sufficient for robotics use case |
| MCP | Yes | Yes | Both support MCP |
| Maintenance | Active (TypeScript team) | Active (Python team) | Both viable |

**Decision**: Nanobot. Python is the language of robotics (PyTorch, ROS2, MuJoCo, Isaac). Nanobot's 4k LOC is readable in a day. Same architectural patterns as OpenClaw, 99% less code.

### B.2 Why Model Manager as Separate Process

Model loading is expensive (10-60s, multi-GB GPU transfer). Embedding it in the agent loop would:
- Block agent message processing during loads
- Risk VRAM fragmentation from repeated load/unload
- Couple model lifecycle to agent lifecycle

A separate process:
- Loads models independently of agent restarts
- Manages GPU VRAM as a shared resource
- Exposes clean API (tools + MCP) to any consumer
- Can be monitored/restarted independently

### B.3 Why asyncio over threading

| threading | asyncio |
|-----------|---------|
| GIL prevents true parallelism | No GIL contention (single thread) |
| 30 threads × 3 locks = deadlock risk | Cooperative scheduling, no locks needed |
| Context switching overhead at 50Hz | Single event loop, predictable latency |
| Hard to debug race conditions | Deterministic coroutine scheduling |
| Can't scale past ~100 threads | Can scale to 10,000+ coroutines |

### B.4 Why LoRA as First-Class Citizen

Customer fine-tuning is the primary value proposition beyond demo:
- Base models work for generic tasks (demo "pick up apple")
- Real deployment needs task-specific precision (customer's specific objects, scenes, tools)
- LoRA files are small (10-200MB) and fast to swap (<2s)
- Route config naturally maps difficulty → LoRA: easy tasks use base model, hard tasks use fine-tuned LoRA

Making LoRA a first-class concept in routing, adapters, and model management ensures the framework supports the customer's most important workflow: train → register → deploy → iterate.

### B.5 Why Tool = Tool (Unified Abstraction)

A robot `move_arm` and a communication `send_email` share the same interface:

```python
class Tool(ABC):
    name: str
    description: str
    parameters: dict
    async def execute(**kwargs) -> str
```

This means:
- Agent can decide to email an operator after a safety event — same mechanism as moving the arm
- MCP tools (databases, config servers) are first-class alongside robot tools
- No special "robot tool" vs "software tool" distinction — the LLM sees all tools equally
- Adding a new capability (e.g., Slack notification on anomaly) requires zero framework changes — just register a new tool
