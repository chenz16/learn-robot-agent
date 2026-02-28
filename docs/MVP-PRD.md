# MVP PRD: Robot Agent v0.1-alpha

## 1. Executive Summary

A robot agent built on [nanobot](https://github.com/HKUDS/nanobot) (~4,000-line Python asyncio framework) that receives natural language instructions via CLI, orchestrates VLM (visual understanding) and VLA (action generation) models, and controls a robot to complete manipulation tasks in the LIBERO simulation environment. Core philosophy: robot actions and software actions are the same abstraction — a Tool.

**Target users**: Robotics research teams who need a lightweight agent framework to validate the end-to-end feasibility of "LLM orchestrates VLA/VLM to execute manipulation tasks."

**Core demo**: Two LIBERO-10 (Long) multi-step tasks that validate **LLM task decomposition + VLA step-by-step execution** —
1. Repeat decomposition: "put both the alphabet soup and the tomato sauce in the basket" (same action pattern applied to different objects across two rounds)
2. Chain decomposition: "put the black bowl in the bottom drawer of the cabinet and close it" (different actions chained in sequence: open → place → close)

Both tasks require the LLM to decompose into multiple subtasks that a VLA cannot complete in a single step. The agent autonomously completes the perceive → reason → execute → verify loop, with a Franka Panda arm successfully performing the operations in LIBERO simulation.

---

## 2. Scope

### 2.1 IN (MVP Scope)

| Dimension | Content |
|-----------|---------|
| Robot count | Single robot |
| Simulation | LIBERO (MuJoCo), Franka Panda fixed-base, 130+ tabletop manipulation tasks |
| Interaction | CLI terminal (text input) |
| Agent framework | nanobot, reusing AgentLoop, ToolRegistry, SkillsLoader, etc. |
| Models | VLM (scene understanding) + VLA (action generation) + LLM (reasoning & orchestration) |
| Control loop | PRAE: Prepare → Perceive → Reason → Act → Evaluate |
| Robot Tools | 8 base tools |
| VLA integration | Mock adapter (dev/debug) + HTTP adapter (real VLA inference service) |
| Safety | Emergency stop + velocity limits |
| Run modes | Mock mode (no GPU) + LIBERO simulation mode (CPU) |

### 2.2 OUT (Deferred)

| Feature | Reason for deferral |
|---------|-------------------|
| Multi-robot coordination | Single robot is sufficient to validate the architecture |
| Voice input (onboard mic / Web Voice) | Requires ASR/VAD dependencies; CLI is sufficient |
| LoRA hot-swap | Base model is sufficient for demo tasks |
| Model lifecycle management (load/unload/VRAM) | Models are started manually by the developer |
| OTA updates / model maintenance | Operational concern, not architecture validation |
| Autonomous patrol / task auto-claim | Application-layer behavior, implementable later via SKILL.md |
| Sim fork-compare-promote | Requires multiple parallel sim instances |
| Web UI / multi-channel (Telegram, Slack, etc.) | CLI is sufficient; nanobot channels can be enabled later with zero code changes |
| Safety zone fencing / audit logging | Requires 3D spatial awareness; not needed for MVP |
| Real hardware (Unitree G1 + GR00T) | MVP validates in simulation first |
| Navigation (mobile base) | LIBERO is fixed-base; navigation requires RoboCasa or similar mobile-base simulation |

---

## 3. Architecture Overview

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User (CLI Terminal)                  │
└──────────────────────┬──────────────────────────────┘
                       │ natural language instruction
                       ▼
┌─────────────────────────────────────────────────────┐
│                Agent Core (nanobot)                   │
│                                                       │
│  AgentLoop ──► ToolRegistry ──► Robot Tools           │
│      │              │                                 │
│      │         ┌────┴────────────────────────┐       │
│      │         │ look, move, grasp, perceive │       │
│      │         │ start_subtask, check_loops  │       │
│      │         │ wait_subtask, emergency_stop│       │
│      │         └────┬──────────┬─────────────┘       │
│      │              │          │                      │
│  ContextBuilder  MemoryStore  SkillsLoader            │
└──────┬──────────────┼──────────┼─────────────────────┘
       │              │          │
       ▼              ▼          ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│ LLM      │  │ Loop Manager │  │ VLA Adapter  │
│ Provider │  │              │  │              │
│          │  │ VLA @ 2-20Hz │  │ Mock / HTTP  │
│ Local    │  │ Terminators  │  │              │
│ vLLM     │  └──────┬───────┘  └──────┬───────┘
└──────────┘         │                 │
              ┌──────┴─────────────────┴──────┐
              │        LIBERO (MuJoCo)         │
              │  130+ manipulation tasks       │
              │  Gymnasium API                 │
              └────────────────────────────────┘
```

### 3.2 Three-Layer Abstraction

The agent's reasoning and control are divided into three logical layers with increasing frequency and decreasing abstraction:

| Layer | Responsibility | Typical Latency | MVP Implementation |
|-------|---------------|----------------|-------------------|
| **Intention** | Intent recognition, slot extraction, decide if full planning is needed | 50-800ms | Single-turn LLM inference |
| **Cognition** | Task reasoning/planning, outputs structured plan; contains optional Perception sub-module | 0.5-10s | LLM planning, optionally preceded by VLM perception |
| **Action** | High-frequency control loop: VLA prediction + safety checks + execution | 2-20Hz, < 20ms/step | LoopManager + VLA adapter |

**Perception as optional sub-module of Cognition**: Perception (VLM-based scene understanding) is not a fourth layer — it is an optional, independently configurable component within Cognition. When enabled, Cognition calls Perception before Reasoning to obtain a scene description. When disabled (e.g., mock mode), Cognition plans directly from simulation ground-truth or prior context. The Perception backend is replaceable (VLM / traditional CV / ground-truth) without changing Cognition's external interface.

Design constraint: The Action layer must not be blocked by Intention/Cognition — Cognition latency only affects the next subtask decision, never interrupts a running action loop.

### 3.3 Nanobot Reuse vs New Code

| Component | Source | Description |
|-----------|--------|-------------|
| AgentLoop | nanobot reuse | Consume messages → call LLM → dispatch tool calls → save session |
| ToolRegistry | nanobot reuse | Dynamic register/unregister, JSON Schema validation |
| ContextBuilder | nanobot reuse | Assemble system prompt: identity + memory + skills |
| MemoryStore | nanobot reuse | LLM-driven memory compaction |
| SkillsLoader | nanobot reuse | Load SKILL.md files with YAML frontmatter |
| SubagentManager | nanobot reuse | Background asyncio subagents, max 15 iterations |
| LLMProvider | nanobot reuse | Supports 20+ model providers |
| **Robot Tools** | **new** | look, move, grasp, etc. (8 tools) |
| **VLA Adapters** | **new** | Mock + HTTP (2 adapters) |
| **LoopManager** | **new** | asyncio control loop + cognition handoff |
| **Terminators** | **new** | StepLimit + PositionThreshold |
| **SafetyManager** | **new** | E-Stop + velocity limits |
| **RobotEnv** | **new** | Mock + LIBERO environment interface |

**Estimated new code**: ~800 lines. Everything else is reused from nanobot.

---

## 4. Functional Requirements

### FR-1: Agent Core

**Requirement**: Reuse nanobot's agent core without modification.

- AgentLoop consumes messages, calls LLM, dispatches tool calls, saves sessions
- ToolRegistry supports dynamic registration; all tool parameters validated via JSON Schema
- ContextBuilder assembles system prompt from identity + memory + skills + bootstrap files
- MemoryStore automatically compacts when session exceeds `memory_window`
- SkillsLoader loads SKILL.md files from the `skills/` directory
- SubagentManager supports spawning subagents for perception/execution/evaluation subtasks

**Configuration requirements**:
- Specify workspace directory, model, max tool iterations, memory window, temperature
- Model defaults to local vLLM; optionally configure a remote LLM (user-provided API)

### FR-2: Robot Tools

**Requirement**: 8 base tools covering the minimum capabilities for a PRAE loop.

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `look` | question: str | Scene description + object list | Capture image + VLM analysis |
| `move` | target: str, position: [x,y,z] | Execution status | Move end-effector to target position |
| `grasp` | action: "open" \| "close" | Gripper state | Control gripper |
| `perceive` | goal: str | Deep perception result | Multi-step observation analysis via subagent |
| `start_subtask` | instruction: str, target: dict | Subtask ID + status | Launch VLA control loop |
| `check_loops` | — | Loop status + stats | Query current control loop state |
| `wait_subtask` | timeout: float | Completion status + result | Block until subtask completes |
| `emergency_stop` | — | Confirmation | Immediately halt all robot motion |

**Extension mechanism**: Users can register custom tools via Python code (subclass Tool ABC, call `agent.tools.register()`). Custom tools and base tools are treated identically by the LLM — the ToolRegistry is flat.

### FR-3: Model Management

**Requirement**: Minimal model management — confirm externally started model services are ready.

In the MVP, model services (VLM, VLA) are started manually by the developer. The agent only needs:

| Capability | Description |
|------------|-------------|
| **Health check** | Query whether each model service is online and its response latency |
| **Status query** | Which models are currently loaded, service addresses |
| **Readiness confirmation** | Before starting a PRAE loop, confirm VLM + VLA + Sim are all ready |

The agent provides `model_health` and `model_ensure` as internal tools. No model loading/unloading/VRAM management.

### FR-4: VLA Adapter Layer

**Requirement**: A unified VLA interface abstraction with two implementations for the MVP.

The VLA adapter defines a unified interface: input (image + instruction + robot state) → output (action sequence).

| Adapter | Purpose | Backend |
|---------|---------|---------|
| **MockVLAAdapter** | Development and debugging, no GPU required | In-process simulation, returns random/fixed action sequences |
| **HTTPVLAAdapter** | Connect to a real VLA inference service | HTTP POST to VLA server (SmolVLA / pi0 / others) |

Interface contract:
- **predict**: Input observation dict + instruction string → return action sequence (action_horizon × action_dim)
- **reset**: Reset internal state (new episode)
- **health_check**: Liveness check
- **get_action_horizon**: Return the number of action steps per prediction (chunking size)

Adapter type is switched via configuration, transparent to LoopManager and the upper-layer agent.

### FR-5: Multi-Frequency Control

**Requirement**: LoopManager manages the VLA control loop, decoupled from upper-layer reasoning.

LoopManager is the execution engine of the Action layer:

- **start_subtask**: Receive instruction + target + route config, launch async control loop
- **Control loop**: Run at action_hz frequency — get observation → VLA predict → safety check → execute action → check termination
- **wait_for_completion**: Block until subtask completes or times out
- **stop**: Force-stop all active loops

Design constraints:
- The control loop runs as an asyncio task, never blocking the agent main loop
- If VLA predict() is a blocking call, wrap it with `asyncio.to_thread()`
- Cognition layer latency does not affect a running Action loop

### FR-6: Termination Strategies

**Requirement**: VLAs do not self-terminate; external termination strategies are required. The MVP provides 2 strategies.

| Strategy | Trigger Condition | Purpose |
|----------|------------------|---------|
| **StepLimitTerminator** | step_count >= max_steps | Safety fallback, prevent infinite execution |
| **PositionThresholdTerminator** | end-effector distance to target < threshold | Completion detection for move-to-target tasks |

The two strategies can be combined (AND / OR). The route config specifies which termination strategy combination to use for each difficulty level.

### FR-7: Routing

**Requirement**: Task difficulty determines control parameters. The MVP provides 1 default route config with 2 difficulty levels.

**Difficulty classification**:
- **easy**: Known object + simple verb ("pick up the red cup")
- **hard**: Compound multi-step task ("put the black bowl in the bottom drawer of the cabinet and close it")

**Parameters determined by route**:

| Parameter | easy | hard |
|-----------|------|------|
| action_hz | 10 | 20 |
| max_steps | 100 | 500 |
| position_threshold | 0.05m | 0.03m |
| Termination strategy | StepLimit + PositionThreshold | StepLimit + PositionThreshold |
| LLM re-planning | None | Re-plan after each subtask |

Route config is stored as a YAML file. The agent selects difficulty and route during the Reason phase of the PRAE loop.

### FR-8: Safety

**Requirement**: Basic safety guarantees — the ability to stop and limit speed.

| Safety Layer | Behavior |
|-------------|----------|
| **E-Stop** | When `emergency_stop` is triggered, immediately halt all motion. State persists across restarts. Manual reset required. |
| **Velocity Limit** | Before each action execution, check velocity. If exceeding max_velocity, clamp automatically (do not reject the action). |

Safety checks are inside the LoopManager control loop, executed before `env.step()`. Each safety event is logged.

---

## 5. PRAE Loop

Both acceptance tasks are from **LIBERO-10 (Long)** and require LLM decomposition with VLA step-by-step execution.

### 5.1 Task A (Repeat Decomposition): put both the alphabet soup and the tomato sauce in the basket

**LIBERO task ID**: `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`

**LLM decomposition focus**: The same action pattern (pick → place) applied to different objects across two rounds, with re-perception between rounds.

```
Step 1: PREPARE
  Agent receives user instruction → model_ensure confirms VLM + VLA + Sim all ready
  LIBERO loads LIVING_ROOM_SCENE2

Step 2: PERCEIVE
  look("describe all objects on the table and their positions") →
    VLM returns: "alphabet soup at [0.2, -0.1, 0.10], tomato sauce at [0.4, 0.2, 0.10],
                  basket at [0.6, 0.0, 0.05], ..."

Step 3: REASON (LLM task decomposition)
  LLM analyzes instruction "put both ... in the basket" →
    Identifies: 2 target objects, 1 target container
    Decomposes into 2 rounds, each round = pick + place:
      Round 1: pick up alphabet soup → place in basket
      Round 2: pick up tomato sauce → place in basket
    Difficulty: hard (multi-object compound task)
    Route: default/hard (action_hz=20, max_steps=500)

=== Round 1: alphabet soup ===

Step 4: ACT (Subtask 1: pick up alphabet soup)
  start_subtask("pick up the alphabet soup",
                target={"object": "alphabet_soup", "position": [0.2, -0.1, 0.10]}) →
    VLA control loop @ 20Hz → approach → descend → close gripper → lift
    StepLimit(500) termination

Step 5: ACT (Subtask 2: place in basket)
  start_subtask("place the alphabet soup in the basket",
                target={"object": "basket", "position": [0.6, 0.0, 0.05]}) →
    VLA control loop → move above basket → descend → open gripper
    StepLimit(500) termination

Step 6: EVALUATE (intermediate check)
  look("is the alphabet soup in the basket?") →
    VLM confirms: "alphabet soup is inside the basket"
  Pass → proceed to Round 2

=== Round 2: tomato sauce ===

Step 7: PERCEIVE (re-perceive — scene has changed, soup is no longer on table)
  look("where is the tomato sauce now?") →
    VLM returns: "tomato sauce at [0.4, 0.2, 0.10]"

Step 8: ACT (Subtask 3: pick up tomato sauce)
  start_subtask("pick up the tomato sauce") →
    VLA control loop → pick up tomato sauce

Step 9: ACT (Subtask 4: place in basket)
  start_subtask("place the tomato sauce in the basket") →
    VLA control loop → place in basket

Step 10: EVALUATE (final)
  look("are both the alphabet soup and the tomato sauce in the basket?") →
    VLM confirms: "both items are in the basket"
  Success → reply to user "done: both items placed in the basket"

=== Failure handling ===
  If a round's EVALUATE fails → re-PERCEIVE → ACT → EVALUATE for that round
  Max 3 retries per round
  After 3 failures → report to user which round failed
    (e.g., "failed to place tomato sauce, gripper did not close")
```

### 5.2 Task B (Chain Decomposition): put the black bowl in the bottom drawer of the cabinet and close it

**LIBERO task ID**: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`

**LLM decomposition focus**: Different action types chained in sequence (open drawer → pick bowl → place in drawer → close drawer). Each subtask has a different action pattern, and there are dependency relationships (drawer must be opened before placing, placing must complete before closing).

```
Step 1: PREPARE
  Agent receives user instruction → model_ensure confirms ready
  LIBERO loads KITCHEN_SCENE4

Step 2: PERCEIVE
  look("describe the scene: where is the black bowl and the cabinet?") →
    VLM returns: "black bowl on the table at [0.3, 0.0, 0.08],
                  cabinet with bottom drawer at [0.6, -0.2, 0.15], drawer is closed"

Step 3: REASON (LLM task decomposition)
  LLM analyzes instruction "put the black bowl in the bottom drawer ... and close it" →
    Identifies dependency chain: must open drawer → place inside → close drawer
    Decomposes into 4 sequential subtasks:
      Subtask 1: open the bottom drawer of the cabinet
      Subtask 2: pick up the black bowl
      Subtask 3: place the black bowl in the drawer
      Subtask 4: close the bottom drawer
    Difficulty: hard (4-step chain with dependencies)
    Route: default/hard (action_hz=20, max_steps=500)

Step 4: ACT (Subtask 1: open drawer)
  start_subtask("open the bottom drawer of the cabinet",
                target={"object": "bottom_drawer", "position": [0.6, -0.2, 0.15]}) →
    VLA control loop → approach drawer handle → grasp → pull outward
    StepLimit(500) termination

Step 5: EVALUATE (intermediate check 1)
  look("is the bottom drawer open?") →
    VLM confirms: "the bottom drawer is open"
  Pass → continue

Step 6: ACT (Subtask 2: pick up bowl)
  start_subtask("pick up the black bowl from the table") →
    VLA control loop → approach bowl → descend → close gripper → lift

Step 7: ACT (Subtask 3: place in drawer)
  start_subtask("place the black bowl in the open drawer") →
    VLA control loop → move above drawer → descend into drawer → open gripper

Step 8: EVALUATE (intermediate check 2)
  look("is the black bowl inside the drawer?") →
    VLM confirms: "the black bowl is in the bottom drawer"
  Pass → continue

Step 9: ACT (Subtask 4: close drawer)
  start_subtask("close the bottom drawer of the cabinet") →
    VLA control loop → approach drawer → push inward

Step 10: EVALUATE (final)
  look("is the drawer closed with the bowl inside?") →
    VLM confirms: "the bottom drawer is closed"
  Success → reply to user "done: bowl placed in drawer and drawer closed"

=== Failure handling ===
  Subtask failure → rollback strategy based on current state:
    - Subtask 1 fails (open drawer) → retry opening
    - Subtask 3 fails (place in drawer) → may need to re-pick bowl (retry from Subtask 2)
    - Subtask 4 fails (close drawer) → retry closing
  Max 3 retries per subtask
  After 3 failures → report to user with current state
    (e.g., "drawer is open, bowl is on the table, failed to pick up bowl")
```

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Metric | Target |
|--------|--------|
| VLA control loop latency | < 20ms / step |
| VLM perception latency | < 2s / frame |
| LLM planning latency | < 10s / decision |
| End-to-end task (simple pick-and-place) | < 60s |

### 6.2 Concurrency

- All I/O uses asyncio; no threading.Lock in hot paths
- Single event loop handles all coroutines
- Blocking VLA predict() calls wrapped with `asyncio.to_thread()`

### 6.3 Reliability

| Scenario | Behavior |
|----------|----------|
| VLA service unreachable | Retry 3 times, then report to user |
| LLM API timeout | Exponential backoff retry, fall back to local model |
| Subtask execution failure | PRAE retry (max 3 times), report to user after 3 failures |
| E-Stop triggered | All motion stops immediately, manual reset required |

---

## 7. Verification Criteria

Two LIBERO-10 (Long) tasks validating two LLM decomposition patterns + VLA step-by-step execution.

### 7.1 Task A: Repeat Decomposition (Mock + LIBERO)

**LIBERO task**: `put both the alphabet soup and the tomato sauce in the basket` (LIVING_ROOM_SCENE2)

**Verification focus**: LLM can decompose "both X and Y" into 2 rounds of the same pick-and-place pattern, with re-perception between rounds.

| Mode | Pass Criteria |
|------|--------------|
| **Mock** | LLM correctly decomposes into 4 subtasks (pick soup → place → pick sauce → place), PERCEIVE between rounds, full PRAE loop execution |
| **LIBERO** | Franka Panda in LIBERO simulation sequentially places 2 objects in basket, VLM intermediate check + final evaluation confirm success |

### 7.2 Task B: Chain Decomposition (Mock + LIBERO)

**LIBERO task**: `put the black bowl in the bottom drawer of the cabinet and close it` (KITCHEN_SCENE4)

**Verification focus**: LLM can identify dependencies between actions (must open drawer before placing, must place before closing) and decompose into 4 different-typed sequential subtasks.

| Mode | Pass Criteria |
|------|--------------|
| **Mock** | LLM correctly decomposes into 4 subtasks (open drawer → pick bowl → place in drawer → close drawer), identifies dependency order, full PRAE loop execution |
| **LIBERO** | Franka Panda in LIBERO simulation completes the full chain: open drawer → pick bowl → place inside → close drawer, with VLM intermediate checks at key points (drawer open? bowl inside?) + final evaluation |

### 7.3 Common Acceptance Criteria

- LLM correctly classifies both tasks as hard difficulty and selects the corresponding route
- Each subtask is executed by an independent VLA control loop; LLM makes orchestration decisions between subtasks
- Automatic retry on failure (max 3 times); after 3 failures, report current state and which step failed
- Emergency stop tool can interrupt execution at any time
- Mock mode requires no GPU; LIBERO mode only requires GPU for VLA inference service

---

## 8. Future Scope

The following features will be implemented in post-MVP versions:

| Feature | Description |
|---------|-------------|
| **Multi-Robot** | Multiple robots coordinated via MessageBus with role specialization (scout / manipulator / monitor) |
| **Voice Input** | Onboard microphone (local Whisper ASR) + Web Voice (browser Web Speech API) |
| **Input Routing** | Unified Input Router normalizing all input channels to ChannelMessage with priority and rate limiting |
| **LoRA Hot-Swap** | Runtime VLA LoRA weight switching, per-subtask specialization (< 2s swap) |
| **Model Lifecycle** | Separate process managing model load/unload/VRAM allocation/health monitoring |
| **OTA Updates** | Incremental pull of new adapter/model versions with checksum verification and archival |
| **Autonomous Ops** | Patrol routes, task auto-claim, demo show mode, interactive idle mode |
| **Sim Fork-Compare** | Parallel simulation of multiple strategies, compare results, promote winner to main environment |
| **Web UI** | Browser interface: real-time simulation view + command input + status dashboard |
| **Multi-Channel** | Enable nanobot built-in channels (Telegram, Slack, Feishu, Email, etc.) |
| **4-Layer Safety** | Safety zone fencing + audit logging (extending E-Stop + Velocity) |
| **Dynamic Routing** | Multiple route profiles (cautious, fast_manipulation), 3 difficulty levels, per-route LoRA |
| **VLM Terminator** | VLM-based subtask completion detection for complex multi-step tasks |
| **Navigation** | Mobile base navigation, requires switching to RoboCasa (PandaMobile) or Habitat simulation |
| **Real Hardware** | Unitree G1 + GR00T N1.6 (ZMQ adapter) |

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **VLA** | Vision-Language-Action model. Input: image + text instruction. Output: robot action sequence. |
| **VLM** | Vision-Language Model. Input: image + text question. Output: text description. |
| **LLM** | Large Language Model. Text-based reasoning and planning. |
| **PRAE** | Prepare → Perceive → Reason → Act → Evaluate. The agent's core execution loop. |
| **Intention** | Logical layer: fast intent recognition and task framing. |
| **Cognition** | Logical layer: task reasoning/planning, outputs structured plan. Contains an optional Perception sub-module. |
| **Perception** | Optional sub-module of Cognition. Produces scene descriptions from visual input (VLM / CV / ground-truth). Can be enabled/disabled independently. |
| **Action** | Logical layer: high-frequency robot control loop (VLA + safety + actuation). |
| **Tool** | An async function registered with the agent that the LLM can invoke via tool_call. Robot actions and software operations share the same abstraction. |
| **LoopManager** | Component managing VLA control loops: start/wait/terminate/stats. |
| **Terminator** | Strategy that decides when a VLA control loop should stop. |
| **Action Horizon** | Number of action steps a VLA produces per inference call (chunking size). |
| **Route** | YAML config specifying control parameters (frequency, step limits, termination strategies) for a given task difficulty. |
| **LIBERO** | MuJoCo-based tabletop manipulation simulation platform. Franka Panda fixed-base, 130+ tasks (5 suites), 7-dim action space (3D position + 3D rotation + gripper). Natively supported by SmolVLA/pi0/openpi. |
| **Nanobot** | Ultra-lightweight Python agent framework (~4,000 LOC). The foundation of this project. |
| **SKILL.md** | Skill description file with YAML frontmatter + Markdown instructions that the agent can load and execute step-by-step. |
