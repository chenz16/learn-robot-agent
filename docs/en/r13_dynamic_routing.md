# R13 — Multi-Frequency Dynamic Routing

## Mental Model

In r01-r12, every model call runs at the LLM's speed (~0.1Hz). The LLM
decides, VLM looks, LLM decides, VLA acts. Everything waits for the
slowest component.

Real robots need **three concurrent frequency loops**:

```
┌─ Planning Loop (LLM) ──────── 0.1 Hz ──── REMOTE ────────────────┐
│  Decompose task, select route, monitor progress                    │
│                                                                    │
│  ┌─ Perception Loop (VLM) ──── 1-5 Hz ──── LOCAL ──────────────┐ │
│  │  Scene understanding, termination detection                   │ │
│  │                                                               │ │
│  │  ┌─ Control Loop (VLA) ──── 2-50 Hz ──── LOCAL ────────────┐ │ │
│  │  │  obs → VLA.predict() → step_delta() → repeat             │ │ │
│  │  │  Runs until termination signal                            │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

## What Changed

| Component | r12 | r13 |
|-----------|-----|-----|
| Model calls | All synchronous, ~0.1Hz | **Three concurrent frequency loops** |
| VLA | Single HTTP call per action | **Continuous loop at 2-50Hz** |
| VLM | On-demand look() | **Periodic loop at 1-5Hz** |
| Routing | Hardcoded | **YAML config per difficulty** |
| VLA adapter | Direct HTTP | **Abstraction (Mock/HTTP/GR00T/OpenPI)** |
| Termination | Manual (LLM decides) | **Automatic (5 strategies)** |
| LoRA | Not supported | **Hint in route config, adapter loads** |
| Tools | 20 | + **start_subtask, check_loops, wait_subtask, stop_loops, set_route, list_routes, classify_task, loop_stats** |

## Three Frequency Tiers

| Tier | Component | Frequency | Location | Latency Budget |
|------|-----------|-----------|----------|---------------|
| 1 | VLA Control | 2-50 Hz | LOCAL GPU | < 20ms |
| 2 | VLM Perception | 0.5-5 Hz | LOCAL/EDGE | 200ms-2s |
| 3 | LLM Planning | 0.1 Hz | REMOTE | 1-10s |
| 4 | Evaluator | 0.1-1 Hz | LOCAL/REMOTE | 1-3s |

## Dynamic Routing

Task difficulty determines which models run and at what frequency.
Routes are defined in YAML files under `routes/`:

```yaml
# routes/default.yaml
routes:
  easy:
    planner: null           # skip LLM replanning
    perception_hz: 1
    action_hz: 20
    termination: [position_threshold, step_limit]
    vla_lora: null          # base model

  medium:
    planner: local-vlm
    perception_hz: 2
    action_hz: 20
    termination: [vlm_check, step_limit]

  hard:
    planner: remote-llm
    perception_hz: 5
    action_hz: 50
    termination: [vlm_check, gripper_state, step_limit]
    vla_lora: task-specific  # load fine-tuned LoRA
    evaluator: remote-llm
```

Three routes included: `default`, `fast_manipulation`, `cautious`.

## Difficulty Classification

```
classify_difficulty(instruction, scene)
  │
  ├── Easy: known object + simple verb ("pick up red apple")
  │   Route: VLA@20Hz, VLM@1Hz, no LLM replanning
  │
  ├── Medium: known type + new parameters ("pick up the blue mug")
  │   Route: VLA@20Hz, VLM@2Hz, VLM as quick planner
  │
  └── Hard: compound or novel ("pour water, then place mug on shelf")
      Route: VLA@50Hz, VLM@5Hz, full LLM evaluation
```

## VLA Adapter Abstraction

Support current and future VLAs through a common interface:

```
BaseVLAAdapter
  ├── predict(observation, instruction) → actions[]
  ├── activate(lora_name=None) → signal server to start
  ├── deactivate() → signal server to idle
  ├── reset()
  ├── get_action_horizon() → int (action chunking)
  └── health_check() → bool

MockVLAAdapter    — synthetic deltas toward target (MOCK mode)
HTTPVLAAdapter    — POST /predict (current GR00T proxy pattern)
GR00TStub         — ZMQ + msgpack protocol (documented)
OpenPIStub        — WebSocket + msgpack_numpy (documented)
```

## Termination Strategies

VLAs do NOT self-terminate. External supervision is required:

| Strategy | When It Fires | Used For |
|----------|---------------|----------|
| `position_threshold` | ee within 0.05m of target | Move-to-target |
| `gripper_state` | Gripper matches expected state | Grasp/release |
| `vlm_check` | VLM says subtask is done | Complex tasks |
| `step_limit` | Max N steps reached | Safety fallback |
| `composite` | Any/all of the above | Combined |

## Service Lifecycle

VLA and VLM are **resident processes** (always running):

```
Agent starts → services already running (resident, idle)
  start_subtask() → activate("vla", lora="apple-v2") → inference begins
  subtask completes → deactivate("vla") → back to idle
  next subtask → activate("vla", lora="mug-v1") → LoRA hot-swap
Agent exits → services still running (ready for next session)
```

## LoRA Integration

The routing config specifies `vla_lora` per difficulty level:
- `null` — use base VLA model
- `"task-specific"` — load user's fine-tuned LoRA
- `"apple-pick-v2"` — specific named LoRA adapter

The agent sends the LoRA name via `adapter.activate(lora="...")`.
The VLA server handles the actual weight loading. The agent's
responsibility ends at telling the server which LoRA to use.

## Demo

```
r13 >> put the apple on the plate

[classify_task] "pick up red apple" → easy
[set_route] default/easy → VLA@20Hz, VLM@1Hz

[start_subtask] instruction="move to red apple", target={name: "red apple", pos: [0.45, 0.12, 0.82]}
Subtask a1b2 started (route=default/easy, action@20Hz, perception@1Hz)

<loop-results>
[subtask:a1b2] completed after 22 steps (1.1s)
  termination: position_threshold (0.03m < 0.05m)
  achieved action_hz=19.8
</loop-results>

[start_subtask] instruction="close gripper"
<loop-results>
[subtask:c3d4] completed after 3 steps (0.15s)
  termination: gripper_state (closed, grasped 'red apple')
</loop-results>

[start_subtask] instruction="move to white plate"
<loop-results>
[subtask:e5f6] completed after 28 steps (1.4s)
  termination: position_threshold
</loop-results>

[start_subtask] instruction="open gripper"
<loop-results>
[subtask:g7h8] completed after 2 steps (0.1s)
  termination: gripper_state (opened, released onto 'white plate')
</loop-results>

[look] "apple on plate?" → Yes!

Total: 55 VLA steps, 2.75s, all terminations automatic.
```

## REPL Commands

- `loops` — show loop status and stats
- `routes` — list available routing configs
- `stats` — performance statistics
- `tasks` — show task board
- `bg` — list background jobs
- `skills` — list skills
- `tokens` — token estimate
- `/compact` — force compression
- `reset` — clear everything

## Summary: r01-r13 Stack

| Layer | Lesson | What It Adds |
|-------|--------|-------------|
| Core | r01 | Agent loop + look |
| Action | r02 | Move + grasp tools |
| Planning | r03 | Task decomposition |
| Perception | r04 | Subagent observer |
| Skills | r05 | SKILL.md recipes |
| Memory | r06 | Context compression |
| Persistence | r07 | Disk-based task board |
| Concurrency | r08 | Background threads |
| Teams | r09 | Multi-robot coordination |
| Safety | r10 | E-stop + safety zones |
| Autonomy | r11 | Patrol + auto-claim |
| Simulation | r12 | Parallel sandboxes |
| **Routing** | **r13** | **Multi-freq loops + dynamic routing** |
