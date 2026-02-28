# learn-robot-agent

A progressive 13-lesson tutorial for building robot agents — from a single loop to multi-frequency dynamic routing.

Inspired by [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code), applied to robotics.

## Core Idea

> One loop, a few tools, and an LLM — that's a robot agent.

```python
while True:
    response = LLM(messages, tools=[look, move, grasp, ...])
    if no tool_call: break
    execute_tools(response) → append results → loop
```

The loop never changes. Every lesson adds tools and pre/post-processing — the core stays the same.

## Lessons

Each lesson is a **self-contained Python file** (~400-1100 lines). No cross-file imports.
Every file works in **MOCK mode** without external services.

| # | File | Topic | Tools | Key Concept |
|---|------|-------|-------|-------------|
| r01 | [r01_agent_loop.py](agents/r01_agent_loop.py) | Agent Loop + Look | 1 | The immutable while loop |
| r02 | [r02_tool_use.py](agents/r02_tool_use.py) | Tool Use | 3 | Dispatch map + stateful env |
| r03 | [r03_task_decomposition.py](agents/r03_task_decomposition.py) | Task Decomposition | 4 | TodoManager + nag reminder |
| r04 | [r04_perception_subagent.py](agents/r04_perception_subagent.py) | Perception Subagent | 5 | Fresh-context child agent |
| r05 | [r05_robot_skills.py](agents/r05_robot_skills.py) | Robot Skills | 6 | SKILL.md two-layer injection |
| r06 | [r06_context_compact.py](agents/r06_context_compact.py) | Context Compression | 7 | Three-layer compression |
| r07 | [r07_persistent_tasks.py](agents/r07_persistent_tasks.py) | Persistent Task Board | 10 | JSON files + dependency graph |
| r08 | [r08_background_tasks.py](agents/r08_background_tasks.py) | Background Tasks | 13 | Threaded execution + notification queue |
| r09 | [r09_multi_robot_teams.py](agents/r09_multi_robot_teams.py) | Multi-Robot Teams | 18 | JSONL inboxes + RobotTeamManager |
| r10 | [r10_safety_protocols.py](agents/r10_safety_protocols.py) | Safety Protocols | 19 | E-stop, zones, velocity, audit |
| r11 | [r11_autonomous_patrol.py](agents/r11_autonomous_patrol.py) | Autonomous Patrol | 22 | Waypoint patrol + idle-cycle auto-claim |
| r12 | [r12_parallel_sim.py](agents/r12_parallel_sim.py) | Parallel Sim | 20 | Fork/compare/promote sim sandboxes |
| r13 | [r13_dynamic_routing.py](agents/r13_dynamic_routing.py) | Dynamic Routing | 28 | Multi-freq loops + VLA adapters + YAML routes |

Detailed docs for each lesson: [docs/en/](docs/en/)

## Progression

```
r01  Loop + Look          ← start here
 │
r02  + Move, Grasp        ← robot can act
 │
r03  + Task Decomp        ← robot can plan
 │
r04  + Perception Sub     ← thorough observation
 │
r05  + Skills (SKILL.md)  ← reusable recipes
 │
r06  + Compression        ← infinite sessions
 │
r07  + Persistent Tasks   ← survives restarts
 │
r08  + Background Threads ← async perception + action
 │
r09  + Multi-Robot Teams  ← scout, manipulator, monitor
 │
r10  + Safety Pipeline    ← e-stop, zones, audit
 │
r11  + Autonomous Patrol  ← finds work itself
 │
r12  + Parallel Sims      ← try strategies in sandbox
 │
r13  + Dynamic Routing    ← VLA@50Hz / VLM@5Hz / LLM@0.1Hz
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/chenz16/learn-robot-agent.git
cd learn-robot-agent

# 2. Install
pip install -r requirements.txt

# 3. Configure
cd agents
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY (or ANTHROPIC_BASE_URL for proxy)

# 4. Run any lesson (each is self-contained)
python r01_agent_loop.py          # simplest — one tool
python r05_robot_skills.py        # mid — skills + subagent
python r12_parallel_sim.py        # full — everything
```

All lessons work in **MOCK mode** by default (no external services needed).
Set `SIM_URL`, `VLM_URL`, `VLA_URL` in `.env` for real robot backends.

## Architecture

```
User: "put the apple on the plate"
         │
         ▼
┌─────────────────────────────────────┐
│          Agent Loop (LLM)           │
│                                     │
│  tools: look, move, grasp, perceive │
│         load_skill, task_*, bg_*    │
│         spawn_robot, e_stop, ...    │
│                                     │
│  pre-loop: drain bg notifications   │
│            drain inbox messages     │
│            micro_compact old results│
│  post-loop: auto_compact if needed  │
│             idle-cycle (r11+)       │
└────────────┬────────────────────────┘
             │ HTTP (real mode)
        ┌────┼────┐
        ▼    ▼    ▼
      VLM  VLA   Sim         MockRobotEnv
     :8010 :8020 :8030       (mock mode, default)
```

## Project Structure

```
learn-robot-agent/
├── agents/
│   ├── .env.example          # API key + service URLs
│   ├── r01_agent_loop.py     # ... through ...
│   └── r13_dynamic_routing.py
├── skills/
│   ├── top-grasp/SKILL.md    # top-down pick recipe
│   ├── side-grasp/SKILL.md   # side approach for mugs
│   ├── pour/SKILL.md         # pour liquid recipe
│   └── precision-place/SKILL.md
├── routes/
│   ├── default.yaml          # standard routing config
│   ├── fast_manipulation.yaml
│   └── cautious.yaml
├── docs/en/                  # detailed lesson docs
├── requirements.txt
└── README.md
```

## Three Operating Modes

| Mode | Config | What Happens |
|------|--------|-------------|
| **MOCK** (default) | No URLs set | Hardcoded scene, simulated physics |
| **SIM-only** | `SIM_URL` + `VLM_URL` | Real sim renders + VLM analysis |
| **FULL** | All 3 URLs | Sim + VLM + VLA (real robot control) |

## Target Hardware

The tutorials are designed around:
- **Robot**: Unitree G1 humanoid (30-DOF, walk + manipulate)
- **VLA**: GR00T N1.6 (`nvidia/GR00T-N1.6-G1-PnPAppleToPlate`)
- **VLM**: Qwen2.5-VL-7B-Instruct
- **Sim**: G1 MuJoCo WholeBodyControl

But MOCK mode works without any of these.

## Prerequisites

- Python 3.10+
- An Anthropic API key (or compatible proxy via `ANTHROPIC_BASE_URL`)
- `pip install anthropic python-dotenv requests pyyaml`

## License

MIT
