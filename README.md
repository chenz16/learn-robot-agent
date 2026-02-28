# learn-robot-agent

A progressive tutorial for building robot agents — from a single loop to multi-robot teams.

Inspired by [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code), applied to robotics.

## Core Idea

> One loop, a few tools, and an LLM — that's a robot agent.

```
while True:
    response = LLM(messages, tools=[look, grasp, move, navigate, evaluate])
    if no tool_call: break
    execute_tools(response) → append results → loop
```

## Lessons

| # | Topic | What You Learn |
|---|-------|---------------|
| r01 | Agent Loop + Look | The core loop with a VLM `look` tool |
| r02 | Tool Use (Grasp/Move) | Add VLA and Sim tools |
| r03 | Task Decomposition | Break "pick apple" into subtasks |
| r04 | Perception Subagent | Spawn a fresh-context observer |
| r05 | Robot Skills (SKILL.md) | Load manipulation recipes on demand |
| r06 | Context Compression | Keep long-horizon tasks in context |
| r07 | Persistent Task Board | Multi-step mission tracking |
| r08 | Async Sensors & Inference | Background VLA + sensor polling |
| r09 | Multi-Robot Teams | Coordinate multiple agents |
| r10 | Safety Protocols | E-stop, collision, force limits |
| r11 | Autonomous Patrol | Idle → scan → claim → execute |
| r12 | Parallel Sim Environments | Isolated sim instances per task |

## Quick Start

```bash
cd agents
cp .env.example .env  # add your API key
python r01_agent_loop.py
```

## Architecture

```
User: "put the apple on the plate"
         │
         ▼
┌─────────────────────┐
│   Agent Loop (LLM)  │  ← One while loop
│   tools: look, grasp│
│   move, evaluate ... │
└────────┬────────────┘
         │ HTTP calls
    ┌────┼────┐
    ▼    ▼    ▼
  VLM  VLA  Sim
 :8010 :8020 :8030
```
