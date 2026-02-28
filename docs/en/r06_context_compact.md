# R06 — Context Compression

## Mental Model

Robot tasks are long. "Set the table" might take 30+ tool calls.
Each `look()` returns ~100 tokens, each skill ~300, each perceive ~500.
Without compression, context fills up after ~15 subtasks.

Three layers solve this:

```
Every turn:
  Layer 1: micro_compact (silent)
    → Old tool results become "[Previous: used look]"
    → Keeps last 3 results intact
    → Runs every turn, no cost

  If tokens > threshold:
    Layer 2: auto_compact (automatic)
      → Save full transcript to .transcripts/
      → LLM summarizes: task progress + robot state
      → Replace all messages with summary
      → Preserves: todos, robot state, remaining plan

  Model can also call:
    Layer 3: compact tool (manual)
      → Same as auto_compact, triggered by the agent
      → Use when "I've been running a while, let me compress"
```

## What Changed

| Component | r05 | r06 |
|-----------|-----|-----|
| Tools | 6 | + **compact** |
| Loop | standard | + **three-layer compression** |
| Context | fills up, session limited | **infinite sessions** |

## The Three Layers

### Layer 1: micro_compact (every turn, free)

```python
def micro_compact(messages):
    # Find all tool_result entries
    # Keep last 3 intact
    # Replace older ones: content -> "[Previous: used {tool_name}]"
```

This is invisible to the agent. Old scene descriptions, skill loads,
and action results get replaced with one-line placeholders.

Before: `"Scene: kitchen counter with red apple at [0.45, 0.12...]..."` (100+ tokens)
After: `"[Previous: used look]"` (5 tokens)

### Layer 2: auto_compact (when tokens > threshold)

```python
def auto_compact(messages):
    # 1. Save full transcript to .transcripts/transcript_*.jsonl
    # 2. Ask LLM to summarize, including:
    #    - Original task
    #    - Completed subtasks
    #    - Remaining subtasks
    #    - Current robot state (ee_pos, gripper, holding)
    # 3. Replace all messages with [summary] + ack
```

The summary is **robot-aware** — it preserves ee position, gripper state,
held objects, and todo progress. The agent can continue seamlessly.

### Layer 3: compact tool (model decides)

The agent can call `compact()` explicitly when it feels context is heavy.
Same mechanism as auto_compact, just triggered manually.

## Robot-Specific Summary

When compressing, we inject current state:

```
Current state: Robot: ee=[0.45, -0.15, 0.82], gripper=open, holding=nothing
Todos:
[x] #1: Pick fork, place on plate
[x] #2: Pick napkin, place on plate
[>] #3: Pick mug, place on plate
[ ] #4: Verify all items placed
```

This means after compression, the agent knows exactly where it left off.

## Demo

```
r06 >> set the table with everything on the plate

... (20+ tool calls: perceive, plan, pick fork, place, pick napkin, place...) ...

[auto_compact] ~14000 tokens > threshold
[transcript saved: transcript_1709123456.jsonl]

[Context compressed]
Summary: User asked to set table. Fork and napkin placed on plate.
Mug pickup next using side-grasp. Apple remaining after.
Robot: ee=[0.45, -0.15, 0.82], gripper=open, holding=nothing

(Agent continues seamlessly with mug pickup)
```

## REPL Commands

- `tokens` — show estimated token count
- `/compact` — force compression now
- `todos` — print todo list
- `skills` — list skills
- `reset` — reset everything

## Why This Matters

| Without compression | With compression |
|--------------------|-----------------|
| ~15 subtasks max | Unlimited subtasks |
| Context error crashes session | Graceful degradation |
| Lose progress | Summary preserves state |
| Cannot do "clean the whole kitchen" | Multi-room, multi-hour tasks |

## What's Next

In **r07**, we add a persistent task board — tasks stored as JSON files
on disk that survive context compression and even agent restarts.
Combined with compression, the agent can handle multi-session missions.
