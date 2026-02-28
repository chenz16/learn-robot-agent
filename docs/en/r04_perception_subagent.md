# R04 — Perception Subagent

## Mental Model

Before r04, the parent agent does everything: perceive, plan, and act.
A thorough scene analysis (5+ look calls) fills up context fast.

In r04, we **delegate perception** to a child agent with fresh context:

```
Parent: "perceive(goal='what's on the table?')"
  |
  +---> Child (fresh messages=[])
        |  look("overview")
        |  look("where is each object?")
        |  look("reachability analysis")
        |  look("spatial relationships")
        |  look("obstacles?")
        |  -> "Summary: 5 objects on counter, apple at [0.45...]..."
        |
  <--- Only the summary returns
       Child context is discarded
```

Parent keeps a clean context. Child can be thorough.

## What Changed

| Component | r03 | r04 |
|-----------|-----|-----|
| Tools | look, move, grasp, todo | + **perceive** |
| New concept | — | **Subagent** (fresh context, summary return) |
| Perception | Quick `look` calls | `perceive` for thorough analysis, `look` for quick checks |

## Two Levels of Perception

| | `look` | `perceive` |
|---|--------|-----------|
| Context | Same as parent | Fresh (isolated) |
| Depth | 1 call, 1 answer | 3-5+ calls, comprehensive |
| Cost | Low tokens | Higher tokens (but isolated) |
| Use when | Quick check during execution | Before planning, after completion |

## The Subagent Pattern

From learn-claude-code s04, adapted for robotics:

```python
def run_perceive(goal: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh!

    for turn in range(15):
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,  # child's own context
            tools=CHILD_TOOLS,     # only 'look'
        )
        # ... execute look calls ...

        if response.stop_reason != "tool_use":
            break

    # Only the summary returns to parent
    return extract_text(response)
```

Key properties:
- **Fresh context**: `sub_messages = []` — child starts blank
- **Filtered tools**: child only gets `look` (no move/grasp/perceive)
- **Summary return**: only final text goes to parent
- **Safety limit**: max 15 turns (prevents runaway)

## Demo

```
r04 >> set the table with all items on the plate

[perceive] goal: identify all objects and plan table setting
  [subagent] spawned
  [subagent] look: (overview)
  [subagent] look: reachability of each object
  [subagent] look: spatial relationships between objects
  [subagent] look: current state of the plate
  [subagent] done (4 turns)

Scene Analysis Summary:
- 5 objects on counter: red apple, white plate, blue mug, fork, napkin
- Plate is empty at [0.45, -0.15, 0.80]
- All objects reachable (within 0.40m of workspace center)
- Suggested order: fork first (closest), then napkin, mug, apple
- No obstacles between robot and any object

[todo]
[ ] #1: Pick fork, place on plate
[ ] #2: Pick napkin, place on plate
[ ] #3: Pick mug, place on plate
[ ] #4: Pick apple, place on plate
[ ] #5: Verify all items on plate

... (executes each subtask) ...

[perceive] goal: verify all items are on the plate
  [subagent] spawned
  [subagent] look: what's on the plate?
  [subagent] done (2 turns)

Verification: fork, napkin, mug, and apple are all on the plate. Done!
```

## Why Subagents Matter for Robotics

Real robot perception is expensive:
- Multiple camera viewpoints (ego + third-person + depth)
- Object detection + pose estimation
- Spatial reasoning about reachability
- Safety checks (obstacles, collision paths)

All of this generates tokens. Without subagents, the parent's context
fills up fast. With subagents:
- Perception is thorough (child can take 5-10 turns)
- Planning stays clean (parent only sees the summary)
- Context budget is used efficiently

## What's Next

In **r05**, we add robot skills — reusable manipulation recipes stored
as SKILL.md files. Instead of figuring out "how to pick up a mug"
from scratch each time, the agent loads a skill on demand:
"grasp-from-above for tall objects, side-grasp for flat objects."
