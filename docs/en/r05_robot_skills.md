# R05 — Robot Skills (SKILL.md)

## Mental Model

Before r05, the agent figures out "how to grasp a mug" from scratch
every time. That's wasteful — manipulation strategies are reusable.

In r05, we store recipes as **SKILL.md files** and load them on demand:

```
System prompt (Layer 1 — always visible, ~30 tokens each):
  "Skills: top-grasp, side-grasp, pour, precision-place"

LLM: "I need to pick up the mug. It's tall with a handle."
LLM: load_skill("side-grasp")

Tool result (Layer 2 — injected on demand):
  <skill name="side-grasp">
    1. Pre-position to the SIDE of the object
    2. Open gripper
    3. Approach horizontally
    4. Grasp, lift
    Failure recovery: ...
  </skill>

LLM now follows the recipe step by step.
```

## What Changed

| Component | r04 | r05 |
|-----------|-----|-----|
| Tools | look, move, grasp, todo, perceive | + **load_skill** |
| New concept | — | **Two-layer skill injection** |
| Knowledge | Hardcoded in system prompt | Loaded on demand from files |

## The Two Layers

| | Layer 1 | Layer 2 |
|---|---------|---------|
| Where | System prompt | tool_result |
| When | Always | On demand |
| Cost | ~30 tokens/skill | ~200-400 tokens/skill |
| Content | Name + one-line description | Full procedure + examples |

This scales: 100 skills = ~3000 tokens in system prompt (just names).
Only the skill actually needed gets fully loaded.

## SKILL.md Format

```markdown
---
name: top-grasp
description: Grasp from above. For flat/low objects on surfaces.
tags: grasp, manipulation
---

# Top Grasp Skill

## When to Use
- Fruit on counter, plates, boxes...

## Procedure
1. Move 15cm above target
2. Open gripper
3. Descend to object
4. Close gripper
5. Lift 15cm

## Failure Recovery
- If grasp fails: adjust, retry (max 2)
```

YAML frontmatter for metadata, markdown body for instructions.

## Four Built-in Skills

| Skill | Use For | Key Technique |
|-------|---------|--------------|
| `top-grasp` | Apples, plates, flat objects | Approach from above, descend, grasp |
| `side-grasp` | Mugs, bottles, handled objects | Approach from side, grasp body/handle |
| `pour` | Transfer liquid between containers | Grasp source, position above target, tilt |
| `precision-place` | Table setting, stacking | Align above target, slow descent, verify |

## Demo

```
r05 >> pick up the mug and place it on the plate

[perceive] identify objects and grasp strategies
  [subagent] look: what objects are there and their shapes?
  => blue mug: shape=tall-handle. Recommended: side-grasp
  => red apple: shape=round-flat. Recommended: top-grasp
  [subagent] done

[todo]
[ ] #1: Load side-grasp skill for mug
[ ] #2: Pick up mug using side-grasp
[ ] #3: Load precision-place skill
[ ] #4: Place mug on plate
[ ] #5: Verify

[load_skill] side-grasp
<skill name="side-grasp">
  1. Pre-position to the SIDE...
  2. Open gripper...
  ...
</skill>

[move] {"position": [0.50, 0.42, 0.82]}   # beside mug
[grasp] {"action": "open"}
[move] {"position": [0.50, 0.32, 0.82]}   # approach
[grasp] {"action": "close"}                 # grab mug
[move] {"position": [0.50, 0.32, 0.92]}   # lift

[load_skill] precision-place
<skill name="precision-place">
  1. Position above target...
  ...
</skill>

[move] {"position": [0.45, -0.15, 0.92]}  # above plate
[move] {"position": [0.45, -0.15, 0.82]}  # descend
[grasp] {"action": "open"}                  # release
[move] {"position": [0.45, -0.15, 0.92]}  # retract

[look] is the mug on the plate?
=> blue mug is on white plate. Done!
```

## Why Skills Matter for Robotics

Real robots need different strategies for different objects:

| Object | Wrong Strategy | Right Strategy |
|--------|---------------|----------------|
| Apple on counter | Side-grasp (rolls away) | Top-grasp (pin from above) |
| Mug with handle | Top-grasp (hits rim) | Side-grasp (grab body) |
| Fragile glass | Fast grab (breaks) | Precision-place (gentle) |

Without skills, the agent guesses each time. With skills:
- Consistent, tested manipulation procedures
- Failure recovery built into each recipe
- New skills added by dropping a SKILL.md file — no code changes

## Adding Custom Skills

Create a new directory under `skills/` with a SKILL.md:

```
skills/
  my-new-skill/
    SKILL.md    # frontmatter + instructions
```

The SkillLoader picks it up automatically on next run.

## REPL Commands

- `skills` — list available skills
- `todos` — print todo list
- `reset` — reset environment

## What's Next

In **r06**, we add context compression — the three-layer system that
keeps long-horizon tasks from running out of context window. Old tool
results get replaced with placeholders, and when context gets too large,
the LLM summarizes the conversation so far.
