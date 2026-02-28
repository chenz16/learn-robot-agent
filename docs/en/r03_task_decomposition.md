# R03 — Task Decomposition

## Mental Model

In r02, the agent reacts: "put the apple on the plate" triggers a
sequence of look/move/grasp calls, but the agent has no structured plan.

In r03, the agent **plans first, then executes**:

```
"set the table" ->
  [ ] 1. observe scene
  [ ] 2. pick up fork, place near plate
  [ ] 3. pick up napkin, place near plate
  [ ] 4. pick up mug, place near plate
  [ ] 5. verify table is set
```

The TodoManager gives the agent self-awareness about progress.

## What Changed

| Component | r02 | r03 |
|-----------|-----|-----|
| Tools | look, move, grasp | look, move, grasp, **todo** |
| Loop | unchanged | +nag reminder (3 lines) |
| Behavior | reactive | **plan-then-execute** |

The loop change is minimal — just tracking `rounds_since_todo` and
injecting a reminder if the agent forgets to update its plan.

## The TodoManager

Same pattern as learn-claude-code s03:

```python
class TodoManager:
    def update(self, items) -> str:
        # Validate: max 20 items, valid statuses, only 1 in_progress
        self.items = validated
        return self.render()

    def render(self) -> str:
        # [ ] pending  [>] in_progress  [x] completed
```

The LLM calls `todo` with the full item list each time. This is a
**replace-all** operation, not incremental — simpler to implement and
harder for the model to get wrong.

## Nag Reminder

If the agent goes 3 rounds without updating todos, we inject:

```python
if rounds_since_todo >= 3:
    results.insert(0, {
        "type": "text",
        "text": "<reminder>Update your todo list.</reminder>",
    })
```

This prevents drift — the agent stays on plan.

## Demo: "set the table"

```
r03 >> set the table with the fork and mug next to the plate

[todo]
[ ] #1: Observe scene to locate all objects
[ ] #2: Pick up fork and place near plate
[ ] #3: Pick up mug and place near plate
[ ] #4: Verify table is set

(0/4 completed)

[todo]
[>] #1: Observe scene to locate all objects
[ ] #2: Pick up fork and place near plate
...

[look] {}
Scene observation:
  - red apple: pos=[0.45, 0.12, 0.82], on counter
  - white plate: pos=[0.45, -0.15, 0.80], on counter
  - fork: pos=[0.55, -0.20, 0.81], on counter
  - napkin: pos=[0.60, 0.00, 0.81], on counter
  - blue mug: pos=[0.50, 0.30, 0.82], on counter

[todo]
[x] #1: Observe scene to locate all objects
[>] #2: Pick up fork and place near plate
...

[move] {"target": "fork"}
Moved: ... near 'fork'

[grasp] {"action": "close"}
Gripper closed. Grasped 'fork'.

[move] {"target": "white plate"}
Moved: ... Carrying fork.

[grasp] {"action": "open"}
Gripper opened. Released 'fork' onto 'white plate'.

[todo]
[x] #1: Observe scene
[x] #2: Pick up fork and place near plate
[>] #3: Pick up mug and place near plate
[ ] #4: Verify table is set

(2/4 completed)

... (continues until all done)
```

## Why This Matters for Robotics

Real-world robot tasks are rarely single-step:

| Simple (r02) | Complex (r03) |
|--------------|---------------|
| "pick up the apple" | "set the table for dinner" |
| "open the gripper" | "clean up all items on the counter" |
| 1-3 tool calls | 10-30+ tool calls |

Without a todo list, the agent loses track after 5-6 steps.
With a todo list, it can handle 20+ step sequences reliably.

## Key Insight

Traditional robotics handles multi-step tasks with:
- Behavior trees (manually designed)
- State machines (manually coded transitions)
- Task and motion planning (TAMP — computationally expensive)

Agent robotics:
- LLM decomposes the goal into subtasks (natural language)
- Executes each subtask with the same tool set
- Tracks progress with a simple todo list
- Handles failures by re-planning ("that grasp failed, let me try again")

The todo list is the agent's equivalent of a behavior tree —
but it writes it itself at runtime.

## REPL Commands

- Type a task to execute (e.g., "set the table")
- `todos` — print current todo list
- `reset` — reset environment, todos, and history

## What's Next

In **r04**, we add a perception subagent — a child agent with fresh
context that specializes in scene analysis. The parent delegates
"understand this scene" to the child, keeping its own context clean
for planning and execution.
