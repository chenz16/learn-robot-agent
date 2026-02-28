# R02 — Robot Tools

## Mental Model

The agent loop from r01 didn't change at all. We just added tools to the
array and a dispatch map to route calls:

```python
TOOL_HANDLERS = {
    "look":  lambda **kw: run_look(...),
    "move":  lambda **kw: run_move(...),
    "grasp": lambda **kw: run_grasp(...),
}
```

In learn-claude-code s02, tools are `bash`, `read_file`, `write_file`, `edit_file`.
In learn-robot-agent r02, tools are `look`, `move`, `grasp`.

**Same pattern, different domain.**

## Architecture

```
User: "put the apple on the plate"
         |
         v
+------------------+
|   Agent Loop     |
|   (unchanged)    |
+--------+---------+
         |
    dispatch map
    +----+----+----+
    |    |    |    |
    v    v    v    v
  look  move grasp  (future tools...)
    |    |    |
    v    v    v
  VLM  Sim   VLA
```

## The Three Tools

| Tool | Purpose | Mock | Real |
|------|---------|------|------|
| `look` | Perceive the scene | Stateful scene description | Sim camera -> VLM |
| `move` | Move end-effector | Update position + carry held object | VLA -> Sim step |
| `grasp` | Gripper control | Pick/place with distance check | VLA -> Sim step |

## Stateful Mock Environment

The key difference from r01: the mock environment is now **stateful**.
Actions change the world, and subsequent `look` calls reflect those changes.

```python
class MockRobotEnv:
    def __init__(self):
        self.objects = {"red apple": ..., "white plate": ..., "blue mug": ...}
        self.ee_pos = [0.30, 0.0, 0.95]
        self.gripper = "open"
        self.holding = None

    def move(self, target):
        # Move ee_pos to target object
        # If holding something, it moves with the gripper

    def grasp(self, action):
        # "close" -> if near an object, pick it up
        # "open"  -> release held object onto nearest surface
```

This means the agent can complete a full pick-and-place task in mock mode:

```
r02 >> put the apple on the plate
[look] {}
Scene observation:
  - red apple: pos=[0.45, 0.12, 0.82], on counter
  - white plate: pos=[0.45, -0.15, 0.80], on counter
  - blue mug: pos=[0.50, 0.30, 0.82], on counter
Robot: ee=[0.30, 0.0, 0.95], gripper=open, holding=nothing

[move] {"target": "red apple"}
Moved end-effector: [0.30, 0.0, 0.95] -> [0.45, 0.12, 0.82] (0.22m)

[grasp] {"action": "close"}
Gripper closed. Grasped 'red apple' (distance was 0.000m).

[move] {"target": "white plate"}
Moved end-effector: [0.45, 0.12, 0.82] -> [0.45, -0.15, 0.80] (0.27m). Carrying red apple.

[grasp] {"action": "open"}
Gripper opened. Released 'red apple' onto 'white plate'.

[look] {"question": "is the apple on the plate?"}
=> Objects on plate: ['red apple']. Task looks successful!
```

## Real Mode

In real mode (with SIM_URL + VLA_URL + VLM_URL set), the tools call actual
HTTP services:

- `look` -> Sim `/render` -> VLM `/analyze` -> natural language
- `move` / `grasp` -> VLA `/predict` (generates motor commands) -> Sim `/step` (executes)

The VLA (GR00T N1.6) receives the current camera observation plus a text
instruction and produces low-level joint actions. We execute multiple
steps to complete each high-level tool call.

## Dispatch Map Pattern

The loop is identical to r01. The only new code is:

```python
TOOL_HANDLERS = {
    "look":  lambda **kw: run_look(kw.get("question", "")),
    "move":  lambda **kw: run_move(kw.get("target", ""), kw.get("position")),
    "grasp": lambda **kw: run_grasp(kw["action"]),
}
```

Adding a new tool = adding one entry to this dict + one tool schema.
The agent loop never needs to change.

## Key Insight

Traditional robotics pick-and-place requires:
- Motion planning (RRT, OMPL)
- Inverse kinematics
- Grasp planning (GraspIt, Contact-GraspNet)
- State machine coordination

Agent robotics pick-and-place:
- LLM decides the sequence: look -> move -> grasp -> move -> grasp -> look
- Each tool call is one HTTP request
- The LLM handles error recovery naturally ("the grasp failed, let me try again")

## What's Next

r02 can do single-step tasks. In **r03**, we add task decomposition —
breaking complex goals into subtasks with a todo list, so the agent can
tackle multi-step missions like "set the table" or "clean up the kitchen".
