# R11 — Autonomous Patrol

## Mental Model

Before r11, the robot waits for commands. It only acts when you tell it to.
In r11, the robot **finds work itself** — it patrols waypoints, detects
anomalies, claims tasks from the board, and keeps working until everything
is done.

```
+--------+
| START  |
+---+----+
    |
    v
+--------+  waypoint    +---------+
| PATROL | -----------> | LOOK    |
+---+----+              +----+----+
    |                        |
    | all done               | anomaly?
    v                        v
+--------+              +---------+
| IDLE   | poll 5s      | REPORT  | → create task
+---+----+              +---------+
    |
    +---> check inbox → message? → resume PATROL
    |
    +---> scan .tasks/ → unclaimed? → claim → WORK
    |
    +---> patrol active? → next waypoint → PATROL
    |
    +---> nothing (60s) → restart cycle
```

## What Changed

| Component | r10 | r11 |
|-----------|-----|-----|
| Behavior | Reactive (waits for commands) | **Autonomous (finds work)** |
| Tools | 19 | + **add_route, start_patrol, stop_patrol, patrol_status, next_waypoint, list_routes** |
| Idle phase | Return to REPL | **Poll tasks + patrol waypoints** |
| Anomaly detection | Manual | **Automatic keyword scanning** |

## Patrol Routes

Predefined waypoint lists, each with a position and perception query:

```python
DEFAULT_ROUTES = {
    "kitchen_sweep": [
        {"name": "counter_left",   "position": [0.30, -0.20, 0.90],
         "check": "any spills or fallen objects on the left counter?"},
        {"name": "counter_center", "position": [0.45, 0.00, 0.90],
         "check": "any obstacles or misplaced items?"},
        {"name": "counter_right",  "position": [0.60, 0.20, 0.90],
         "check": "any hazards on the right counter?"},
        {"name": "sink_area",      "position": [0.70, -0.10, 0.85],
         "check": "is the sink area clear? any water spills?"},
    ],
}
```

## Autonomous Agent Loop

The key change is in the agent loop — after the LLM stops calling tools,
instead of returning to the REPL, the loop checks for more work:

```python
def agent_loop(messages):
    while True:
        # ... normal LLM + tool loop ...
        if response.stop_reason != "tool_use":
            # IDLE PHASE: auto-inject next work

            # 1. Next patrol waypoint?
            wp = PATROL.next_waypoint()
            if wp:
                messages.append({"role": "user", "content":
                    f"<patrol>Move to '{wp['name']}' at {wp['position']}. "
                    f"Then check: {wp['check']}</patrol>"})
                continue  # re-enter loop

            # 2. Unclaimed tasks?
            unclaimed = scan_unclaimed_tasks()
            if unclaimed:
                claim_task(unclaimed[0]["id"], "patrol-robot")
                messages.append({"role": "user", "content":
                    f"<claimed>Task #{task['id']}: {task['subject']}</claimed>"})
                continue

            return  # truly idle
```

## Anomaly Detection

At each waypoint, the agent uses `look()`. If the result contains
anomaly keywords, a task is automatically created:

```python
ANOMALY_KEYWORDS = [
    "obstacle", "spill", "fallen", "broken",
    "human", "person", "fire", "smoke", "damage"
]
```

## Demo

```
r11 >> start a kitchen sweep patrol

[start_patrol] route="kitchen_sweep", mode="once"
Patrol 'kitchen_sweep' started (once mode, 4 waypoints).

<patrol>Move to 'counter_left' at [0.30, -0.20, 0.90].
Then check: any spills or fallen objects?</patrol>

[move] position=[0.30, -0.20, 0.90]
Moved to counter_left.

[look] "any spills or fallen objects on the left counter?"
Scene: ... No spills detected. Area clear.

<patrol>Move to 'counter_center' at [0.45, 0.00, 0.90].
Then check: any obstacles or misplaced items?</patrol>

[move] position=[0.45, 0.00, 0.90]
[look] "any obstacles or misplaced items?"
Scene: ... Fork displaced from plate.

[task_create] "Return displaced fork to plate"

<patrol>Move to 'counter_right'...</patrol>
... (continues through all waypoints) ...

Patrol complete. Checking task board...
[auto-claimed] Task #1: Return displaced fork to plate
[move] target="fork"
[grasp] close
...
```

## Patrol Modes

| Mode | Behavior |
|------|----------|
| `once` | Visit all waypoints, then idle |
| `loop` | Repeat route continuously |

## REPL Commands

- `patrol` — show patrol status
- `routes` — list available routes
- `safety` — safety status
- `tasks` — show task board
- `bg` — list background jobs
- `skills` — list skills
- `tokens` — token estimate
- `/compact` — force compression
- `reset` — clear everything

## What's Next

In **r12**, we add parallel simulation environments — fork the environment,
try multiple strategies in isolated sandboxes, compare results, and promote
the best approach.
