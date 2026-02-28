# R10 — Safety Protocols

## Mental Model

In r09, robots can move and grasp freely. Nothing stops a robot from
crashing into a human or entering a dangerous zone. In r10, every action
passes through a **safety pipeline** before execution.

```
Action request          Safety pipeline
+--------------+       +------------------+
| move(target) | ----> | 1. E-stop check  |  → BLOCKED if active
+--------------+       +--------+---------+
                                |
                       +--------v---------+
                       | 2. Zone check    |  → BLOCKED if forbidden
                       | (forbidden/warn) |  → WARNING if caution
                       +--------+---------+
                                |
                       +--------v---------+
                       | 3. Velocity limit|  → BLOCKED if too fast
                       | (max 0.5m/step)  |
                       +--------+---------+
                                |
                       +--------v---------+
                       | 4. Execute + log |
                       +------------------+
```

## What Changed

| Component | r09 | r10 |
|-----------|-----|-----|
| Safety | None | **4-layer safety pipeline** |
| Tools | 18 | + **e_stop, reset_estop, add_safety_zone, remove_safety_zone, safety_status, safety_log** |
| Move/Grasp | Direct execution | **Safety-wrapped** |
| Logging | None | **Audit trail (.safety/audit.jsonl)** |
| Human detection | Manual | **Automatic zone creation** |

## Four Safety Layers

### Layer 1: E-Stop
```python
SAFETY.emergency_stop()   # Freeze everything
# All move/grasp calls return: "E-STOP active. All actions blocked."
SAFETY.reset_estop()      # Resume operations
```

### Layer 2: Zone Enforcement
```python
SAFETY.add_zone("human_area", center=[0.5, 0.3, 0.8], radius=0.3, level="forbidden")
# "forbidden" → blocks movement into zone
# "warning"   → allows with caution message

SAFETY.check_move(from_pos, to_pos)
# Returns (allowed: bool, reason: str)
```

### Layer 3: Velocity Limits
```python
MAX_VELOCITY = 0.5  # meters per move
distance = dist(from_pos, to_pos)
if distance > MAX_VELOCITY:
    return (False, f"Move too fast: {distance:.2f}m > {MAX_VELOCITY}m limit")
```

### Layer 4: Audit Logging
Every check is logged to `.safety/audit.jsonl`:
```json
{"action": "move", "target": "red apple", "allowed": true, "reason": "No violations", "ts": 1709123456}
{"action": "move", "target": "blue mug", "allowed": false, "reason": "Enters forbidden zone 'mug_zone'", "ts": 1709123457}
```

## Human Detection

When `look()` returns results containing "human" or "person", the safety
system automatically adds a warning zone:

```python
def run_look_with_safety(question=""):
    result = run_look(question)
    if "human" in result.lower() or "person" in result.lower():
        SAFETY.add_zone("human_detected", mock_env.ee_pos, 0.3, "warning")
        result += "\n[SAFETY] Human detected — warning zone added."
    return result
```

## Demo

```
r10 >> add a safety zone around the mug

[add_safety_zone] name="mug_zone", center=[0.50, 0.30, 0.82], radius=0.1
Safety zone 'mug_zone' added (forbidden).

r10 >> pick up the red apple

[move] target="red apple"
[SAFETY] Move allowed. No zone violations.
Moved to red apple.

[grasp] action="close"
Grasped 'red apple'.

r10 >> now move to the blue mug

[move] target="blue mug"
[SAFETY] BLOCKED: enters forbidden zone 'mug_zone' (0.05m < 0.10m radius)

r10 >> emergency stop

[e_stop]
EMERGENCY STOP activated. All actions blocked.

r10 >> move to apple

[move] target="red apple"
[SAFETY] E-STOP active. Call reset_estop to resume.
```

## Safety Status

```
r10 >> safety

Zones:
  mug_zone: center=[0.50, 0.30, 0.82], radius=0.10, level=forbidden
E-Stop: ACTIVE
Recent audit (last 5):
  [BLOCKED] move -> blue mug: enters forbidden zone
  [BLOCKED] move -> red apple: E-STOP active
```

## REPL Commands

- `safety` — show safety status
- `tasks` — show task board
- `bg` — list background jobs
- `skills` — list skills
- `tokens` — token estimate
- `/compact` — force compression
- `reset` — clear everything

## What's Next

In **r11**, we add autonomous patrol — the robot autonomously visits
waypoints, scans for anomalies, and claims tasks from the board when idle.
Safety checks from r10 apply to all patrol movements.
