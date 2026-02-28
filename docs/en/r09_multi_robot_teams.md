# R09 — Multi-Robot Teams

## Mental Model

Before r09, one agent does everything — perceive, plan, act. Slow and
sequential. In r09, a **lead** robot spawns specialist teammates that
each run their own agent loop in a separate thread.

```
Lead Robot (planner)
  │
  ├── spawn_robot("scout", "perception", "scan the area")
  │       └── Thread: scout agent_loop(tools=[look, move, send_message])
  │
  ├── spawn_robot("arm1", "manipulator", "pick up the apple")
  │       └── Thread: arm1 agent_loop(tools=[look, move, grasp, send_message])
  │
  └── spawn_robot("watcher", "monitor", "watch for humans")
          └── Thread: watcher agent_loop(tools=[look, send_message])

Communication: JSONL inboxes
  .team/inbox/scout.jsonl
  .team/inbox/arm1.jsonl
  .team/inbox/lead.jsonl
```

## What Changed

| Component | r08 | r09 |
|-----------|-----|-----|
| Agents | 1 (lead only) | **1 lead + N teammates** |
| Tools | 13 | + **spawn_robot, list_robots, send_message, read_inbox, broadcast** |
| Execution | Single-threaded + bg | **Multi-agent threaded** |
| Communication | None | **JSONL inbox per robot** |
| Environment | Single-owner | **Thread-safe shared** |

## Subagent (r04) vs Teammate (r09)

| | Subagent | Teammate |
|---|---|---|
| Lifetime | spawn → execute → return → destroyed | spawn → work → idle → work → ... |
| Context | Fresh (empty) | Persistent (own messages) |
| Communication | Return value only | Bidirectional messaging |
| Tools | Subset (look only) | Full robot tools |
| Thread | Blocks parent | Independent thread |

## MessageBus

```python
class MessageBus:
    def send(self, sender, to, content, msg_type="message"):
        # Append to .team/inbox/{to}.jsonl
        msg = {"type": msg_type, "from": sender, "content": content}
        inbox_path.open("a").write(json.dumps(msg))

    def read_inbox(self, name):
        # Read and drain .team/inbox/{name}.jsonl
        messages = [json.loads(line) for line in inbox_path]
        inbox_path.write_text("")  # drain
        return messages

    def broadcast(self, sender, content, teammates):
        # Send to all except sender
```

## Teammate Agent Loop

Each teammate runs its own agent loop in a thread:

```python
def _teammate_loop(self, name, role, prompt):
    messages = [{"role": "user", "content": prompt}]
    tools = [look, move, grasp, send_message, read_inbox]

    for _ in range(50):
        # Check inbox for new messages
        inbox = BUS.read_inbox(name)
        for msg in inbox:
            messages.append({"role": "user", "content": json.dumps(msg)})

        response = client.messages.create(model=MODEL, ...)
        if response.stop_reason != "tool_use":
            break
        # Execute tools...
```

## Demo

```
r09 >> pick up the apple while monitoring for obstacles

[spawn_robot] {"name": "watcher", "role": "monitor", "prompt": "Watch for obstacles..."}
Spawned 'watcher' (role: monitor)

[spawn_robot] {"name": "arm1", "role": "manipulator", "prompt": "Pick up red apple..."}
Spawned 'arm1' (role: manipulator)

  [watcher] look: Scene: ... No obstacles detected.
  [watcher] send_message: Sent message to lead
  [arm1] move: Moved: [0.30, 0.0, 0.95] -> [0.45, 0.12, 0.82]
  [arm1] grasp: Grasped 'red apple'.
  [arm1] send_message: Sent message to lead

[read_inbox]
  {"from": "watcher", "content": "Area clear, no obstacles."}
  {"from": "arm1", "content": "Apple picked up successfully."}

Done. Apple held by arm1.
```

## Thread Safety

The MockRobotEnv is shared across threads. All methods use `threading.Lock()`:

```python
class MockRobotEnv:
    def __init__(self):
        self._lock = threading.Lock()
    def move(self, target="", position=None):
        with self._lock:
            # ... mutate state safely ...
```

## When to Use Teams vs Single Agent

| Single agent | Multi-robot team |
|-------------|------------------|
| Simple pick-and-place | Scout + manipulate simultaneously |
| One object, one action | Multiple objects, parallel actions |
| No safety monitoring needed | Continuous safety monitoring |
| Quick tasks (<30s) | Complex multi-step missions |

## REPL Commands

- `team` — list team members
- `inbox` — read lead's inbox
- `tasks` — show task board
- `bg` — list background jobs
- `skills` — list skills
- `tokens` — token estimate
- `/compact` — force compression
- `reset` — clear everything

## What's Next

In **r10**, we add safety protocols — e-stop, safety zones, pre-action
checks, and audit logging. Every robot action passes through a safety
pipeline before execution.
