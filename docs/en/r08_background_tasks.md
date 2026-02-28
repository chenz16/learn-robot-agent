# R08 — Background Tasks

## Mental Model

Before r08, every tool call blocks. `look()` takes 500ms-3s (VLM inference),
`move()` takes 2-10s (VLA + sim steps). The agent waits idle.

In r08, fire-and-forget: start a perception or action in the background,
keep planning, get results delivered automatically.

```
Agent ----[bg_act("move to apple")]----[plan subtask 3]----[bg_look("check")]----
               |                                                |
               v                                                v
           [VLA+sim runs in thread]                     [VLM runs in thread]
               |                                                |
               +--- notification queue ----> [injected before next LLM call]
```

## What Changed

| Component | r07 | r08 |
|-----------|-----|-----|
| Tools | 10 (all blocking) | + **bg_look, bg_act, check_background** |
| Execution | Sequential | **Parallel** (threaded) |
| Loop | standard | + **notification drain** before each LLM call |

## Two New Tools

| Tool | Sync equivalent | Behavior |
|------|----------------|----------|
| `bg_look(question)` | `look(question)` | Returns immediately with job_id. Result delivered next turn. |
| `bg_act(instruction)` | `move`/`grasp` | Fires VLA+sim action in background. Agent keeps thinking. |

The agent also has `check_background(job_id)` to poll status.

## Notification Queue

```python
class BackgroundManager:
    def run(self, job_id, func, args, description):
        # Start daemon thread, return immediately
        thread = Thread(target=self._execute, ...)
        thread.start()
        return f"Job {job_id} started"

    def _execute(self, job_id, func, args):
        result = func(*args)          # runs in background
        self._queue.append(result)     # push to notification queue

    def drain(self):
        # Called before each LLM call
        notifs = list(self._queue)
        self._queue.clear()
        return notifs
```

Before each LLM call, the agent loop drains the queue and injects:
```xml
<background-results>
[bg:a1b2c3d4] completed: act: move to apple
  Moved: [0.30, 0.0, 0.95] -> [0.45, 0.12, 0.82] (0.22m)
</background-results>
```

## Demo

```
r08 >> pick up the apple while monitoring for obstacles

[bg_look] {"question": "any obstacles or humans nearby?"}
Background job abc123 started: look: any obstacles or humans nearby?

[task_create] {"subject": "Pick up apple using top-grasp"}
[task_update] {"task_id": 1, "status": "in_progress"}
[load_skill] top-grasp

[bg notification] 1 job(s) completed    <-- delivered automatically
  [bg:abc123] completed: look: any obstacles or humans nearby?
    => No obstacles detected. Path clear.

[move] {"target": "red apple"}
Moved to apple.

[bg_look] {"question": "still clear?"}   <-- monitor while grasping

[grasp] {"action": "close"}
Grasped 'red apple'.

[bg notification] 1 job(s) completed
  [bg:def456] completed: look: still clear?
    => Clear. Apple now held.
```

## Sync vs Async Decision

| Use sync when | Use async when |
|--------------|----------------|
| Result needed immediately | Can plan while waiting |
| Simple quick check | Long VLM/VLA inference |
| During critical grasp sequence | Monitoring / safety checks |
| Verifying final result | Parallel perception + action |

## REPL Commands

- `bg` — list background jobs
- `tasks` — show task board
- `skills` — list skills
- `tokens` — token estimate
- `/compact` — force compression
- `reset` — clear everything

## What's Next

In **r09**, we add multi-robot teams — multiple agents coordinating
via message-passing inboxes. One robot scouts, another manipulates,
a third monitors safety. Each runs its own agent loop in a thread.
