# R07 — Persistent Task Board

## Mental Model

r03's TodoManager lived in memory — context compression wiped it.
r07's TaskManager lives on disk — it survives compression, restarts, everything.

```
.tasks/
  task_1.json  {"id":1, "subject":"perceive scene",    "status":"completed"}
  task_2.json  {"id":2, "subject":"pick up apple",     "status":"in_progress", "blockedBy":[]}
  task_3.json  {"id":3, "subject":"place on plate",    "status":"pending",     "blockedBy":[2]}
  task_4.json  {"id":4, "subject":"verify placement",  "status":"pending",     "blockedBy":[3]}
```

After context compression, the agent calls `task_list` and knows
exactly where it left off — no information lost.

## What Changed

| Component | r06 | r07 |
|-----------|-----|-----|
| Planning | TodoManager (in-memory) | **TaskManager (on-disk JSON)** |
| Tools | todo | **task_create, task_update, task_list, task_get** |
| Dependencies | none | **blockedBy / blocks graph** |
| Persistence | lost on compress | **survives everything** |

## r03 TodoManager vs r07 TaskManager

| | TodoManager (r03) | TaskManager (r07) |
|---|---|---|
| Storage | In-memory list | `.tasks/task_*.json` files |
| Update | Replace-all (full list) | CRUD (create/update individual) |
| Dependencies | None | blockedBy + blocks (bidirectional) |
| Survives compress | No | **Yes** |
| Survives restart | No | **Yes** |
| Concurrency-safe | No | File-level isolation |

## Dependency Graph

Tasks can depend on each other:

```python
task_create("perceive scene")                    # task 1
task_create("pick up apple")                     # task 2
task_update(2, addBlockedBy=[1])                 # task 2 waits for task 1

# When task 1 completes:
task_update(1, status="completed")
# -> automatically removes 1 from task 2's blockedBy
# -> task 2 is now unblocked
```

This maps naturally to robot task ordering:
- "pick apple" blocked by "perceive scene" (must see before acting)
- "place on plate" blocked by "pick apple" (must hold before placing)
- "verify" blocked by "place on plate" (must place before checking)

## Demo

```
r07 >> put all items on the plate

[perceive] identify all objects

[task_create] {"subject": "Pick up fork, place on plate"}
[task_create] {"subject": "Pick up napkin, place on plate"}
[task_create] {"subject": "Pick up mug, place on plate"}
[task_create] {"subject": "Pick up apple, place on plate"}
[task_create] {"subject": "Verify all items on plate"}
[task_update] {"task_id": 5, "addBlockedBy": [1, 2, 3, 4]}

[task_update] {"task_id": 1, "status": "in_progress"}
[load_skill] top-grasp
[move] ... [grasp] ... [move] ... [grasp] ...
[task_update] {"task_id": 1, "status": "completed"}

[auto_compact] context compressed...

[task_list]
[x] #1: Pick up fork, place on plate
[ ] #2: Pick up napkin, place on plate    <-- agent knows to do this next
[ ] #3: Pick up mug, place on plate
[ ] #4: Pick up apple, place on plate
[ ] #5: Verify all items on plate (blocked by: [2, 3, 4])

(continues seamlessly after compression)
```

## Compression + Persistent Tasks = Infinite Missions

The combination of r06 (compression) and r07 (persistent tasks) enables:

1. Agent starts a 20-step mission
2. After 8 steps, context gets compressed
3. Agent calls `task_list` — sees exactly which 12 steps remain
4. Continues without missing a beat
5. Can even restart the Python process and resume

## REPL Commands

- `tasks` — show task board
- `skills` — list skills
- `tokens` — token estimate
- `/compact` — force compression
- `reset` — clear tasks, env, history

## What's Next

In **r08**, we add background tasks — async sensor polling and VLA
inference that run in separate threads while the agent keeps thinking.
The robot doesn't have to block-wait for every perception or action call.
