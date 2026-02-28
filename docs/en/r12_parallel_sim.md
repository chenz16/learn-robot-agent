# R12 — Parallel Simulation Environments

## Mental Model

Before r12, the robot commits to one strategy. If it fails, undo is
expensive. In r12, **fork the environment**, try multiple approaches in
isolated sandboxes, compare results, and promote the best one.

```
Main Environment          Sim "top-grasp"        Sim "side-grasp"
+----------------+       +----------------+      +----------------+
| apple on table |  fork | apple on table |      | apple on table |
| ee=[0.30,...]  | ----> | ee=[0.30,...]  |      | ee=[0.30,...]  |
+----------------+       +--------+-------+      +--------+-------+
                                  |                       |
                          try top-grasp            try side-grasp
                                  |                       |
                          apple held ✓             apple dropped ✗
                                  |                       |
                          +-------v-------+               |
                          | PROMOTE       |        DISCARD
                          | copy to main  |
                          +---------------+
```

## What Changed

| Component | r11 | r12 |
|-----------|-----|-----|
| Environment | Single (shared) | **Main + N isolated sims** |
| Strategy | Commit-and-hope | **Try-compare-promote** |
| Tools | 25 | + **create_sim, sim_action, compare_sims, promote_sim, discard_sim, list_sims** |
| Risk | Actions are permanent | **Sandboxed exploration** |

## SimEnvironment

Each sim is an isolated deep copy of MockRobotEnv:

```python
class SimEnvironment:
    def __init__(self, name, source_env):
        self.name = name
        self.env = MockRobotEnv()
        # Deep copy state from source
        self.env.objects = {k: dict(v) for k, v in source_env.objects.items()}
        self.env.ee_pos = list(source_env.ee_pos)
        self.env.gripper = source_env.gripper
        self.env.holding = source_env.holding
        self.action_log = []
        self.status = "active"  # active | promoted | discarded
```

## SimManager

```python
SIMS = SimManager(SIMS_DIR, mock_env)

# Fork: snapshot current state
SIMS.create("approach-A")
SIMS.create("approach-B")

# Execute in isolation (main env untouched)
SIMS.run_action("approach-A", "move", {"target": "red apple"})
SIMS.run_action("approach-A", "grasp", {"action": "close"})
SIMS.run_action("approach-B", "move", {"target": "red apple"})

# Compare side-by-side
SIMS.compare("approach-A", "approach-B")

# Promote winner → copies state back to main
SIMS.promote("approach-A")

# Discard loser
SIMS.discard("approach-B")
```

## Demo

```
r12 >> pick up the apple, test top-grasp vs side approach

[create_sim] name="top-grasp"
Sim 'top-grasp' created (forked from main env).

[create_sim] name="side-grasp"
Sim 'side-grasp' created (forked from main env).

[sim_action] sim="top-grasp", tool="move", args={"target": "red apple"}
[sim:top-grasp] Moved: [0.30, 0.0, 0.95] -> [0.45, 0.12, 0.82] (0.22m)

[sim_action] sim="top-grasp", tool="grasp", args={"action": "close"}
[sim:top-grasp] Grasped 'red apple'.

[sim_action] sim="side-grasp", tool="move", args={"target": "red apple"}
[sim:side-grasp] Moved: [0.30, 0.0, 0.95] -> [0.45, 0.12, 0.82] (0.22m)

[sim_action] sim="side-grasp", tool="grasp", args={"action": "close"}
[sim:side-grasp] Grasped 'red apple'.

[compare_sims] sim1="top-grasp", sim2="side-grasp"
=== Comparison: top-grasp vs side-grasp ===
top-grasp:
  ee=[0.45, 0.12, 0.82], holding=red apple, actions=2
side-grasp:
  ee=[0.45, 0.12, 0.82], holding=red apple, actions=2
Differences: (none — both succeeded in mock)

[promote_sim] name="top-grasp"
Sim 'top-grasp' promoted to main environment. 2 actions applied.

[discard_sim] name="side-grasp"
Sim 'side-grasp' discarded.
```

## When to Use Sim Isolation

| Direct execution | Sim isolation |
|-----------------|---------------|
| Simple, low-risk actions | Risky or uncertain strategies |
| One obvious approach | Multiple valid approaches |
| Fast recovery if failed | Expensive to undo |
| Well-known objects | Unknown object properties |

## Workflow: Sim + Tasks

```
1. task_create("Pick up fragile vase")
2. create_sim("gentle-approach", task_id=1)
3. create_sim("standard-approach", task_id=1)
4. Test both in parallel sims
5. compare_sims → pick the gentler one
6. promote_sim("gentle-approach")
7. task_update(1, status="completed")
```

## REPL Commands

- `sims` — list sim environments
- `safety` — safety status
- `tasks` — show task board
- `bg` — list background jobs
- `skills` — list skills
- `tokens` — token estimate
- `/compact` — force compression
- `reset` — clear everything

## Summary: The Complete Stack (r01–r12)

| Layer | Lesson | What It Adds |
|-------|--------|-------------|
| Core | r01 | Agent loop + look |
| Action | r02 | Move + grasp tools |
| Planning | r03 | Task decomposition |
| Perception | r04 | Subagent observer |
| Skills | r05 | SKILL.md recipes |
| Memory | r06 | Context compression |
| Persistence | r07 | Disk-based task board |
| Concurrency | r08 | Background threads |
| Teams | r09 | Multi-robot coordination |
| Safety | r10 | E-stop + safety zones |
| Autonomy | r11 | Patrol + auto-claim |
| Simulation | r12 | Parallel sandboxes |
