---
name: side-grasp
description: Grasp an object from the side. Best for tall objects with handles (mugs, bottles, tools).
tags: grasp, manipulation
---

# Side Grasp Skill

Use this skill when picking up tall objects, objects with handles,
or objects where top-grasp clearance is limited.

## When to Use

- Mugs (grasp the handle or body)
- Bottles (grasp the neck or body)
- Tools with handles (spatula, ladle)
- Objects near walls/shelves (no room above)

## When NOT to Use

- Flat objects on open surfaces (use top-grasp, it's simpler)
- Very wide objects (gripper can't span)
- Fragile objects (side force may tip them)

## Procedure

1. **Pre-position**: Move end-effector to the SIDE of the object
   - Offset: target_pos + [0, +0.12, 0] (approach from the side)
   - Match height to the object's midpoint

2. **Open gripper**: Ensure gripper is fully open

3. **Approach**: Move horizontally toward the object
   - Move to target_pos + [0, +0.02, 0]
   - Slow lateral approach for precision

4. **Grasp**: Close gripper around the object
   - For mugs: aim for the body, not the handle (more stable)
   - Verify grasp succeeded

5. **Lift**: Move straight up 10cm before any lateral movement
   - Move to current_pos + [0, 0, +0.10]
   - Prevents dragging or tipping neighbors

## Example Tool Sequence

```
move(position=[target_x, target_y + 0.12, target_z])  # beside
grasp(action="open")                                     # ensure open
move(position=[target_x, target_y + 0.02, target_z])  # approach
grasp(action="close")                                    # grab
move(position=[target_x, target_y + 0.02, target_z + 0.10])  # lift
```

## Failure Recovery

- If approach knocks object: stop, look to reassess position, retry
- If grasp is unstable: place down gently, try different approach angle
- For mugs with handles: if body-grasp fails, try handle-grasp
