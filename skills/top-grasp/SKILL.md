---
name: top-grasp
description: Grasp an object from above. Best for flat/low objects on surfaces (apples, plates, boxes).
tags: grasp, manipulation
---

# Top Grasp Skill

Use this skill when picking up objects that are flat, low-profile, or
sitting on a surface with clearance above.

## When to Use

- Fruit on a counter (apple, orange)
- Plates, bowls (grasp the rim)
- Boxes, books
- Any object with clear space above

## When NOT to Use

- Tall objects with handles (use side-grasp instead)
- Objects near walls or under shelves (no clearance above)
- Very heavy objects (need two-hand grasp)

## Procedure

1. **Pre-position**: Move end-effector 15cm ABOVE the target object
   - Offset: target_pos + [0, 0, +0.15]
   - This avoids collision during approach

2. **Open gripper**: Ensure gripper is open before descending

3. **Descend**: Move straight down to the object
   - Move to target_pos + [0, 0, +0.02] (slightly above surface)
   - Slow approach prevents knocking the object

4. **Grasp**: Close gripper firmly
   - Verify grasp by checking tool result

5. **Lift**: Move straight up 15cm
   - Move to current_pos + [0, 0, +0.15]
   - Lifting before lateral movement prevents dragging

## Example Tool Sequence

```
move(position=[target_x, target_y, target_z + 0.15])  # above
grasp(action="open")                                    # ensure open
move(position=[target_x, target_y, target_z + 0.02])  # descend
grasp(action="close")                                   # grab
move(position=[target_x, target_y, target_z + 0.15])  # lift
```

## Failure Recovery

- If grasp fails (nothing in gripper): re-open, adjust position slightly, retry
- If object slips during lift: lower back down, re-grasp with tighter approach
- Max 2 retries before reporting failure
