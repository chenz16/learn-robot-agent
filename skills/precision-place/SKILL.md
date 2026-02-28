---
name: precision-place
description: Place an object precisely at a target location. Use for stacking, alignment, or careful placement.
tags: manipulation, placement
---

# Precision Place Skill

Place a held object at an exact position with controlled descent.

## When to Use

- Placing items on a plate (table setting)
- Stacking objects (cups on saucers, items in boxes)
- Alignment-critical placement (utensils beside plate)
- Placing fragile items gently

## Prerequisites

- Object must already be grasped and lifted
- Target location must be known (from look() or perceive())

## Procedure

1. **Position above target**: Move held object directly above placement point
   - Offset: target_pos + [0, 0, +0.12]
   - Ensure no obstacles in the descent path

2. **Verify alignment**: Use look() to confirm position
   - Check: "am I directly above the target?"
   - Adjust if needed before descending

3. **Descend slowly**: Lower the object to the target surface
   - Move to target_pos + [0, 0, +0.02]
   - Slow descent prevents bouncing or damage

4. **Release**: Open gripper
   - Object should rest on the surface

5. **Retract**: Move gripper away without disturbing the placed object
   - Move UP first: current_pos + [0, 0, +0.10]
   - Then move laterally to clear the area

6. **Verify**: Use look() to confirm successful placement

## Example Tool Sequence

```
# Object already in gripper
move(position=[target_x, target_y, target_z + 0.12])   # above target
look(question="am I above the target position?")         # verify
move(position=[target_x, target_y, target_z + 0.02])   # descend
grasp(action="open")                                     # release
move(position=[target_x, target_y, target_z + 0.12])   # retract up
look(question="is the object placed correctly?")         # verify
```

## Failure Recovery

- If object falls off target: pick up again, realign, retry
- If stacking and top object slides: adjust base stability first
- If alignment is off: pick up, use look() to recalibrate, retry
