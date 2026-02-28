---
name: pour
description: Pour contents from one container to another. Requires careful tilt control.
tags: manipulation, liquid
---

# Pour Skill

Pour liquid or granular material from a source container into a target.

## When to Use

- Pouring water from a mug to another container
- Pouring ingredients while cooking
- Any transfer of contents between containers

## Prerequisites

- Source container must be grasped and lifted first (use top-grasp or side-grasp)
- Target container must be identified and reachable
- Ensure target has enough capacity

## Procedure

1. **Grasp source**: Pick up the source container (use appropriate grasp skill)

2. **Position above target**: Move source container directly above the target
   - Offset: target_pos + [0, 0, +0.15]
   - Center alignment is critical

3. **Tilt to pour**: Gradually tilt the source
   - In simulation: move the source toward the edge of target
   - Slow, controlled motion prevents spilling

4. **Hold position**: Maintain pour angle for 2-3 seconds
   - Use look() to check if target is filling

5. **Return upright**: Slowly tilt source back to vertical
   - Quick movements cause splashing

6. **Place source down**: Return source to its original position or a safe spot

## Example Tool Sequence

```
# Assume source is already grasped
move(target="target container")             # position above
move(position=[target_x + 0.05, target_y, target_z + 0.10])  # tilt position
look(question="is liquid flowing into target?")
move(position=[target_x, target_y, target_z + 0.15])  # return upright
```

## Safety

- Never pour hot liquids near electronics or the robot base
- If liquid spills: stop immediately, report to user
- Check gripper stability before tilting (object must be secure)
