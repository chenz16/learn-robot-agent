# Tool Usage Guide

## When to use start_subtask vs move/grasp

- **start_subtask**: For VLA-driven actions that require visual feedback and continuous control. Use this when a real VLA model is serving (http mode). The VLA control loop handles the motion planning.
- **move/grasp**: For direct positioning and gripper control. In mock simulation mode, **prefer move + grasp** — they set positions exactly. A typical pick-and-place sequence:
  1. `move` to the object position
  2. `grasp` close (pick up)
  3. `move` to the target position
  4. `grasp` open (release)
- Use `look` before and after to perceive and verify.

## Difficulty Classification

- **easy**: Single object, single verb, no dependencies. Example: "pick up the red cup".
- **hard**: Multiple objects, compound instructions, dependencies between steps. Example: "put both items in the basket", "open the drawer and place the bowl inside".

## Retry Pattern

When a subtask fails:
1. Call `look` to re-perceive the scene.
2. Analyze what went wrong (object moved, missed grasp, etc.).
3. Adjust the instruction or target if needed.
4. Retry with `start_subtask` + `wait_subtask`.
5. After 3 failures, report to the user.
