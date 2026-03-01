# Tool Usage Guide

## When to use start_subtask vs move/grasp

- **start_subtask**: For VLA-driven actions that require visual feedback and continuous control. Use this for pick, place, open drawer, close drawer, etc. The VLA control loop handles the motion planning.
- **move/grasp**: For direct state manipulation. In mock mode, these immediately update positions and gripper state. Useful for quick testing without running VLA loops.

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
