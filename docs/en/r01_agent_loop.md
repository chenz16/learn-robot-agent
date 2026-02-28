# R01 — The Robot Agent Loop

## Mental Model

A robot agent is just a while loop:

```
while the LLM calls a tool:
    execute the tool
    feed the result back
```

In learn-claude-code, the first tool is `bash` — the agent interacts with a computer.
In learn-robot-agent, the first tool is `look` — the agent perceives the physical world.

**Before you can grasp, move, or navigate — you must see.**

## Architecture

```
User: "what's on the table?"
         |
         v
+------------------+
|   Agent Loop     |
|   (LLM + tools)  |
+--------+---------+
         |
         v
+------------------+       +------------------+
|   look tool      | ----> | Sim Server :8030 |
|                  |       | (camera image)   |
|                  |       +--------+---------+
|                  |                |
|                  |                v
|                  |       +------------------+
|                  | <---- | VLM Server :8010 |
+--------+---------+       | (scene analysis) |
         |                 +------------------+
         v
"I see a red apple on the counter,
 ~27cm from a white plate."
```

## Three Modes

| Mode | Config | What happens |
|------|--------|-------------|
| **MOCK** | Default (no env vars) | Returns hardcoded scene description |
| **SIM** | `SIM_URL=http://...` | Gets real observation from simulator |
| **FULL** | `SIM_URL` + `VLM_URL` | Sim camera image -> VLM -> natural language |

Mock mode lets you run r01 immediately with zero setup.

## The Code

The entire agent is ~170 lines. The core loop is 20 lines — identical to learn-claude-code s01:

```python
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return  # LLM decided to stop — done

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_look(block.input.get("question", ""))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

**The loop never changes.** Every future lesson adds tools and mechanisms
on top of this exact structure.

## Key Insight

In traditional robotics, perception is a pipeline:
`Camera -> Preprocessing -> Object Detection -> State Estimation -> Planning`

In agent robotics, perception is a tool call:
`LLM -> look() -> "I see a red apple 27cm from the plate"`

The LLM decides *when* to look, *what* to ask about, and *how to interpret*
the result — all in natural language.

## Try It

```bash
cd agents
cp .env.example .env   # add your ANTHROPIC_API_KEY
python r01_agent_loop.py
```

```
r01 >> what do you see?
[look] (general observation)
Kitchen counter — simulated environment.
Objects:
  - red apple at [0.45, 0.12, 0.82], on counter
  - white plate at [0.45, -0.15, 0.80], on counter
  ...

"I can see a kitchen counter with three objects:
 a red apple, a white plate, and a blue mug..."

r01 >> can the robot reach the apple?
[look] can the robot reach the apple?
=> The apple is within arm's reach (~20cm from end-effector).

"Yes, the apple is within reach. The end-effector is at
 [0.30, 0.0, 0.95] and the apple is at [0.45, 0.12, 0.82],
 about 20cm away."
```

## What's Next

r01 can see but not act. In **r02**, we add `move` and `grasp` tools —
turning observation into action.
