"""Demo: Agent decomposes goal, monitors VLA execution in same scene.

Goal: "put both the alphabet soup and the tomato sauce in the basket"
Agent monitors gripper state to detect subtask transitions:
  grip close → grip open = object placed = subtask boundary

Full instruction sent to VLA (matches training data).
Agent overlay shows which subtask phase the robot is in.
"""

import asyncio
import os
import time
import json
import numpy as np
from PIL import Image, ImageDraw

from robot_agent.env.libero import LiberoEnv
from robot_agent.vla.websocket import WebSocketVLAAdapter

OUT_DIR = "docs/images"
os.makedirs(OUT_DIR, exist_ok=True)

GOAL = "put both the alphabet soup and the tomato sauce in the basket"
SUBTASK_LABELS = [
    "Subtask 1: pick up alphabet soup -> basket",
    "Subtask 2: pick up tomato sauce -> basket",
]
MAX_STEPS = 400


def draw_bar(draw, x, y, w, h, value, max_val, color, label):
    draw.rectangle([x, y, x + w, y + h], outline=(80, 80, 80))
    fill_w = int(w * min(abs(value) / max_val, 1.0))
    if fill_w > 0:
        draw.rectangle([x, y, x + fill_w, y + h], fill=color)
    draw.text((x + w + 4, y - 1), label, fill=(200, 200, 200))


def annotate_frame(img_array, step, eef_pos, eef_pos_prev, gripper_qpos,
                   replan_id, action_in_chunk, action, vla_ms, elapsed,
                   is_new_replan, subtask_idx, subtask_label, phase_name,
                   objects_placed, success_checked):
    img = Image.fromarray(img_array).resize((640, 640), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    W, H = 640, 640

    # Top: goal + agent decomposition
    draw.rectangle([0, 0, W, 50], fill=(0, 0, 0))
    draw.text((8, 4), f"Goal: {GOAL}", fill=(200, 200, 200))
    st_color = (0, 200, 255) if subtask_idx == 0 else (255, 165, 0)
    draw.text((8, 22), f"Agent: {subtask_label}", fill=st_color)
    # Objects placed indicator
    draw.text((8, 38), f"Objects placed: {objects_placed}/2", fill=(180, 180, 180))

    # Left: State
    py = 56
    draw.rectangle([0, py, 265, py + 125], fill=(0, 0, 0, 180))
    gripper_state = "OPEN" if gripper_qpos[0] > 0.5 else "CLOSED"
    gripper_color = (0, 255, 0) if gripper_qpos[0] > 0.5 else (255, 100, 100)
    delta = eef_pos - eef_pos_prev if eef_pos_prev is not None else np.zeros(3)
    vel = np.linalg.norm(delta)

    lines = [
        ("ROBOT STATE", (0, 200, 255)),
        (f" EEF:     [{eef_pos[0]:+.3f}, {eef_pos[1]:+.3f}, {eef_pos[2]:+.3f}]", (220, 220, 220)),
        (f" Vel:     {vel:.4f} m/step", (180, 180, 180)),
        (f" Gripper: {gripper_state} ({gripper_qpos[0]:.3f})", gripper_color),
        (f" Phase:   {phase_name}", (255, 200, 0) if "GRASP" in phase_name or "PLACE" in phase_name else (0, 255, 0) if "REACH" in phase_name else (150, 150, 150)),
    ]
    y = py + 4
    for text, color in lines:
        draw.text((6, y), text, fill=color)
        y += 16

    # Right: Action
    px = W - 270
    draw.rectangle([px, py, W, py + 125], fill=(0, 0, 0, 180))
    action_labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    ry = py + 4
    draw.text((px + 6, ry), "VLA OUTPUT", fill=(255, 165, 0))
    ry += 16
    rc = (255, 255, 0) if is_new_replan else (180, 180, 180)
    draw.text((px + 6, ry), f" Replan #{replan_id}  [{action_in_chunk+1}/5]", fill=rc)
    ry += 16
    if is_new_replan and vla_ms > 0:
        draw.text((px + 6, ry), f" Inference: {vla_ms:.0f}ms", fill=(0, 255, 0))
    else:
        draw.text((px + 6, ry), f" (cached)", fill=(120, 120, 120))
    ry += 14
    for i, (val, label) in enumerate(zip(action, action_labels)):
        bc = (255, 100, 100) if i == 6 else (0, 180, 255)
        draw_bar(draw, px + 10, ry, 80, 9, val, 0.5, bc, f"{label}: {val:+.3f}")
        ry += 13

    # Bottom
    draw.rectangle([0, H - 30, W, H], fill=(0, 0, 0))
    bar_w = W - 300
    progress = step / MAX_STEPS
    draw.rectangle([8, H - 20, 8 + bar_w, H - 9], outline=(80, 80, 80))
    draw.rectangle([8, H - 20, 8 + int(bar_w * min(progress, 1.0)), H - 9], fill=(0, 200, 100))
    draw.text((8 + bar_w + 8, H - 23),
              f"Step {step}/{MAX_STEPS} | Replan {replan_id} | Placed {objects_placed}/2 | {elapsed:.1f}s",
              fill=(200, 200, 200))

    if is_new_replan:
        draw.rectangle([W - 80, py, W, py + 16], fill=(255, 255, 0))
        draw.text((W - 75, py + 1), "REPLAN", fill=(0, 0, 0))

    if success_checked:
        draw.rectangle([W//2 - 80, H//2 - 18, W//2 + 80, H//2 + 18], fill=(0, 150, 0))
        draw.text((W//2 - 65, H//2 - 10), "2/2 COMPLETE!", fill=(255, 255, 255))

    return np.array(img)


def make_card(lines_colors, bg=(15, 15, 30)):
    img = Image.new("RGB", (640, 640), bg)
    draw = ImageDraw.Draw(img)
    y = 640 // 2 - len(lines_colors) * 15
    for text, color in lines_colors:
        x = max(640 // 2 - len(text) * 4, 20)
        draw.text((x, y), text, fill=color)
        y += 25
    return np.array(img)


def detect_phase(gripper_qpos, gripper_prev, action, vel, objects_placed):
    """Detect manipulation phase from gripper state transitions."""
    grip_now = gripper_qpos[0]
    grip_prev = gripper_prev if gripper_prev is not None else grip_now
    closing = grip_now < grip_prev - 0.01
    opening = grip_now > grip_prev + 0.01
    is_closed = grip_now < 0.5
    is_open = grip_now > 0.5

    if opening and is_open:
        return "PLACING"
    if closing or (is_closed and vel > 0.005):
        if vel < 0.005:
            return "GRASPING"
        return "TRANSPORTING"
    if is_open and vel > 0.005:
        return "REACHING"
    if vel < 0.001:
        return "IDLE"
    return "MOVING"


def main():
    print(f"Goal: {GOAL}")

    env = LiberoEnv(task_name="libero_10:0")
    obs = env.reset()
    vla = WebSocketVLAAdapter(host="0.0.0.0", port=8000, resize_size=224, replan_steps=5)

    frames = []
    all_log = []
    loop = asyncio.new_event_loop()
    t_start = time.time()

    # Intro
    card = make_card([
        ("Agent Task Decomposition Demo", (255, 255, 255)),
        ("", (0, 0, 0)),
        (f"Goal: {GOAL}", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("Agent decomposes:", (150, 150, 150)),
        (f"1. {SUBTASK_LABELS[0]}", (0, 200, 255)),
        (f"2. {SUBTASK_LABELS[1]}", (255, 165, 0)),
        ("", (0, 0, 0)),
        ("Same scene, continuous execution", (120, 120, 120)),
    ])
    for _ in range(40):
        frames.append(card)

    total_steps = 0
    replan_id = 0
    eef_pos_prev = None
    gripper_prev = None
    objects_placed = 0
    grip_was_closed = False
    subtask_switch_step = None

    for cycle in range(MAX_STEPS // 5 + 1):
        replan_id += 1
        t_vla = time.time()
        actions = loop.run_until_complete(vla.predict(obs, GOAL))
        t_vla = time.time() - t_vla

        if replan_id <= 3 or replan_id % 10 == 0:
            print(f"  Replan {replan_id}: VLA {t_vla*1000:.0f}ms, placed={objects_placed}")

        for i, action in enumerate(actions):
            action_clipped = np.clip(action, -0.5, 0.5)
            obs, reward, done, info = env.step(action_clipped)
            total_steps += 1
            elapsed = time.time() - t_start

            eef_pos = np.array(obs.get("robot0_eef_pos", np.zeros(3)))
            gripper_qpos = np.array(obs.get("robot0_gripper_qpos", np.zeros(2)))
            delta = eef_pos - eef_pos_prev if eef_pos_prev is not None else np.zeros(3)
            vel = np.linalg.norm(delta)

            # Detect subtask transition: grip closed then opened = object placed
            grip_val = gripper_qpos[0]
            if grip_val < 0.3:
                grip_was_closed = True
            if grip_was_closed and grip_val > 0.7:
                objects_placed += 1
                grip_was_closed = False
                if objects_placed == 1 and subtask_switch_step is None:
                    subtask_switch_step = total_steps
                    print(f"  >>> Object 1 placed at step {total_steps}!")

            subtask_idx = 0 if objects_placed < 1 else 1
            phase = detect_phase(gripper_qpos, gripper_prev, action_clipped, vel, objects_placed)

            all_log.append({
                "step": total_steps,
                "subtask_idx": subtask_idx,
                "phase": phase,
                "objects_placed": objects_placed,
                "replan": replan_id,
                "action_in_chunk": i,
                "vla_inference_ms": round(t_vla * 1000, 1) if i == 0 else 0,
                "eef_pos": [round(float(x), 4) for x in eef_pos],
                "gripper_qpos": [round(float(x), 4) for x in gripper_qpos],
                "action": [round(float(x), 4) for x in action_clipped],
                "elapsed_s": round(elapsed, 3),
            })

            img = obs.get("agentview_image")
            if img is not None:
                img = np.asarray(img, dtype=np.uint8)[::-1, ::-1]
                frame = annotate_frame(
                    img, total_steps, eef_pos, eef_pos_prev, gripper_qpos,
                    replan_id, i, action_clipped,
                    t_vla * 1000 if i == 0 else 0, elapsed,
                    is_new_replan=(i == 0),
                    subtask_idx=subtask_idx,
                    subtask_label=SUBTASK_LABELS[min(subtask_idx, 1)],
                    phase_name=phase,
                    objects_placed=min(objects_placed, 2),
                    success_checked=False,
                )
                frames.append(frame)

            eef_pos_prev = eef_pos.copy()
            gripper_prev = grip_val

            if done or total_steps >= MAX_STEPS:
                break
        if done or total_steps >= MAX_STEPS:
            break

    t_total = time.time() - t_start
    success = bool(env.check_success())

    # End frame
    last_obs = env.get_observation()
    cam = last_obs.get("agentview_image")
    if cam is not None:
        cam = np.asarray(cam, dtype=np.uint8)[::-1, ::-1]
        eef_pos = np.array(last_obs.get("robot0_eef_pos", np.zeros(3)))
        gq = np.array(last_obs.get("robot0_gripper_qpos", np.zeros(2)))
        end_frame = annotate_frame(
            cam, total_steps, eef_pos, eef_pos_prev, gq,
            replan_id, 4, np.zeros(7), 0, t_total,
            False, 1, SUBTASK_LABELS[1], "DONE",
            min(objects_placed, 2), success_checked=success,
        )
        for _ in range(25):
            frames.append(end_frame)

    # Summary
    card = make_card([
        ("DEMO COMPLETE", (255, 255, 255)),
        ("", (0, 0, 0)),
        (f"Goal: {GOAL}", (200, 200, 200)),
        ("", (0, 0, 0)),
        (f"Subtask 1 (soup): done at step {subtask_switch_step or '?'}", (0, 200, 255)),
        (f"Subtask 2 (sauce): done at step {total_steps}", (255, 165, 0)),
        (f"Total: {total_steps} steps, {t_total:.1f}s", (220, 220, 220)),
        (f"Result: {'SUCCESS - both objects in basket!' if success else 'INCOMPLETE'}", (0, 255, 0) if success else (255, 100, 0)),
    ])
    for _ in range(45):
        frames.append(card)

    print(f"\n=== Final ===")
    print(f"Steps: {total_steps}, Replans: {replan_id}")
    print(f"Subtask switch at step: {subtask_switch_step}")
    print(f"Objects placed: {objects_placed}")
    print(f"Time: {t_total:.1f}s")
    print(f"Task success: {'YES' if success else 'NO'}")

    import imageio
    video_path = os.path.join(OUT_DIR, "demo_2tasks.mp4")
    writer = imageio.get_writer(video_path, fps=15, codec="libx264",
                                output_params=["-crf", "20"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Video: {video_path} ({len(frames)} frames, {len(frames)/15:.1f}s)")

    gif_path = os.path.join(OUT_DIR, "demo_2tasks.gif")
    gif_frames = frames[::4]
    imageio.mimsave(gif_path, gif_frames, duration=267, loop=0)
    print(f"GIF: {gif_path} ({len(gif_frames)} frames)")

    log_path = os.path.join(OUT_DIR, "state_log_2tasks.json")
    with open(log_path, "w") as f:
        json.dump({
            "goal": GOAL,
            "subtasks": SUBTASK_LABELS,
            "subtask_switch_step": subtask_switch_step,
            "total_steps": total_steps,
            "total_time_s": round(t_total, 2),
            "success": success,
            "steps": all_log,
        }, f, indent=2)
    print(f"Log: {log_path}")

    env.close()
    vla.close()


if __name__ == "__main__":
    main()
