"""Capture video: 2-object sequential task (libero_10:0) - single VLA run."""

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

TASK_NAME = "libero_10:0"
INSTRUCTION = "put both the alphabet soup and the tomato sauce in the basket"
MAX_STEPS = 600


def draw_bar(draw, x, y, w, h, value, max_val, color, label):
    draw.rectangle([x, y, x + w, y + h], outline=(80, 80, 80))
    fill_w = int(w * min(abs(value) / max_val, 1.0))
    if fill_w > 0:
        draw.rectangle([x, y, x + fill_w, y + h], fill=color)
    draw.text((x + w + 4, y - 1), label, fill=(200, 200, 200))


def annotate_frame(img_array, step, eef_pos, eef_pos_prev, gripper_qpos,
                   replan_id, action_in_chunk, action, vla_ms, elapsed,
                   is_new_replan, success_checked):
    img = Image.fromarray(img_array).resize((640, 640), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    W, H = 640, 640

    # Top banner
    draw.rectangle([0, 0, W, 28], fill=(0, 0, 0))
    draw.text((8, 6), f"Task: {INSTRUCTION}", fill=(255, 255, 255))

    # Left panel: State
    py = 34
    draw.rectangle([0, py, 265, py + 140], fill=(0, 0, 0, 180))

    gripper_state = "OPEN" if gripper_qpos[0] > 0.5 else "CLOSED"
    gripper_color = (0, 255, 0) if gripper_qpos[0] > 0.5 else (255, 100, 100)

    delta = eef_pos - eef_pos_prev if eef_pos_prev is not None else np.zeros(3)
    vel = np.linalg.norm(delta)

    if vel < 0.001:
        phase, phase_color = "IDLE", (150, 150, 150)
    elif action[6] < -0.3 and gripper_qpos[0] < 0.5:
        phase, phase_color = "GRASPING", (255, 200, 0)
    elif action[6] > 0.3:
        phase, phase_color = "RELEASING", (0, 255, 200)
    elif vel > 0.01:
        phase, phase_color = "MOVING", (0, 255, 0)
    else:
        phase, phase_color = "ADJUSTING", (200, 200, 0)

    lines = [
        ("ROBOT STATE", (0, 200, 255)),
        (f" EEF:     [{eef_pos[0]:+.3f}, {eef_pos[1]:+.3f}, {eef_pos[2]:+.3f}]", (220, 220, 220)),
        (f" Delta:   [{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}]", (180, 180, 180)),
        (f" Vel:     {vel:.4f} m/step", (180, 180, 180)),
        (f" Gripper: {gripper_state} ({gripper_qpos[0]:.3f})", gripper_color),
        (f" Phase:   {phase}", phase_color),
    ]
    y = py + 4
    for text, color in lines:
        draw.text((6, y), text, fill=color)
        y += 16

    # Right panel: Action
    px = W - 270
    draw.rectangle([px, py, W, py + 140], fill=(0, 0, 0, 180))

    action_labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    ry = py + 4
    replan_color = (255, 255, 0) if is_new_replan else (180, 180, 180)
    draw.text((px + 6, ry), "VLA ACTION", fill=(255, 165, 0))
    ry += 16
    draw.text((px + 6, ry), f" Replan #{replan_id}  chunk [{action_in_chunk+1}/5]", fill=replan_color)
    ry += 16
    if is_new_replan and vla_ms > 0:
        draw.text((px + 6, ry), f" Inference: {vla_ms:.0f}ms", fill=(0, 255, 0))
    else:
        draw.text((px + 6, ry), f" (cached chunk)", fill=(120, 120, 120))
    ry += 16

    for i, (val, label) in enumerate(zip(action, action_labels)):
        bar_color = (255, 100, 100) if i == 6 else (0, 180, 255)
        draw_bar(draw, px + 10, ry, 80, 9, val, 0.5, bar_color, f"{label}: {val:+.3f}")
        ry += 13

    # Bottom bar
    draw.rectangle([0, H - 30, W, H], fill=(0, 0, 0))
    progress = step / MAX_STEPS
    bar_w = W - 250
    draw.rectangle([8, H - 20, 8 + bar_w, H - 9], outline=(80, 80, 80))
    draw.rectangle([8, H - 20, 8 + int(bar_w * progress), H - 9], fill=(0, 200, 100))
    draw.text((8 + bar_w + 8, H - 23),
              f"Step {step}/{MAX_STEPS} | Replan {replan_id} | {elapsed:.1f}s",
              fill=(200, 200, 200))

    if is_new_replan:
        draw.rectangle([W - 80, py, W, py + 16], fill=(255, 255, 0))
        draw.text((W - 75, py + 1), "REPLAN", fill=(0, 0, 0))

    if success_checked:
        draw.rectangle([W//2 - 80, H//2 - 20, W//2 + 80, H//2 + 20], fill=(0, 150, 0))
        draw.text((W//2 - 70, H//2 - 12), "BOTH DONE!", fill=(255, 255, 255))

    return np.array(img)


def main():
    print(f"=== Task: {INSTRUCTION} ===")
    print(f"=== Max steps: {MAX_STEPS} ===")

    print("\n=== Initializing LIBERO environment ===")
    env = LiberoEnv(task_name=TASK_NAME)
    obs = env.reset()

    print("=== Connecting to VLA server ===")
    vla = WebSocketVLAAdapter(host="0.0.0.0", port=8000, resize_size=224, replan_steps=5)

    frames = []
    state_log = []
    total_steps = 0
    replan_id = 0
    eef_pos_prev = None
    loop = asyncio.new_event_loop()

    t_start = time.time()

    for cycle in range(MAX_STEPS // 5 + 1):
        replan_id += 1
        t_vla_start = time.time()
        actions = loop.run_until_complete(vla.predict(obs, INSTRUCTION))
        t_vla = time.time() - t_vla_start

        if replan_id <= 5 or replan_id % 10 == 0:
            print(f"  Replan {replan_id}: VLA {t_vla*1000:.0f}ms")

        for i, action in enumerate(actions):
            action_clipped = np.clip(action, -0.5, 0.5)
            obs, reward, done, info = env.step(action_clipped)
            total_steps += 1
            elapsed = time.time() - t_start

            eef_pos = np.array(obs.get("robot0_eef_pos", np.zeros(3)))
            gripper_qpos = np.array(obs.get("robot0_gripper_qpos", np.zeros(2)))

            state_log.append({
                "step": total_steps,
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
                    is_new_replan=(i == 0), success_checked=False,
                )
                frames.append(frame)

            eef_pos_prev = eef_pos.copy()

            if done:
                break
            if total_steps >= MAX_STEPS:
                break

        if done or total_steps >= MAX_STEPS:
            break

    t_total = time.time() - t_start
    success = bool(env.check_success())

    # Success/end frame
    last_obs = env.get_observation()
    cam = last_obs.get("agentview_image")
    if cam is not None:
        cam = np.asarray(cam, dtype=np.uint8)[::-1, ::-1]
        end_frame = annotate_frame(
            cam, total_steps, eef_pos, eef_pos_prev, gripper_qpos,
            replan_id, 4, np.zeros(7), 0, t_total,
            False, success_checked=success,
        )
        for _ in range(30):
            frames.append(end_frame)

    print(f"\n=== Result ===")
    print(f"Steps: {total_steps}, Replans: {replan_id}")
    print(f"Time: {t_total:.1f}s, Rate: {total_steps/t_total:.1f} steps/s")
    print(f"Task success: {'YES' if success else 'NO'}")

    import imageio
    video_path = os.path.join(OUT_DIR, "demo_multi_task.mp4")
    writer = imageio.get_writer(video_path, fps=15, codec="libx264",
                                output_params=["-crf", "20"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Video: {video_path} ({len(frames)} frames, {len(frames)/15:.1f}s)")

    gif_path = os.path.join(OUT_DIR, "demo_multi_task.gif")
    gif_frames = frames[::4]
    imageio.mimsave(gif_path, gif_frames, duration=267, loop=0)
    print(f"GIF: {gif_path} ({len(gif_frames)} frames)")

    log_path = os.path.join(OUT_DIR, "state_log_multi.json")
    with open(log_path, "w") as f:
        json.dump({
            "task": INSTRUCTION,
            "total_steps": total_steps,
            "total_replans": replan_id,
            "total_time_s": round(t_total, 2),
            "success": success,
            "steps": state_log,
        }, f, indent=2)
    print(f"Log: {log_path}")

    env.close()
    vla.close()


if __name__ == "__main__":
    main()
