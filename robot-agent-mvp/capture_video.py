"""Capture full video from a LIBERO pipeline run with detailed HUD overlay."""

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

INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def draw_bar(draw, x, y, w, h, value, max_val, color, label):
    """Draw a small horizontal bar indicator."""
    draw.rectangle([x, y, x + w, y + h], outline=(100, 100, 100))
    fill_w = int(w * min(abs(value) / max_val, 1.0))
    if fill_w > 0:
        draw.rectangle([x, y, x + fill_w, y + h], fill=color)
    draw.text((x + w + 4, y - 1), label, fill=(200, 200, 200))


def annotate_frame(img_array, step, total_steps_est, eef_pos, eef_pos_prev,
                   gripper_qpos, replan_id, action_in_chunk, action,
                   vla_ms, elapsed, is_new_replan, success_checked):
    """Render full HUD overlay on frame."""
    # Upscale to 640x640 for better readability
    img = Image.fromarray(img_array).resize((640, 640), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    W, H = 640, 640

    # === Top banner: Task instruction ===
    draw.rectangle([0, 0, W, 28], fill=(0, 0, 0, 180))
    task_short = INSTRUCTION[:80]
    draw.text((8, 6), f"Task: {task_short}", fill=(255, 255, 255))

    # === Left panel: State ===
    panel_y = 36
    draw.rectangle([0, panel_y, 260, panel_y + 155], fill=(0, 0, 0, 160))

    gripper_state = "OPEN" if gripper_qpos[0] > 0.5 else "CLOSED"
    gripper_color = (0, 255, 0) if gripper_qpos[0] > 0.5 else (255, 100, 100)

    # Compute velocity from position delta
    delta = eef_pos - eef_pos_prev if eef_pos_prev is not None else np.zeros(3)
    vel = np.linalg.norm(delta)

    lines = [
        ("STATE", (0, 200, 255)),
        (f"  EEF Pos:  [{eef_pos[0]:+.3f}, {eef_pos[1]:+.3f}, {eef_pos[2]:+.3f}]", (220, 220, 220)),
        (f"  Delta:    [{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}]", (180, 180, 180)),
        (f"  Velocity: {vel:.4f} m/step", (180, 180, 180)),
        (f"  Gripper:  {gripper_state} ({gripper_qpos[0]:.3f})", gripper_color),
    ]

    y = panel_y + 4
    for text, color in lines:
        draw.text((6, y), text, fill=color)
        y += 16

    # Phase detection
    if vel < 0.001:
        phase = "IDLE"
        phase_color = (150, 150, 150)
    elif action[6] < -0.3 and gripper_qpos[0] < 0.5:
        phase = "GRASPING"
        phase_color = (255, 200, 0)
    elif action[6] > 0.3:
        phase = "RELEASING"
        phase_color = (0, 255, 200)
    elif vel > 0.01:
        phase = "MOVING"
        phase_color = (0, 255, 0)
    else:
        phase = "ADJUSTING"
        phase_color = (200, 200, 0)

    draw.text((6, y), f"  Phase:    {phase}", fill=phase_color)

    # === Right panel: Action ===
    panel_x = W - 270
    draw.rectangle([panel_x, panel_y, W, panel_y + 155], fill=(0, 0, 0, 160))

    action_labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]

    ry = panel_y + 4
    replan_color = (255, 255, 0) if is_new_replan else (180, 180, 180)
    draw.text((panel_x + 6, ry), "VLA ACTION", fill=(255, 165, 0))
    ry += 16
    draw.text((panel_x + 6, ry), f"  Replan #{replan_id}  chunk [{action_in_chunk+1}/5]", fill=replan_color)
    ry += 16
    if is_new_replan and vla_ms > 0:
        draw.text((panel_x + 6, ry), f"  Inference: {vla_ms:.0f}ms", fill=(0, 255, 0))
    else:
        draw.text((panel_x + 6, ry), f"  (cached chunk)", fill=(120, 120, 120))
    ry += 18

    # Action component bars
    for i, (val, label) in enumerate(zip(action, action_labels)):
        bar_color = (255, 100, 100) if i == 6 else (0, 180, 255)
        draw_bar(draw, panel_x + 10, ry, 80, 10, val, 0.5, bar_color, f"{label}: {val:+.3f}")
        ry += 14

    # === Bottom bar: Progress ===
    draw.rectangle([0, H - 32, W, H], fill=(0, 0, 0, 180))

    # Progress bar
    progress = step / 300  # max_steps
    bar_w = W - 200
    draw.rectangle([8, H - 22, 8 + bar_w, H - 10], outline=(80, 80, 80))
    draw.rectangle([8, H - 22, 8 + int(bar_w * progress), H - 10], fill=(0, 200, 100))

    draw.text((8 + bar_w + 8, H - 25),
              f"Step {step} | Replan {replan_id} | {elapsed:.1f}s",
              fill=(200, 200, 200))

    # New replan flash indicator
    if is_new_replan:
        draw.rectangle([W - 90, 36, W, 54], fill=(255, 255, 0))
        draw.text((W - 85, 38), "REPLAN", fill=(0, 0, 0))

    # Success indicator
    if success_checked:
        draw.rectangle([W//2 - 60, H//2 - 15, W//2 + 60, H//2 + 15], fill=(0, 180, 0))
        draw.text((W//2 - 50, H//2 - 10), "SUCCESS!", fill=(255, 255, 255))

    return np.array(img)


def main():
    print("=== Initializing LIBERO environment ===")
    env = LiberoEnv(task_name="libero_spatial:0")
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
    print(f"=== Running: '{INSTRUCTION}' ===")

    for cycle in range(60):
        replan_id += 1
        t_vla_start = time.time()
        actions = loop.run_until_complete(vla.predict(obs, INSTRUCTION))
        t_vla = time.time() - t_vla_start

        print(f"  Replan {replan_id}: VLA {t_vla*1000:.0f}ms, {len(actions)} actions")

        for i, action in enumerate(actions):
            action_clipped = np.clip(action, -0.5, 0.5)
            obs, reward, done, info = env.step(action_clipped)
            total_steps += 1
            elapsed = time.time() - t_start

            eef_pos = np.array(obs.get("robot0_eef_pos", np.zeros(3)))
            gripper_qpos = np.array(obs.get("robot0_gripper_qpos", np.zeros(2)))

            # State log
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

            # Render frame
            img = obs.get("agentview_image")
            if img is not None:
                img = np.asarray(img, dtype=np.uint8)[::-1, ::-1]
                frame = annotate_frame(
                    img, total_steps, 300, eef_pos, eef_pos_prev,
                    gripper_qpos, replan_id, i, action_clipped,
                    t_vla * 1000 if i == 0 else 0, elapsed,
                    is_new_replan=(i == 0),
                    success_checked=False,
                )
                frames.append(frame)

            eef_pos_prev = eef_pos.copy()

            if done:
                break

        if done:
            break

    t_total = time.time() - t_start
    success = env.check_success()

    # Add success frame (hold for 1 second at 15fps)
    if success and frames:
        img = obs.get("agentview_image")
        if img is not None:
            img = np.asarray(img, dtype=np.uint8)[::-1, ::-1]
            eef_pos = np.array(obs.get("robot0_eef_pos", np.zeros(3)))
            gripper_qpos = np.array(obs.get("robot0_gripper_qpos", np.zeros(2)))
            success_frame = annotate_frame(
                img, total_steps, 300, eef_pos, eef_pos_prev,
                gripper_qpos, replan_id, 4, np.zeros(7),
                0, t_total, False, success_checked=True,
            )
            for _ in range(15):  # hold 1 second
                frames.append(success_frame)

    print(f"\n=== Result ===")
    print(f"Steps: {total_steps}, Replans: {replan_id}")
    print(f"Total time: {t_total:.1f}s, Rate: {total_steps/t_total:.1f} steps/s")
    print(f"Task success: {'YES' if success else 'NO'}")

    # Save MP4
    import imageio
    video_path = os.path.join(OUT_DIR, "demo_run.mp4")
    writer = imageio.get_writer(video_path, fps=15, codec="libx264",
                                output_params=["-crf", "20"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Video: {video_path} ({len(frames)} frames, {len(frames)/15:.1f}s)")

    # Save GIF
    gif_path = os.path.join(OUT_DIR, "demo_run.gif")
    gif_frames = frames[::2]  # every 2nd frame
    imageio.mimsave(gif_path, gif_frames, duration=133, loop=0)
    print(f"GIF: {gif_path} ({len(gif_frames)} frames)")

    # Save state log
    log_path = os.path.join(OUT_DIR, "state_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "task": INSTRUCTION,
            "total_steps": total_steps,
            "total_replans": replan_id,
            "total_time_s": round(t_total, 2),
            "avg_vla_ms": round(np.mean([s["vla_inference_ms"] for s in state_log if s["vla_inference_ms"] > 0]), 1),
            "success": success,
            "steps": state_log,
        }, f, indent=2)
    print(f"Log: {log_path}")

    env.close()
    vla.close()


if __name__ == "__main__":
    main()
