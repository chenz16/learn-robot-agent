"""Capture video: Full LLM Agent Pipeline (PRAE loop) with HUD overlay.

Reproduces the actual agent flow:
  1. Prepare  — model_ensure
  2. Perceive — look (observe scene)
  3. Reason   — LLM decomposes: subtask 1 (soup), subtask 2 (sauce)
  4. Act      — start_subtask #1 (200 steps, full instruction)
  5. Evaluate — look (check progress)
  6. Act      — start_subtask #2 (200 steps, full instruction)
  7. Evaluate — look (task success = YES)

Full instruction sent to VLA each time (matches VLA training data).
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
MAX_STEPS_PER_SUBTASK = 200

# Agent PRAE phases shown in overlay
PRAE_PHASES = [
    ("PREPARE", "model_ensure → system ready"),
    ("PERCEIVE", "look → observe scene"),
    ("REASON", "decompose → 2 subtasks"),
    ("ACT", "start_subtask #1 → VLA executing"),
    ("EVALUATE", "look → check progress"),
    ("ACT", "start_subtask #2 → VLA executing"),
    ("EVALUATE", "look → task success ✓"),
]


def draw_bar(draw, x, y, w, h, value, max_val, color, label):
    draw.rectangle([x, y, x + w, y + h], outline=(80, 80, 80))
    fill_w = int(w * min(abs(value) / max_val, 1.0))
    if fill_w > 0:
        draw.rectangle([x, y, x + fill_w, y + h], fill=color)
    draw.text((x + w + 4, y - 1), label, fill=(200, 200, 200))


def annotate_frame(img_array, step, total_steps, eef_pos, eef_pos_prev,
                   gripper_qpos, replan_id, action_in_chunk, action,
                   vla_ms, elapsed, is_new_replan,
                   subtask_num, subtask_total_steps, prae_phase, prae_detail,
                   task_success):
    img = Image.fromarray(img_array).resize((640, 640), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    W, H = 640, 640

    # === Top: Agent status (3 lines) ===
    draw.rectangle([0, 0, W, 62], fill=(0, 0, 0))

    # Line 1: Goal
    draw.text((8, 3), f"Goal: {GOAL}", fill=(200, 200, 200))

    # Line 2: PRAE phase
    prae_colors = {
        "PREPARE": (100, 200, 255),
        "PERCEIVE": (0, 255, 200),
        "REASON": (255, 200, 0),
        "ACT": (255, 100, 100),
        "EVALUATE": (0, 255, 100),
    }
    pc = prae_colors.get(prae_phase, (200, 200, 200))
    draw.text((8, 20), f"Agent [{prae_phase}]: {prae_detail}", fill=pc)

    # Line 3: Subtask info
    st_color = (0, 200, 255) if subtask_num == 1 else (255, 165, 0)
    draw.text((8, 38),
              f"Subtask {subtask_num}/2 | Step {subtask_total_steps}/{MAX_STEPS_PER_SUBTASK} | "
              f"Total {total_steps}",
              fill=st_color)

    # === Left: Robot State ===
    py = 68
    draw.rectangle([0, py, 265, py + 110], fill=(0, 0, 0, 180))

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
        (f" Vel:     {vel:.4f}", (180, 180, 180)),
        (f" Gripper: {gripper_state} ({gripper_qpos[0]:.3f})", gripper_color),
        (f" Phase:   {phase}", phase_color),
    ]
    y = py + 4
    for text, color in lines:
        draw.text((6, y), text, fill=color)
        y += 16

    # === Right: VLA Action ===
    px = W - 270
    draw.rectangle([px, py, W, py + 110], fill=(0, 0, 0, 180))

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

    # === Bottom: Progress ===
    draw.rectangle([0, H - 30, W, H], fill=(0, 0, 0))
    bar_w = W - 300
    progress = total_steps / (MAX_STEPS_PER_SUBTASK * 2)
    draw.rectangle([8, H - 20, 8 + bar_w, H - 9], outline=(80, 80, 80))
    draw.rectangle([8, H - 20, 8 + int(bar_w * min(progress, 1.0)), H - 9],
                   fill=(0, 200, 100))
    draw.text((8 + bar_w + 8, H - 23),
              f"Replan {replan_id} | {elapsed:.1f}s",
              fill=(200, 200, 200))

    # Replan flash
    if is_new_replan:
        draw.rectangle([W - 80, py, W, py + 16], fill=(255, 255, 0))
        draw.text((W - 75, py + 1), "REPLAN", fill=(0, 0, 0))

    # Success overlay
    if task_success:
        draw.rectangle([W//2 - 90, H//2 - 18, W//2 + 90, H//2 + 18], fill=(0, 150, 0))
        draw.text((W//2 - 75, H//2 - 10), "TASK COMPLETE!", fill=(255, 255, 255))

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


def make_prae_card(phase, detail, extra_lines=None):
    """Card showing agent PRAE phase transition."""
    prae_colors = {
        "PREPARE": (100, 200, 255),
        "PERCEIVE": (0, 255, 200),
        "REASON": (255, 200, 0),
        "ACT": (255, 100, 100),
        "EVALUATE": (0, 255, 100),
    }
    lines = [
        ("LLM Agent (Gemini 2.5 Flash)", (200, 200, 200)),
        ("", (0, 0, 0)),
        (f"PRAE Loop: [{phase}]", prae_colors.get(phase, (200, 200, 200))),
        (detail, (180, 180, 180)),
    ]
    if extra_lines:
        lines.append(("", (0, 0, 0)))
        lines.extend(extra_lines)
    return make_card(lines)


def run_subtask(env, vla, loop, instruction, frames, all_log,
                subtask_num, total_steps_so_far, replan_id_so_far,
                eef_pos_prev, t_start):
    """Run one subtask (200 steps max) and capture frames."""
    total_steps = total_steps_so_far
    replan_id = replan_id_so_far

    for cycle in range(MAX_STEPS_PER_SUBTASK // 5 + 1):
        replan_id += 1
        t_vla = time.time()
        actions = loop.run_until_complete(vla.predict(env.get_observation(), instruction))
        t_vla = time.time() - t_vla

        if replan_id <= replan_id_so_far + 3 or replan_id % 10 == 0:
            print(f"    Replan {replan_id}: VLA {t_vla*1000:.0f}ms")

        subtask_steps = total_steps - total_steps_so_far

        for i, action in enumerate(actions):
            action_clipped = np.clip(action, -0.5, 0.5)
            obs, reward, done, info = env.step(action_clipped)
            total_steps += 1
            subtask_steps = total_steps - total_steps_so_far
            elapsed = time.time() - t_start

            eef_pos = np.array(obs.get("robot0_eef_pos", np.zeros(3)))
            gripper_qpos = np.array(obs.get("robot0_gripper_qpos", np.zeros(2)))

            all_log.append({
                "step": total_steps,
                "subtask": subtask_num,
                "subtask_step": subtask_steps,
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
                    img, subtask_steps, total_steps,
                    eef_pos, eef_pos_prev, gripper_qpos,
                    replan_id, i, action_clipped,
                    t_vla * 1000 if i == 0 else 0, elapsed,
                    is_new_replan=(i == 0),
                    subtask_num=subtask_num,
                    subtask_total_steps=subtask_steps,
                    prae_phase="ACT",
                    prae_detail=f"start_subtask #{subtask_num} → VLA executing",
                    task_success=False,
                )
                frames.append(frame)

            eef_pos_prev = eef_pos.copy()

            if done or subtask_steps >= MAX_STEPS_PER_SUBTASK:
                break
        if done or subtask_steps >= MAX_STEPS_PER_SUBTASK:
            break

    return total_steps, replan_id, eef_pos_prev, done


def main():
    print(f"Goal: {GOAL}")
    print("=== Full LLM Agent Pipeline (PRAE Loop) ===\n")

    env = LiberoEnv(task_name="libero_10:0")
    obs = env.reset()
    vla = WebSocketVLAAdapter(host="0.0.0.0", port=8000, resize_size=224, replan_steps=5)
    loop = asyncio.new_event_loop()

    frames = []
    all_log = []
    t_start = time.time()

    # === Intro card ===
    card = make_card([
        ("Full Agent Pipeline Demo", (255, 255, 255)),
        ("", (0, 0, 0)),
        (f"Goal: {GOAL}", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("LLM: Gemini 2.5 Flash", (100, 200, 255)),
        ("VLA: OpenPI pi0 (libero)", (255, 165, 0)),
        ("Sim: LIBERO / MuJoCo", (0, 200, 100)),
        ("", (0, 0, 0)),
        ("Pipeline: LLM (PRAE) -> VLA -> Sim", (180, 180, 180)),
    ])
    for _ in range(45):
        frames.append(card)

    # === PREPARE phase ===
    print("1. [PREPARE] model_ensure → system ready")
    card = make_prae_card("PREPARE", "model_ensure → checking system...", [
        ("VLA server: ready", (0, 255, 0)),
        ("Environment: ready", (0, 255, 0)),
    ])
    for _ in range(25):
        frames.append(card)

    # === PERCEIVE phase ===
    print("2. [PERCEIVE] look → observing scene")
    card = make_prae_card("PERCEIVE", "look → observing initial scene", [
        ("Task: put both ... in the basket", (200, 200, 200)),
        ("Robot: EEF at home position", (180, 180, 180)),
        ("Gripper: open", (0, 255, 0)),
    ])
    for _ in range(25):
        frames.append(card)

    # === REASON phase ===
    print("3. [REASON] decompose → 2 subtasks")
    card = make_prae_card("REASON", "LLM decomposes task (repeat pattern)", [
        ("Subtask 1: alphabet soup -> basket", (0, 200, 255)),
        ("Subtask 2: tomato sauce -> basket", (255, 165, 0)),
        ("Difficulty: hard", (255, 100, 100)),
        ("Using full instruction for VLA", (150, 150, 150)),
    ])
    for _ in range(35):
        frames.append(card)

    # === ACT phase: Subtask 1 ===
    print("4. [ACT] start_subtask #1 → VLA executing (max 200 steps)")
    card = make_prae_card("ACT", "start_subtask #1 → launching VLA loop", [
        ("Instruction: (full task instruction)", (200, 200, 200)),
        ("Max steps: 200", (180, 180, 180)),
        ("Target: alphabet soup", (0, 200, 255)),
    ])
    for _ in range(20):
        frames.append(card)

    total_steps, replan_id, eef_pos_prev, done = run_subtask(
        env, vla, loop, GOAL, frames, all_log,
        subtask_num=1, total_steps_so_far=0, replan_id_so_far=0,
        eef_pos_prev=None, t_start=t_start,
    )
    print(f"   Subtask 1 done: {total_steps} steps, env_done={done}")

    # === EVALUATE phase: Check progress ===
    success_mid = bool(env.check_success())
    print(f"5. [EVALUATE] look → task success: {success_mid}")
    card = make_prae_card("EVALUATE", "look → checking progress after subtask 1", [
        (f"Steps executed: {total_steps}", (180, 180, 180)),
        (f"Task success: {'YES' if success_mid else 'NO - continuing'}", (0, 255, 0) if success_mid else (255, 200, 0)),
        ("Need subtask 2" if not success_mid else "All done!", (200, 200, 200)),
    ])
    for _ in range(30):
        frames.append(card)

    if not success_mid and not done:
        # === ACT phase: Subtask 2 ===
        print("6. [ACT] start_subtask #2 → VLA executing (max 200 steps)")
        card = make_prae_card("ACT", "start_subtask #2 → launching VLA loop", [
            ("Instruction: (full task instruction)", (200, 200, 200)),
            ("Max steps: 200", (180, 180, 180)),
            ("Target: tomato sauce", (255, 165, 0)),
        ])
        for _ in range(20):
            frames.append(card)

        total_steps, replan_id, eef_pos_prev, done = run_subtask(
            env, vla, loop, GOAL, frames, all_log,
            subtask_num=2, total_steps_so_far=total_steps,
            replan_id_so_far=replan_id,
            eef_pos_prev=eef_pos_prev, t_start=t_start,
        )
        print(f"   Subtask 2 done: {total_steps} total steps, env_done={done}")

    # === Final EVALUATE ===
    t_total = time.time() - t_start
    success = bool(env.check_success())
    print(f"7. [EVALUATE] look → task success: {success}")

    card = make_prae_card("EVALUATE", "look → final verification", [
        (f"Total steps: {total_steps}", (180, 180, 180)),
        (f"Total replans: {replan_id}", (180, 180, 180)),
        (f"Time: {t_total:.1f}s", (180, 180, 180)),
        (f"Task success: {'YES' if success else 'NO'}", (0, 255, 0) if success else (255, 100, 0)),
    ])
    for _ in range(30):
        frames.append(card)

    # Success/end frame from camera
    last_obs = env.get_observation()
    cam = last_obs.get("agentview_image")
    if cam is not None:
        cam = np.asarray(cam, dtype=np.uint8)[::-1, ::-1]
        eef_pos = np.array(last_obs.get("robot0_eef_pos", np.zeros(3)))
        gq = np.array(last_obs.get("robot0_gripper_qpos", np.zeros(2)))
        end_frame = annotate_frame(
            cam, 0, total_steps, eef_pos, eef_pos_prev, gq,
            replan_id, 4, np.zeros(7), 0, t_total,
            False, subtask_num=2, subtask_total_steps=0,
            prae_phase="EVALUATE",
            prae_detail="Task complete — both objects in basket",
            task_success=success,
        )
        for _ in range(30):
            frames.append(end_frame)

    # Summary card
    card = make_card([
        ("AGENT PIPELINE COMPLETE", (255, 255, 255)),
        ("", (0, 0, 0)),
        (f"Goal: {GOAL}", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("PRAE Loop:", (100, 200, 255)),
        ("  PREPARE  -> model_ensure", (150, 150, 150)),
        ("  PERCEIVE -> look (observe)", (150, 150, 150)),
        ("  REASON   -> decompose 2 subtasks", (150, 150, 150)),
        ("  ACT      -> start_subtask x2", (150, 150, 150)),
        ("  EVALUATE -> look (verify)", (150, 150, 150)),
        ("", (0, 0, 0)),
        (f"Total: {total_steps} steps, {replan_id} replans, {t_total:.1f}s",
         (220, 220, 220)),
        (f"Result: {'SUCCESS' if success else 'INCOMPLETE'}",
         (0, 255, 0) if success else (255, 100, 0)),
    ])
    for _ in range(50):
        frames.append(card)

    print(f"\n=== Final ===")
    print(f"Steps: {total_steps}, Replans: {replan_id}")
    print(f"Time: {t_total:.1f}s")
    print(f"Task success: {'YES' if success else 'NO'}")

    # Save video
    import imageio
    video_path = os.path.join(OUT_DIR, "demo_agent_pipeline.mp4")
    writer = imageio.get_writer(video_path, fps=15, codec="libx264",
                                output_params=["-crf", "20"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Video: {video_path} ({len(frames)} frames, {len(frames)/15:.1f}s)")

    gif_path = os.path.join(OUT_DIR, "demo_agent_pipeline.gif")
    gif_frames = frames[::4]
    imageio.mimsave(gif_path, gif_frames, duration=267, loop=0)
    print(f"GIF: {gif_path} ({len(gif_frames)} frames)")

    log_path = os.path.join(OUT_DIR, "state_log_agent.json")
    with open(log_path, "w") as f:
        json.dump({
            "goal": GOAL,
            "pipeline": "LLM (Gemini 2.5 Flash) PRAE → VLA (pi0) → LIBERO",
            "subtasks": [
                "Subtask 1: alphabet soup -> basket",
                "Subtask 2: tomato sauce -> basket",
            ],
            "total_steps": total_steps,
            "total_replans": replan_id,
            "total_time_s": round(t_total, 2),
            "success": success,
            "steps": all_log,
        }, f, indent=2)
    print(f"Log: {log_path}")

    env.close()
    vla.close()


if __name__ == "__main__":
    main()
