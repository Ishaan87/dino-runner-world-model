"""Grabs illustrative frames of the reconstructed env for the blog post:
one clean run frame, one about to jump a cactus, one about to duck a low
bird, and one safely running under a high bird. Saved upscaled (raw sim
frames are 64x64) with a light annotation strip under each panel.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from dino_env import HeadlessDinoEnv

INK = (11, 11, 11)          # #0b0b0b in BGR order for cv2
SURFACE = (251, 252, 252)   # #fcfcfb
SECONDARY = (78, 81, 82)    # #52514e

OUT_DIR = os.path.dirname(__file__)
UPSCALE = 6
PANEL_LABEL_H = 34


def force_obstacle(env, obs_type, y_top_offset, w=20, h=25, x=260):
    import pygame
    env.obstacles = [{
        "rect": pygame.Rect(x, env.ground_y - y_top_offset, w, h),
        "type": obs_type,
        "name": "forced",
    }]


def grab(env, label):
    frame = env._get_obs()[0]  # (64,64) uint8
    big = cv2.resize(frame, (64 * UPSCALE, 64 * UPSCALE), interpolation=cv2.INTER_NEAREST)
    big_bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    panel = np.full((64 * UPSCALE + PANEL_LABEL_H, 64 * UPSCALE, 3), SURFACE, dtype=np.uint8)
    panel[:64 * UPSCALE, :, :] = big_bgr
    cv2.putText(panel, label, (10, 64 * UPSCALE + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, INK, 1, cv2.LINE_AA)
    return panel


def main():
    env = HeadlessDinoEnv()
    env.reset()

    panels = []

    # 1. Clean running frame, no obstacle nearby
    env.obstacles = []
    env.is_jumping = False
    env.is_ducking = False
    panels.append(grab(env, "Run: open ground"))

    # 2. Cactus incoming -> should jump
    force_obstacle(env, "cactus", y_top_offset=25, w=20, h=25, x=200)
    env.is_jumping = False
    env.is_ducking = False
    panels.append(grab(env, "Cactus incoming -> Jump"))

    # 3. Low bird incoming -> should duck
    force_obstacle(env, "bird", y_top_offset=45, w=22, h=16, x=200)
    env.is_jumping = False
    env.is_ducking = False
    panels.append(grab(env, "Low bird incoming -> Duck"))

    # 4. High bird incoming -> safe to run under
    force_obstacle(env, "bird", y_top_offset=70, w=22, h=16, x=200)
    env.is_jumping = False
    env.is_ducking = False
    panels.append(grab(env, "High bird incoming -> Run (safe)"))

    # 5. Mid-jump over a cactus
    force_obstacle(env, "cactus", y_top_offset=25, w=20, h=25, x=90)
    env.is_jumping = True
    env.dino_vy = -6
    env.dino_y = env.ground_y - 30 - 45
    panels.append(grab(env, "Mid-air over cactus"))

    # 6. Ducking under a low bird
    force_obstacle(env, "bird", y_top_offset=45, w=22, h=16, x=90)
    env.is_jumping = False
    env.is_ducking = True
    panels.append(grab(env, "Ducking under low bird"))

    gap = 8
    cols = 3
    rows = 2
    ph, pw = panels[0].shape[:2]
    grid = np.full((rows * ph + (rows - 1) * gap, cols * pw + (cols - 1) * gap, 3), SURFACE, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y0 = r * (ph + gap)
        x0 = c * (pw + gap)
        grid[y0:y0 + ph, x0:x0 + pw, :] = p

    out_path = os.path.join(OUT_DIR, "env_frames.png")
    cv2.imwrite(out_path, grid)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
