"""Make a small test clip — drifting bouncing shapes on a textured background."""
import argparse
import os
import sys

import cv2
import numpy as np


def make_clip(out_dir, n=12, w=128, h=96, seed=1):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Textured backdrop so the DCT actually has something to do.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 120.0 + 50.0 * np.sin(xs / 18.0) + 30.0 * np.cos(ys / 15.0)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    backdrop = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    sprites = []
    for _ in range(5):
        sprites.append(dict(
            p=np.array([rng.integers(15, h - 25),
                        rng.integers(15, w - 25)], dtype=np.float32),
            v=rng.uniform(-3.5, 3.5, 2).astype(np.float32),
            r=int(rng.integers(7, 18)),
            c=tuple(int(x) for x in rng.integers(60, 240, 3)),
        ))

    for i in range(n):
        frame = backdrop.copy()
        for s in sprites:
            s["p"] += s["v"]
            if s["p"][0] < s["r"] or s["p"][0] > h - s["r"]:
                s["v"][0] *= -1
            if s["p"][1] < s["r"] or s["p"][1] > w - s["r"]:
                s["v"][1] *= -1
            cv2.circle(frame,
                       (int(s["p"][1]), int(s["p"][0])),
                       s["r"], s["c"], -1)
        cv2.imwrite(os.path.join(out_dir, f"f_{i:04d}.png"), frame)
    print(f"wrote {n} frames -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="sample_frames")
    p.add_argument("-n", "--num", type=int, default=12)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=96)
    a = p.parse_args()
    make_clip(a.out, a.num, a.width, a.height)
