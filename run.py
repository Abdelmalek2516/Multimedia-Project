"""
run.py — single entry point with subcommands:

    python run.py encode    sample_frames -o out.bin --gop 8 --q 50
    python run.py decode    out.bin -o decoded [--ref sample_frames]
    python run.py viz       sample_frames out.bin -o pipeline.png
    python run.py sweep     sample_frames -o experiments.png
"""
import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import mpeg_codec as mc


def _load(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png"))
                   + glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        sys.exit(f"no frames found in {folder}")
    return paths, [cv2.imread(p) for p in paths]


def cmd_encode(a):
    paths, frames = _load(a.frames)
    params = mc.Params(
        gop=a.gop, quality=a.q, block=8,
        macroblock=16, search=a.search,
        subsample=not a.no_chroma_subsample,
    )
    raw_bytes = sum(f.nbytes for f in frames)
    blob = mc.encode(frames, params)
    with open(a.out, "wb") as fh:
        fh.write(blob)
    print(f"frames        : {len(frames)}")
    print(f"raw bytes     : {raw_bytes:,}")
    print(f"compressed    : {len(blob):,} bytes")
    print(f"ratio         : {raw_bytes / len(blob):.2f}x")
    print(f"output        : {a.out}")


def cmd_decode(a):
    blob = open(a.bin_path, "rb").read()
    ref_shape = None
    if a.ref:
        ref_paths, ref_frames = _load(a.ref)
        ref_shape = ref_frames[0].shape[:2]
    recon, params, records = mc.decode(blob, output_shape=ref_shape)
    os.makedirs(a.out, exist_ok=True)
    for i, f in enumerate(recon):
        cv2.imwrite(os.path.join(a.out, f"rec_{i:04d}.png"), f)
    n_i, n_p = mc.frame_breakdown(records)
    print(f"decoded {len(recon)} frames -> {a.out}")
    print(f"  I-frames: {n_i}, P-frames: {n_p}")
    if a.ref:
        psnrs = [mc.psnr(r, c) for r, c in zip(ref_frames, recon)]
        print(f"  mean PSNR: {np.mean(psnrs):.2f} dB")
        for i, v in enumerate(psnrs):
            print(f"    frame {i:02d}: {v:.2f} dB")


def cmd_viz(a):
    # Delegate to viz.py
    import viz
    viz.make_figure(a.frames, a.bin_path, a.out)


def cmd_sweep(a):
    _, frames = _load(a.frames)
    raw = sum(f.nbytes for f in frames)

    qs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q_ratios = []
    print("sweep quality...")
    for q in qs:
        blob = mc.encode(frames, mc.Params(
            gop=a.gop, quality=q, block=8, macroblock=16, search=8, subsample=True))
        q_ratios.append(raw / len(blob))
        print(f"  Q={q:>3}  ratio={q_ratios[-1]:.2f}x  ({len(blob):,} bytes)")

    gops = [1, 2, 4, 8, 16]
    g_ratios = []
    print("sweep GOP...")
    for g in gops:
        blob = mc.encode(frames, mc.Params(
            gop=g, quality=a.q, block=8, macroblock=16, search=8, subsample=True))
        g_ratios.append(raw / len(blob))
        print(f"  GOP={g:>3}  ratio={g_ratios[-1]:.2f}x  ({len(blob):,} bytes)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(qs, q_ratios, "o-", color="#1f77b4")
    ax[0].set_xlabel("quality"); ax[0].set_ylabel("compression ratio (x)")
    ax[0].set_title(f"ratio vs quality (GOP={a.gop})"); ax[0].grid(alpha=0.3)
    ax[1].plot(gops, g_ratios, "s-", color="#d62728")
    ax[1].set_xlabel("GOP size"); ax[1].set_ylabel("compression ratio (x)")
    ax[1].set_title(f"ratio vs GOP (Q={a.q})"); ax[1].grid(alpha=0.3)
    fig.suptitle("Experimental sweeps")
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"saved {a.out}")


def main():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("encode")
    p.add_argument("frames")
    p.add_argument("-o", "--out", default="video.bin")
    p.add_argument("--gop", type=int, default=8)
    p.add_argument("--q", type=int, default=50)
    p.add_argument("--search", type=int, default=8)
    p.add_argument("--no-chroma-subsample", action="store_true")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("decode")
    p.add_argument("bin_path")
    p.add_argument("-o", "--out", default="decoded")
    p.add_argument("--ref", default=None)
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("viz")
    p.add_argument("frames")
    p.add_argument("bin_path")
    p.add_argument("-o", "--out", default="pipeline.png")
    p.set_defaults(func=cmd_viz)

    p = sub.add_parser("sweep")
    p.add_argument("frames")
    p.add_argument("-o", "--out", default="experiments.png")
    p.add_argument("--gop", type=int, default=8)
    p.add_argument("--q", type=int, default=50)
    p.set_defaults(func=cmd_sweep)

    args = root.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
