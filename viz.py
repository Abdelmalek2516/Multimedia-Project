"""Stage-by-stage visualisation for the report."""
import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import mpeg_codec as mc


def _load_dir(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png"))
                   + glob.glob(os.path.join(folder, "*.jpg")))
    return [cv2.imread(p) for p in paths]


def make_figure(frames_dir, bin_path, out_png):
    frames = _load_dir(frames_dir)
    blob = open(bin_path, "rb").read()
    recon, params, records = mc.decode(blob, output_shape=frames[0].shape[:2])
    q_y, _ = mc.make_qtables(params.quality)

    fig, axes = plt.subplots(5, 6, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("MPEG-4-like pipeline — stage visualisation", fontsize=13)

    # ---- row 0 : originals
    pick = np.linspace(0, len(frames) - 1, 6, dtype=int)
    for col, idx in enumerate(pick):
        axes[0, col].imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"orig {idx}", fontsize=8)
        axes[0, col].axis("off")
    fig.text(0.005, 0.93, "1) originals", weight="bold", rotation=90)

    # ---- row 1 : colour space
    ycbcr = mc.bgr_to_ycbcr(frames[0])
    chan_titles = ["Y", "Cb", "Cr"]
    chan_cmaps = ["gray", "coolwarm", "coolwarm"]
    # Use 2 axes per channel, hide the others.
    for i in range(3):
        ax = axes[1, i * 2]
        ax.imshow(ycbcr[..., i], cmap=chan_cmaps[i])
        ax.set_title(chan_titles[i], fontsize=9)
        ax.axis("off")
        axes[1, i * 2 + 1].axis("off")
    fig.text(0.005, 0.73, "2) Y / Cb / Cr", weight="bold", rotation=90)

    # ---- row 2 : DCT + quantisation on one 8x8 block
    y0 = ycbcr[..., 0]
    by, bx = y0.shape[0] // 2, y0.shape[1] // 2
    blk = y0[by:by + 8, bx:bx + 8].astype(np.float32) - 128.0
    d = cv2.dct(blk)
    q = np.round(d / q_y).astype(np.int16)
    rec_blk = cv2.idct(q.astype(np.float32) * q_y) + 128.0
    panels = [
        ("raw pixels", blk + 128.0, "gray"),
        ("DCT", np.log1p(np.abs(d)), "viridis"),
        ("quantised", q, "viridis"),
        ("reconstructed", rec_blk, "gray"),
        (f"Q-table (Q={params.quality})", q_y, "magma"),
    ]
    for i, (t, v, cm) in enumerate(panels):
        axes[2, i].imshow(v, cmap=cm)
        axes[2, i].set_title(t, fontsize=9)
        axes[2, i].axis("off")
    axes[2, 5].axis("off")
    fig.text(0.005, 0.53, "3) DCT & quant", weight="bold", rotation=90)

    # ---- row 3 : motion vectors + residual on a P-frame
    p_idx = next((i for i, r in enumerate(records) if r["type"] == "P"), None)
    if p_idx is not None:
        mv = records[p_idx]["mv"]
        big = axes[3, 0].get_gridspec()
        # We'll just place into the first half of the row.
        for c in range(3):
            axes[3, c].axis("off")
        ax_left = fig.add_subplot(big[3, 0:3])
        ax_left.imshow(cv2.cvtColor(frames[p_idx], cv2.COLOR_BGR2RGB))
        rows, cols, _ = mv.shape
        ys = (np.arange(rows) + 0.5) * params.macroblock
        xs = (np.arange(cols) + 0.5) * params.macroblock
        XS, YS = np.meshgrid(xs, ys)
        ax_left.quiver(XS, YS, mv[..., 1], mv[..., 0],
                       color="yellow", angles="xy",
                       scale_units="xy", scale=1, width=0.003)
        ax_left.set_title(f"motion vectors on P-frame {p_idx}", fontsize=9)
        ax_left.axis("off")

        for c in range(3, 6):
            axes[3, c].axis("off")
        ax_right = fig.add_subplot(big[3, 3:6])
        prev_gray = cv2.cvtColor(recon[p_idx - 1], cv2.COLOR_BGR2GRAY).astype(np.int16)
        cur_gray = cv2.cvtColor(recon[p_idx], cv2.COLOR_BGR2GRAY).astype(np.int16)
        im = ax_right.imshow(cur_gray - prev_gray, cmap="seismic", vmin=-40, vmax=40)
        ax_right.set_title(f"residual (P-frame {p_idx} vs prev)", fontsize=9)
        ax_right.axis("off")
        fig.colorbar(im, ax=ax_right, fraction=0.04)
    fig.text(0.005, 0.33, "4 & 5) motion + residual", weight="bold", rotation=90)

    # ---- row 4 : reconstructions
    show = np.linspace(0, len(recon) - 1, 6, dtype=int)
    for col, idx in enumerate(show):
        axes[4, col].imshow(cv2.cvtColor(recon[idx], cv2.COLOR_BGR2RGB))
        axes[4, col].set_title(f"rec {idx} [{records[idx]['type']}]", fontsize=8)
        axes[4, col].axis("off")
    fig.text(0.005, 0.13, "5) reconstructions", weight="bold", rotation=90)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("frames")
    p.add_argument("bin_path")
    p.add_argument("-o", "--out", default="pipeline.png")
    a = p.parse_args()
    make_figure(a.frames, a.bin_path, a.out)
