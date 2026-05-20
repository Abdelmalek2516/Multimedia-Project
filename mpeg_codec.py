"""
mpeg_codec.py
=============
A compact MPEG-4-like video codec.  Procedural / functional layout — every
stage is a top-level function and the bitstream is a struct-packed,
bz2-compressed blob (no pickle, no dataclasses).

Pipeline
--------
encode(frames, params) -> bytes:
    YCbCr ->  4:2:0 split  ->  for each frame: pick I or P
        I-frame: dct -> quantise (luma + chroma tables)
        P-frame: three-step block matching against the previous
                 reconstructed Y plane, then dct on the residual

decode(blob) -> [frames]:
    formal inverse of every stage
"""

import bz2
import struct
from collections import namedtuple

import cv2
import numpy as np


# ============================================================
# 1. parameters
# ============================================================

Params = namedtuple(
    "Params",
    "gop quality block macroblock search subsample",
)

DEFAULT_PARAMS = Params(
    gop=8,
    quality=50,
    block=8,
    macroblock=16,
    search=8,
    subsample=True,
)

# Standard JPEG quant tables (these we tune by `quality`).
_BASE_Q_Y = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)

_BASE_Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def make_qtables(quality):
    """Build (Q_luma, Q_chroma) for a given quality in 1..100."""
    q = max(1, min(100, int(quality)))
    # libjpeg-style scaling factor.
    s = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
    def _scale(t):
        return np.clip(np.floor((t * s + 50.0) / 100.0), 1, 255).astype(np.int32)
    return _scale(_BASE_Q_Y), _scale(_BASE_Q_C)


# ============================================================
# 2. color
# ============================================================

# OpenCV gives us BT.601 YCrCb directly, but we swap the channel order so the
# rest of the code sees (Y, Cb, Cr) for clarity.
def bgr_to_ycbcr(bgr):
    yc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    # YCrCb -> YCbCr
    return yc[..., [0, 2, 1]]


def ycbcr_to_bgr(ycbcr):
    yc = ycbcr[..., [0, 2, 1]].astype(np.float32)
    yc = np.clip(yc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(yc, cv2.COLOR_YCrCb2BGR)


def chroma_down(plane):
    """4:2:0 box-filter downsample."""
    h, w = plane.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    p = plane[:h2, :w2]
    return p.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def chroma_up(plane, target_hw):
    th, tw = target_hw
    up = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)
    return up[:th, :tw]


# ============================================================
# 3. DCT (uses cv2.dct, which expects float32)
# ============================================================

def _pad_to(plane, m):
    h, w = plane.shape
    ph = (-h) % m
    pw = (-w) % m
    return np.pad(plane, ((0, ph), (0, pw)), mode="edge")


def dct_quant_plane(plane, q_table, block):
    """plane -> centred -> 8x8 DCT -> quantise."""
    centred = plane.astype(np.float32) - 128.0
    return _block_dct_quant(centred, q_table, block)


def dct_quant_residual(residual, q_table, block):
    """Same as `dct_quant_plane` but the input is already zero-mean."""
    return _block_dct_quant(residual.astype(np.float32), q_table, block)


def _block_dct_quant(plane, q_table, block):
    h, w = plane.shape
    q_out = np.empty((h // block, w // block, block, block), dtype=np.int16)
    for by in range(h // block):
        for bx in range(w // block):
            tile = plane[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            d = cv2.dct(tile)
            q_out[by, bx] = np.round(d / q_table).astype(np.int16)
    return q_out


def idct_dequant_plane(q_blocks, q_table, block, recentre=True):
    """Inverse of `dct_quant_plane`."""
    rows, cols = q_blocks.shape[:2]
    h, w = rows * block, cols * block
    out = np.empty((h, w), dtype=np.float32)
    for by in range(rows):
        for bx in range(cols):
            d = q_blocks[by, bx].astype(np.float32) * q_table
            t = cv2.idct(d)
            out[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = t
    if recentre:
        out += 128.0
    return out


# ============================================================
# 4. motion estimation — Three-Step Search (TSS)
# ============================================================

def _sad(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def _fetch(ref_pad, y0, x0, mb, pad):
    return ref_pad[y0 + pad:y0 + pad + mb, x0 + pad:x0 + pad + mb]


def three_step_search(current, reference, mb, search):
    """TSS — classic O(log s) block matching.

    Visits 9 points at step s, halves s, repeats until s == 0.
    """
    h, w = current.shape
    rows, cols = h // mb, w // mb
    pad = search
    ref_pad = np.pad(reference, pad, mode="edge")
    vectors = np.zeros((rows, cols, 2), dtype=np.int16)

    for by in range(rows):
        for bx in range(cols):
            y0, x0 = by * mb, bx * mb
            block = current[y0:y0 + mb, x0:x0 + mb]

            best_dy, best_dx = 0, 0
            best = _sad(block, _fetch(ref_pad, y0, x0, mb, pad))
            step = max(1, search // 2)

            while step >= 1:
                cy, cx = best_dy, best_dx
                for dy in (cy - step, cy, cy + step):
                    for dx in (cx - step, cx, cx + step):
                        if abs(dy) > search or abs(dx) > search:
                            continue
                        cand = _fetch(ref_pad, y0 + dy, x0 + dx, mb, pad)
                        c = _sad(block, cand)
                        if c < best:
                            best, best_dy, best_dx = c, dy, dx
                step //= 2

            vectors[by, bx] = (best_dy, best_dx)
    return vectors


def motion_compensate(reference, vectors, mb):
    """Build a prediction from `reference` according to `vectors`."""
    h, w = reference.shape
    rows, cols, _ = vectors.shape
    pad = int(np.max(np.abs(vectors))) if vectors.size else 0
    ref_pad = np.pad(reference, pad, mode="edge")
    pred = np.zeros_like(reference)
    for by in range(rows):
        for bx in range(cols):
            dy, dx = vectors[by, bx]
            y0, x0 = by * mb, bx * mb
            pred[y0:y0 + mb, x0:x0 + mb] = ref_pad[y0 + dy + pad:y0 + dy + pad + mb,
                                                  x0 + dx + pad:x0 + dx + pad + mb]
    return pred


# ============================================================
# 5. bitstream — struct-packed + bz2.
# ============================================================

_MAGIC = b"MV2\x00"
_DTYPE_TAGS = {np.int8: 0, np.int16: 1, np.int32: 2}
_TAG_DTYPES = {v: np.dtype(k) for k, v in _DTYPE_TAGS.items()}


def _pack_array(arr):
    """Encode an ndarray inline: <ndim, *shape, dtype_tag, raw_bytes>."""
    shape = arr.shape
    header = struct.pack("<B", len(shape)) + struct.pack(f"<{len(shape)}I", *shape)
    tag = _DTYPE_TAGS[arr.dtype.type]
    return header + struct.pack("<B", tag) + arr.tobytes()


def _unpack_array(buf, offset):
    ndim = buf[offset]
    offset += 1
    shape = struct.unpack_from(f"<{ndim}I", buf, offset)
    offset += 4 * ndim
    tag = buf[offset]
    offset += 1
    dtype = _TAG_DTYPES[tag]
    count = int(np.prod(shape)) if shape else 0
    arr = np.frombuffer(buf, dtype=dtype, count=count, offset=offset).reshape(shape)
    offset += count * dtype.itemsize
    # numpy arrays from frombuffer are read-only; copy so callers can mutate.
    return arr.copy(), offset


def pack_bitstream(params, luma_shape, chroma_shape, records):
    """Serialise + bz2-compress.

    `records` is a list of dicts with keys depending on type:
        I: {"type": "I", "y": .., "cb": .., "cr": ..}
        P: {"type": "P", "y": .., "cb": .., "cr": .., "mv": ..}
    """
    parts = [_MAGIC, b"\x01"]  # version 1
    parts.append(struct.pack(
        "<H HH HH B B B B B",
        len(records),
        luma_shape[0], luma_shape[1],
        chroma_shape[0], chroma_shape[1],
        params.gop, params.quality, params.macroblock,
        params.search, 1 if params.subsample else 0,
    ))
    for rec in records:
        if rec["type"] == "I":
            parts.append(b"I")
            parts.append(_pack_array(rec["y"]))
            parts.append(_pack_array(rec["cb"]))
            parts.append(_pack_array(rec["cr"]))
        else:
            parts.append(b"P")
            parts.append(_pack_array(rec["mv"]))
            parts.append(_pack_array(rec["y"]))
            parts.append(_pack_array(rec["cb"]))
            parts.append(_pack_array(rec["cr"]))

    raw = b"".join(parts)
    return bz2.compress(raw, compresslevel=9)


def unpack_bitstream(blob):
    raw = bz2.decompress(blob)
    if raw[:4] != _MAGIC:
        raise ValueError("Not a MV2 bitstream")
    version = raw[4]
    if version != 1:
        raise ValueError(f"Unsupported bitstream version: {version}")
    offset = 5
    (n_frames, ly, lx, cy, cx, gop, q, mb, search, sub) = struct.unpack_from(
        "<H HH HH B B B B B", raw, offset)
    offset += struct.calcsize("<H HH HH B B B B B")
    params = Params(gop=gop, quality=q, block=8,
                    macroblock=mb, search=search, subsample=bool(sub))
    luma_shape = (ly, lx)
    chroma_shape = (cy, cx)

    records = []
    for _ in range(n_frames):
        tag = chr(raw[offset]); offset += 1
        if tag == "I":
            y, offset = _unpack_array(raw, offset)
            cb, offset = _unpack_array(raw, offset)
            cr, offset = _unpack_array(raw, offset)
            records.append({"type": "I", "y": y, "cb": cb, "cr": cr})
        elif tag == "P":
            mv, offset = _unpack_array(raw, offset)
            y, offset = _unpack_array(raw, offset)
            cb, offset = _unpack_array(raw, offset)
            cr, offset = _unpack_array(raw, offset)
            records.append({"type": "P", "mv": mv, "y": y, "cb": cb, "cr": cr})
        else:
            raise ValueError(f"Unknown frame tag: {tag!r}")
    return params, luma_shape, chroma_shape, records


# ============================================================
# 6. top-level encode / decode
# ============================================================

def encode(frames_bgr, params=DEFAULT_PARAMS):
    """Encode a list of BGR frames to a bz2 bitstream (bytes)."""
    q_y, q_c = make_qtables(params.quality)
    mb = params.macroblock
    block = params.block

    luma_shape = None
    chroma_shape = None
    records = []
    prev = None  # tuple of reconstructed (Y, Cb, Cr) at padded resolution

    for idx, bgr in enumerate(frames_bgr):
        ycbcr = bgr_to_ycbcr(bgr)
        y = ycbcr[..., 0]
        cb = ycbcr[..., 1]
        cr = ycbcr[..., 2]
        if params.subsample:
            cb = chroma_down(cb)
            cr = chroma_down(cr)

        y = _pad_to(y, mb)
        cb = _pad_to(cb, block)
        cr = _pad_to(cr, block)

        if luma_shape is None:
            luma_shape = y.shape
            chroma_shape = cb.shape

        is_intra = (idx % params.gop) == 0 or prev is None
        if is_intra:
            qy = dct_quant_plane(y, q_y, block)
            qcb = dct_quant_plane(cb, q_c, block)
            qcr = dct_quant_plane(cr, q_c, block)
            records.append({"type": "I", "y": qy, "cb": qcb, "cr": qcr})
            ry = np.clip(idct_dequant_plane(qy, q_y, block), 0, 255)
            rcb = np.clip(idct_dequant_plane(qcb, q_c, block), 0, 255)
            rcr = np.clip(idct_dequant_plane(qcr, q_c, block), 0, 255)
        else:
            py, pcb, pcr = prev
            mv = three_step_search(y.astype(np.uint8), py.astype(np.uint8),
                                   mb, params.search)
            pred_y = motion_compensate(py, mv, mb)
            res_y = y.astype(np.float32) - pred_y.astype(np.float32)
            qy = dct_quant_residual(res_y, q_y, block)
            recon_res_y = idct_dequant_plane(qy, q_y, block, recentre=False)
            ry = np.clip(pred_y + recon_res_y, 0, 255)

            if params.subsample:
                mv_c = (mv // 2).astype(np.int16)
                mb_c = mb // 2
            else:
                mv_c = mv
                mb_c = mb
            pred_cb = motion_compensate(pcb, mv_c, mb_c)
            pred_cr = motion_compensate(pcr, mv_c, mb_c)
            res_cb = cb.astype(np.float32) - pred_cb.astype(np.float32)
            res_cr = cr.astype(np.float32) - pred_cr.astype(np.float32)
            qcb = dct_quant_residual(res_cb, q_c, block)
            qcr = dct_quant_residual(res_cr, q_c, block)
            recon_res_cb = idct_dequant_plane(qcb, q_c, block, recentre=False)
            recon_res_cr = idct_dequant_plane(qcr, q_c, block, recentre=False)
            rcb = np.clip(pred_cb + recon_res_cb, 0, 255)
            rcr = np.clip(pred_cr + recon_res_cr, 0, 255)

            records.append({"type": "P", "mv": mv,
                            "y": qy, "cb": qcb, "cr": qcr})

        prev = (ry, rcb, rcr)

    return pack_bitstream(params, luma_shape, chroma_shape, records)


def decode(blob, output_shape=None):
    """Inverse of `encode`.  Returns a list of BGR uint8 frames.

    `output_shape` is the original (H, W) before padding; if None, the
    padded shape is returned.
    """
    params, luma_shape, chroma_shape, records = unpack_bitstream(blob)
    q_y, q_c = make_qtables(params.quality)
    mb = params.macroblock
    block = params.block

    out_frames = []
    prev = None

    for rec in records:
        if rec["type"] == "I":
            y = np.clip(idct_dequant_plane(rec["y"], q_y, block), 0, 255)
            cb = np.clip(idct_dequant_plane(rec["cb"], q_c, block), 0, 255)
            cr = np.clip(idct_dequant_plane(rec["cr"], q_c, block), 0, 255)
        else:
            if prev is None:
                raise RuntimeError("P-frame before any I-frame")
            py, pcb, pcr = prev
            mv = rec["mv"]
            pred_y = motion_compensate(py, mv, mb)
            res_y = idct_dequant_plane(rec["y"], q_y, block, recentre=False)
            y = np.clip(pred_y + res_y, 0, 255)

            if params.subsample:
                mv_c = (mv // 2).astype(np.int16)
                mb_c = mb // 2
            else:
                mv_c = mv
                mb_c = mb
            pred_cb = motion_compensate(pcb, mv_c, mb_c)
            pred_cr = motion_compensate(pcr, mv_c, mb_c)
            res_cb = idct_dequant_plane(rec["cb"], q_c, block, recentre=False)
            res_cr = idct_dequant_plane(rec["cr"], q_c, block, recentre=False)
            cb = np.clip(pred_cb + res_cb, 0, 255)
            cr = np.clip(pred_cr + res_cr, 0, 255)

        prev = (y, cb, cr)

        if params.subsample:
            cb_full = chroma_up(cb, y.shape)
            cr_full = chroma_up(cr, y.shape)
        else:
            cb_full = cb
            cr_full = cr

        ycbcr = np.stack([y, cb_full, cr_full], axis=-1)
        bgr = ycbcr_to_bgr(ycbcr)

        if output_shape is not None:
            h, w = output_shape
            bgr = bgr[:h, :w]
        out_frames.append(bgr)

    return out_frames, params, records


# ============================================================
# 7. metrics
# ============================================================

def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def frame_breakdown(records):
    n_i = sum(1 for r in records if r["type"] == "I")
    n_p = sum(1 for r in records if r["type"] == "P")
    return n_i, n_p
