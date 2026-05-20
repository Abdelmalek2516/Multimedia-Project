# MPEG-4-lite encoder (Python)

Tiny educational video codec covering the five required stages:

| stage | technique |
|-------|-----------|
| 1. pre-processing | BGR → YCbCr (BT.601) + 4:2:0 chroma subsample |
| 2. I-frames       | 8×8 DCT (`cv2.dct`) + JPEG quant tables       |
| 3. P-frames       | three-step block-matching on 16×16 macroblocks + DCT-coded residual |
| 4. entropy        | struct-packed bitstream → `bz2` level 9       |
| 5. eval + viz     | PSNR, compression ratio, single matplotlib figure |

## Files

```
mpeg_codec.py     - codec functions (encode, decode, helpers)
gen_test_frames.py- make a synthetic sample clip
viz.py            - single-figure stage visualisation
run.py            - CLI driver with sub-commands (encode/decode/viz/sweep)
```

## Install

```
pip install numpy opencv-python matplotlib
```

(no scipy — uses `cv2.dct` directly.)

## Usage

```bash
# generate sample frames
python gen_test_frames.py -o sample_frames -n 12

# encode to .bin
python run.py encode sample_frames -o video.bin --gop 8 --q 50

# decode, write recon images + report PSNR
python run.py decode video.bin -o decoded --ref sample_frames

# build the pipeline visualisation
python run.py viz sample_frames video.bin -o pipeline.png

# sweep quality & GOP for the report
python run.py sweep sample_frames -o experiments.png
```

## Bitstream (MV2)

`magic(4) | version(1) | header | { frame_tag(1) | packed_arrays... }*`

All arrays are packed inline as `ndim(1) | shape(4*ndim) | dtype_tag(1) | bytes`.
The complete blob is then `bz2.compress`ed at level 9.

## Parameters

`Params(gop, quality, block=8, macroblock=16, search, subsample)`
