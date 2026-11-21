#!/usr/bin/env python3
"""Debug capture: a stereo still (tools/stills.py, same directory layout and still.tiff) of the pair pcserver just processed,
PLUS every intermediate buffer of the live depth pipeline for that pair, for offline analysis / optimisation.

captures/stills/<YYYYmmdd_HHMMSS_mmm>/   (everything save_still writes, see tools/stills.py) and
  pipeline.npz    all buffers (np.load(..., allow_pickle=False)); names/shapes/dtypes/meaning in pipeline.json "buffers"
  pipeline.json   manifest: per-buffer description, stage order, timings, matcher + rectification parameters, blur gate
                  decision, poses actually used, pcserver arguments
  index.html      pipeline trace page (tools/debugreport.py): every stage with images, timings, automatic findings
  preview/*.jpg   its images

Stage order (= tools/pcserver.py main loop):
  raw RAW10 (left.y10p/right.y10p) -> bin_raw10 (binned_L/R: half-res BGR float, black removed)
  -> grey + gain (grey_L/R) -> rectify remap (rect_grey_L/R, maps remap_L/R) -> matcher input (matcher_in_L/R, mirrored when
  camL is the right image) -> SGBM (matcher_out_int16, 1/16 px) -> disparity (disp_sgbm, disp after optional median test)
  -> depth (Z: k f B / (disp + d0), range-limited) -> hand mask (hand_mask) -> Z_fused (what TSDF.integrate got, with cone_mask
  in paint mode) ; conf = LiveStereo.confidence (computed for analysis even when pcserver --conf is off).
Load:  d = np.load("pipeline.npz"); m = json.load(open("pipeline.json"))
"""
import json, os, time
import numpy as np
import cv2

import stills
import debugreport

DESCR = {
    "binned_L": "half-res BGR float32 (raw units, black removed), output of tsdf.bin_raw10 (2x2 binning demosaic)",
    "binned_R": "same, right camera",
    "wb": "grey-world white balance gains (BGR) estimated on binned_L",
    "grey_gain": "gain s mapping the grey image to 0..255 (99.5th percentile of binned_L grey)",
    "grey_L": "uint8 grey (BGR2GRAY of binned_L * s), before rectification",
    "grey_R": "same, right camera",
    "remap_L": "rectification map of the left camera (cv2 CV_16SC2 + CV_16UC1 interpolation table pair, stacked: map1 int16 HxWx2)",
    "remap_L_frac": "interpolation table of remap_L (uint16 HxW)",
    "remap_R": "rectification map of the right camera (map1)",
    "remap_R_frac": "interpolation table of remap_R",
    "rect_grey_L": "uint8 rectified grey, camL (geometry of all following buffers)",
    "rect_grey_R": "uint8 rectified grey, camR (row-aligned with rect_grey_L)",
    "matcher_in_L": "exact left input of the matcher (mirrored copies of rect_grey_* when params.left_first is false)",
    "matcher_in_R": "exact right input of the matcher",
    "matcher_out_int16": "raw SGBM output, int16 fixed point (1/16 px), MATCHER orientation (mirrored when left_first is false); < 0 invalid",
    "disp_sgbm": "float32 disparity px in camL orientation, before the optional median consistency test; <= 0 invalid",
    "disp": "float32 disparity px after the median test (= what the depth model used)",
    "rect_colour_raw": "rectified camL colour, BGR float32 raw units (before white balance / tone)",
    "Z": "float32 depth m (rectified camL, along the optical axis), 0 = invalid, after the range limits [min_depth, far]",
    "rgb": "uint8 RGB rectified camL colour as fused (white balance, gain, highlight handling, saturation/gamma finish)",
    "rgb_R": "uint8 RGB rectified camR colour (same tone as rgb), for viewing the pair",
    "hand_mask": "bool, pixels removed as hands/controllers (tools/handmask.py); all False when not applied",
    "cone_mask": "bool, paint-mode cone (pixels allowed to integrate); all True when not in paint mode",
    "Z_fused": "float32 depth actually passed to TSDF.integrate (Z with hand mask applied); zeros if the pair was not fused",
    "conf": "float32 0..1 LiveStereo.confidence(Z, rgb) (texture x depth-edge); used by the fusion only with pcserver --conf",
    "stage_from_rect_tracked": "4x4 STAGE-from-rectified-camL from the SteamVR pose",
    "stage_from_rect_used": "4x4 pose used for the fusion (tracked, or origin-corrected with --apply-origin-corr)",
    "far": "far limit (m) used for Z",
}


def _cmap(x, lo, hi, valid):
    v = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    im = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)
    im[~valid] = 0
    return im


def save(shm, ls, pair, bufs, info, out_root=stills.OUT):
    """pair = (mL, bL, mR, bR) as processed; bufs = name -> ndarray (LiveStereo.debug + pcserver buffers); info = JSON-able dict
    (timings, gate, args, ...). Writes the still (with the SAME depth/colour as the live pipeline) + pipeline.npz/json. -> dir"""
    t0 = time.monotonic()
    rgbR = bufs.get("rgb_R")
    d = stills.save_still(shm, ls, out_root=out_root, pair=pair, depth=(bufs["Z"], bufs["rgb"], rgbR))
    t_still = time.monotonic() - t0
    m1, m2 = ls.mapL; r1, r2 = ls.mapR
    bufs = dict(bufs, remap_L=m1, remap_L_frac=m2, remap_R=r1, remap_R_frac=r2)
    arrs = {k: np.ascontiguousarray(np.asarray(v)) for k, v in bufs.items() if v is not None}
    np.savez(os.path.join(d, "pipeline.npz"), **arrs)             # uncompressed: fast on the headset, ~60 MB at scale 0.35
    man = dict(format="pcdebug/1", still_dir=os.path.basename(d),
               buffers={k: dict(shape=list(v.shape), dtype=str(v.dtype), what=DESCR.get(k, "")) for k, v in arrs.items()},
               params=ls.debug_params(), **info)
    man.setdefault("timings_ms", {})["write_still"] = round(t_still * 1000, 1)
    man["timings_ms"]["write_pipeline"] = round((time.monotonic() - t0 - t_still) * 1000, 1)
    with open(os.path.join(d, "pipeline.json"), "w") as f:
        json.dump(man, f, indent=1)
    try:                                         # pipeline trace page: DIR/index.html + DIR/preview/*.jpg
        t1 = time.monotonic(); debugreport.build(d)
        print(f"debug report: {d}/index.html ({(time.monotonic() - t1) * 1000:.0f} ms)", flush=True)
    except Exception as e:
        print("debug report failed:", e, flush=True)
    return d
