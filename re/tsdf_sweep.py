#!/usr/bin/env python3
"""Fuse the even live7 frames with TSDF variants (depth maps cached) -> PLYs for re/holdout_bias.py."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw
from tsdf import TSDF
import pcpub
rec = sys.argv[1]; out = sys.argv[2]; variants = sys.argv[3:]
ls = LiveStereo(scale=0.35, max_depth=2.5)
recs = [json.loads(l) for l in open(os.path.join(rec, "frames.jsonl"))]
recs = [r for r in recs if r["left"]["pose_valid"] and abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015][0::2]
cache = os.path.join(rec, "even_depth.npz")
if not os.path.exists(cache):
    Zs, Cs, Ts = [], [], []
    for r in recs:
        i = r["i"]
        L = unpack_raw(open(os.path.join(rec, f"{i:05d}_left.y10p"), "rb").read()); R = unpack_raw(open(os.path.join(rec, f"{i:05d}_right.y10p"), "rb").read())
        Z, rgb, _ = ls.depth(L, R, far=4.0); Zs.append(Z.astype(np.float16)); Cs.append(rgb); Ts.append(ls.world_from_rect(r["left"]["pose"]))
    np.savez(cache, Z=np.array(Zs), C=np.array(Cs), T=np.array(Ts))
z = np.load(cache); Zs, Cs, Ts = z["Z"], z["C"], z["T"]
for v in variants:
    kw = dict(voxel=0.02, max_depth=2.5)
    for item in v.split(",")[1:]:
        k, val = item.split("="); kw[k] = float(val)
    mf = int(kw.pop("min_frames", 4)); edge = kw.pop("edge", 0)
    ts = TSDF(**kw)
    for Z, C, T in zip(Zs, Cs, Ts):
        Z = Z.astype(np.float32)
        if edge:   # drop pixels at depth discontinuities (3x3 range > edge * Z)
            import cv2
            Zv = np.where(Z > 0, Z, np.nan).astype(np.float32)
            mx = cv2.dilate(np.nan_to_num(Zv, nan=0), np.ones((3, 3), np.uint8)); mn = -cv2.dilate(-np.nan_to_num(Zv, nan=1e6), np.ones((3, 3), np.uint8))
            Z = np.where((mx - mn) < edge * Z, Z, 0).astype(np.float32)
        ts.integrate(ls, Z, C, T)
    P, Cc, w = ts.extract(min_frames=mf)
    fn = os.path.join(out, f"sw_{v.split(',')[0]}.ply"); pcpub.write_ply(fn, P, Cc); print(fn, len(P), flush=True)
