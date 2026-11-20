#!/usr/bin/env python3
"""A/B of the per-pixel confidence weighting (item 2): fuse a recording with and without conf, compare
floor thickness (robust sigma of y within 6 cm of the floor), point count and isolated-point fraction.
usage: test_conf.py RECDIR [--every 2] [--limit 80] [--g0 8] [--gmin 1.5] [--jump 0.04]"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw
from tsdf import TSDF
import floor as floormod
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--every", type=int, default=2); ap.add_argument("--limit", type=int, default=80)
ap.add_argument("--g0", type=float, default=8.0); ap.add_argument("--gmin", type=float, default=1.5); ap.add_argument("--jump", type=float, default=0.04)
ap.add_argument("--min-frames", type=int, default=3)
a = ap.parse_args()
ls = LiveStereo(scale=0.5, max_depth=4.0)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
recs = [r for r in recs if abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015 and r["left"]["pose_valid"]][::a.every][:a.limit]
cases = {"base": None, "conf": dict(g0=a.g0, gmin=a.gmin, jump=a.jump)}
fus = {k: TSDF(voxel=0.02, max_depth=2.5) for k in cases}
t0 = time.time(); frac_zero = []
for r in recs:
    i = r["i"]
    L = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read()); R = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
    Z, rgb, _ = ls.depth(L, R, far=fus["base"].carve_depth, half=False)
    T = ls.world_from_rect(np.array(r["left"]["pose"]))
    for k, kw in cases.items():
        if kw is None: fus[k].integrate(ls, Z, rgb, T, t=r["left"]["ts"])
        else:
            c = ls.confidence(Z, rgb, **kw); frac_zero.append(((c == 0) & (Z > 0)).sum() / max((Z > 0).sum(), 1))
            fus[k].integrate(ls, Z, rgb, T, t=r["left"]["ts"], conf=c)
print(f"{len(recs)} frames {time.time()-t0:.0f}s; conf rejects {100*np.mean(frac_zero):.1f}% of valid pixels")
for k, f in fus.items():
    P = f.extract(min_frames=a.min_frames)[0]
    y, sh = floormod.floor_level(P)
    s = ""
    if y is not None:
        q = P[np.abs(P[:, 1] - y) < 0.06, 1]; med = np.median(q); s = f"floor y {y:+.3f} sharp {sh:.1f} n {len(q)} sigma {1.4826*np.median(np.abs(q-med))*1000:.1f} mm"
    tree = cKDTree(P); nn = tree.query_ball_point(P, 0.04, return_length=True)
    print(f"{k:5s} {len(P):7d} pts  isolated(<4 nbrs in 4cm) {100*np.mean(nn<5):.1f}%  {s}")
