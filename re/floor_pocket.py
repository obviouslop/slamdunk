#!/usr/bin/env python3
"""Find depressions / pockets below the floor in a recording: fuse it, fit the floor height (lowest large horizontal
histogram peak), list clusters of points below it, and say whether the source pixels were saturated (specular highlights).
  floor_pocket.py captures/live18 [--below 0.04]"""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo
from tsdf import TSDF, bin_raw10
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--below", type=float, default=0.04); a = ap.parse_args()
ls = LiveStereo(scale=0.35, max_depth=2.5); ts = TSDF(voxel=0.02, max_depth=2.5, carve_w=0.3, back_k=8, back_w=3)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]; recs = [r for r in recs if r["left"]["pose_valid"]]
sat = []
for n, r in enumerate(recs):
    i = r["i"]; cL = bin_raw10(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read()); cR = bin_raw10(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
    Z, rgb, _ = ls.depth(cL, cR, far=4.0, half=True); T = ls.world_from_rect(r["left"]["pose"])
    ts.integrate(ls, Z, rgb, T, t=r["left"]["ts"]); print(f"\r{n+1}/{len(recs)}", end="", flush=True)
P, C, w, t, d = ts.extract(min_frames=6, attrs=True); print(f"\n{len(P)} points")
h, e = np.histogram(P[:, 1], bins=np.arange(-0.5, 0.6, 0.02)); big = np.flatnonzero(h > 0.25 * h.max())
fl = (e[big[0]] + e[big[0] + 1]) / 2; print(f"floor peak y = {fl:+.3f} (peak count {h.max()}, lowest strong bin {h[big[0]]})")
lo = P[:, 1] < fl - a.below
print(f"{lo.sum()} points more than {a.below*100:.0f} cm below the floor")
if lo.sum():
    from scipy.cluster.hierarchy import fcluster, linkage
    Q = P[lo]; sub = Q[::max(1, len(Q)//4000)]; lab = fcluster(linkage(sub, "single"), 0.15, "distance")
    for k in np.argsort([-np.sum(lab == c) for c in np.unique(lab)])[:6]:
        c = np.unique(lab)[k]; m = sub[lab == c]
        print(f"  cluster {len(m)*max(1,len(Q)//4000):6d} pts  x {m[:,0].min():+.2f}..{m[:,0].max():+.2f}  z {m[:,2].min():+.2f}..{m[:,2].max():+.2f}  y {m[:,1].min():+.2f}..{m[:,1].max():+.2f}  view dist median {np.median(d[lo][::max(1,len(Q)//4000)][lab==c]):.2f} m")
np.savez(os.path.join(a.rec, "..", "scratch", "pocket_pts.npz"), P=P, C=C, t=t, d=d, fl=fl)
