#!/usr/bin/env python3
"""Depth-bias analysis on a live recording: sample (world ray, disparity) per pixel, then fit a disparity
offset d0 and/or baseline scale k so horizontal surfaces (floor / ceiling) become flat and consistent.
  depth_bias.py captures/live7 --every 3 [--cache rays.npz]
Model: Z = k * f*B / (disp + d0)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw

ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--every", type=int, default=3)
ap.add_argument("--scale", type=float, default=0.35); ap.add_argument("--sub", type=int, default=3)
ap.add_argument("--cache"); a = ap.parse_args()
cache = a.cache or os.path.join(a.rec, f"rays_e{a.every}_s{a.scale}.npz")
if not os.path.exists(cache):
    ls = LiveStereo(scale=a.scale, max_depth=4.0)
    recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
    recs = [r for r in recs if abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015 and r["left"]["pose_valid"]][::a.every]
    O, D, Dp, F, U = [], [], [], [], []
    n = ls.size[0]; vv, uu = np.mgrid[0:n:a.sub, 0:n:a.sub]
    for k, r in enumerate(recs):
        i = r["i"]
        L = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read())
        R = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
        disp, *_ = ls.rectified(L, R)
        d = disp[vv, uu]; ok = d > 0.5
        W = ls.world_from_rect(np.array(r["left"]["pose"]))
        ray = np.stack([(uu[ok] - ls.P1[0, 2]) / ls.f, (vv[ok] - ls.P1[1, 2]) / ls.f, np.ones(ok.sum())], -1) @ W[:3, :3].T
        O.append(np.repeat(W[None, :3, 3], ok.sum(), 0).astype(np.float32)); D.append(ray.astype(np.float32))
        Dp.append(d[ok].astype(np.float32)); F.append(np.full(ok.sum(), k, np.int32)); U.append(np.stack([uu[ok], vv[ok]], -1).astype(np.int16))
        print(f"\r{k+1}/{len(recs)}", end="", flush=True)
    np.savez(cache, O=np.concatenate(O), D=np.concatenate(D), disp=np.concatenate(Dp), frame=np.concatenate(F),
             uv=np.concatenate(U), fB=ls.f * ls.baseline, f=ls.f, cx=ls.P1[0, 2])
    print()
z = np.load(cache); O, D, disp, fB = z["O"], z["D"], z["disp"], float(z["fB"])
print(f"{len(disp)} samples, fB={fB:.3f} (f={float(z['f']):.1f} px)")

def pts(d0=0.0, k=1.0):
    Z = k * fB / np.maximum(disp + d0, 0.3)
    return O + D * Z[:, None], Z

P, Z = pts()
ok = Z < 4.0
for name, sel in (("floor", (D[:, 1] < -0.6) & (P[:, 1] < 1.35)), ("ceiling", (D[:, 1] > 0.6) & (P[:, 1] > 3.0))):
    s = sel & ok
    y = P[s, 1]; m = np.median(y)
    print(f"{name}: n={s.sum()} median y={m:.3f}")
    for lo, hi in ((0.5, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0)):
        b = s & (Z >= lo) & (Z < hi)
        yb = P[b, 1]; yb = yb[np.abs(yb - m) < 0.4]
        if len(yb) > 100:
            print(f"   Z {lo:.1f}-{hi:.1f} m: n={len(yb):7d}  median y={np.median(yb):.3f}  IQR={np.subtract(*np.percentile(yb,[75,25])):.3f}")
