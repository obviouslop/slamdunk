#!/usr/bin/env python3
"""Signed metric bias of fused clouds vs held-out stereo depth: for cloud points whose test-view depth is
within +-12 cm, median (Z_measured - z_point) by distance (negative = cloud lies behind the measured surface)."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("ply", nargs="+"); ap.add_argument("--frames", default="1:309:6")
a = ap.parse_args(); ls = LiveStereo(scale=0.35, max_depth=2.5)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
test = recs[slice(*[int(x) for x in a.frames.split(":")])]
cache = os.path.join(a.rec, "holdout_depth_" + a.frames.replace(":", "_") + ".npz")
if os.path.exists(cache):
    z = np.load(cache); depths = list(zip(z["Z"], z["T"]))
else:
    depths = []
    for r in test:
        i = r["i"]
        L = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read()); R = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
        Z, _, _ = ls.depth(L, R, far=5.0); depths.append((Z, ls.world_from_rect(r["left"]["pose"])))
    np.savez(cache, Z=np.array([d[0] for d in depths]), T=np.array([d[1] for d in depths]))
f = ls.f; cx, cy = ls.P1[0, 2], ls.P1[1, 2]
for p in a.ply:
    raw = open(p, 'rb').read(); h = raw.index(b'end_header\n') + 11
    P = np.frombuffer(raw[h:], dtype=[('p', '<f4', 3), ('c', 'u1', 3)])['p'].astype(float)
    D, ZZ = [], []
    for Z, T in depths:
        Pc = (P - T[:3, 3]) @ T[:3, :3]; z = Pc[:, 2]; n = Z.shape[0]; ok = z > 0.3
        u = np.round(f * Pc[ok, 0] / z[ok] + cx).astype(int); v = np.round(f * Pc[ok, 1] / z[ok] + cy).astype(int)
        inb = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        zz = z[ok][inb]; Zm = Z[v[inb], u[inb]]; g = (Zm > 0) & (zz < 2.5) & (np.abs(Zm - zz) < 0.12)
        D.append(Zm[g] - zz[g]); ZZ.append(zz[g])
    D, ZZ = np.concatenate(D), np.concatenate(ZZ)
    s = f"{os.path.basename(p):28s} n={len(D):7d} median {np.median(D)*100:+.2f} cm |"
    for lo, hi in ((0.3, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5)):
        k = (ZZ >= lo) & (ZZ < hi)
        if k.sum() > 100: s += f"  {lo:.1f}-{hi:.1f} m: {np.median(D[k])*100:+.2f}"
    print(s)
    h, e = np.histogram(D * 100, bins=np.arange(-12, 12.1, 1.5)); print("    hist cm", " ".join(f"{x/len(D)*1000:.0f}" for x in h))
