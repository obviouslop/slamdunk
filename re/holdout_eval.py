#!/usr/bin/env python3
"""Hold-out accuracy of a fused cloud: project points into stereo depth maps of frames NOT used for fusion.
   per visible point: r = (Z_measured - z_point) / sigma(Z);  |r|<2 -> consistent; r>3 -> floater (seen through)
usage: holdout_eval.py RECDIR cloud.ply --frames 0:200:2  (python slice of frames.jsonl indices to test)"""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("ply", nargs="+"); ap.add_argument("--frames", default="0:200:10")
a = ap.parse_args()
ls = LiveStereo(scale=0.35, max_depth=2.5)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
sl = slice(*[int(x) for x in a.frames.split(":")]); test = recs[sl]
depths = []
for r in test:
    i = r["i"]
    L = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read()); R = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
    Z, _, _ = ls.depth(L, R, far=5.0); depths.append((Z, ls.world_from_rect(r["left"]["pose"])))
for p in a.ply:
    raw = open(p, 'rb').read(); h = raw.index(b'end_header\n') + 11
    P = np.frombuffer(raw[h:], dtype=[('p', '<f4', 3), ('c', 'u1', 3)])['p'].astype(float)
    res = []
    for Z, T in depths:
        Pc = (P - T[:3, 3]) @ T[:3, :3]
        z = Pc[:, 2]; f = ls.f; cx, cy = ls.P1[0, 2], ls.P1[1, 2]; n = Z.shape[0]
        ok = z > 0.3
        u = np.round(f * Pc[ok, 0] / z[ok] + cx).astype(int); v = np.round(f * Pc[ok, 1] / z[ok] + cy).astype(int)
        inb = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        zz = z[ok][inb]; Zm = Z[v[inb], u[inb]]; good = (Zm > 0) & (zz < 2.5)
        sig = Zm[good] ** 2 / (ls.f * ls.baseline) * 0.35
        res.append((Zm[good] - zz[good]) / np.maximum(sig, 0.005))
    r = np.concatenate(res)
    print(f"{os.path.basename(p)}: {len(P)} pts, {len(r)} visible samples: consistent(|r|<2) {np.mean(np.abs(r)<2)*100:.1f}%  "
          f"floaters(r>3) {np.mean(r>3)*100:.1f}%  behind(r<-3) {np.mean(r<-3)*100:.1f}%  median r {np.median(r):.2f}  hist " + " ".join(f"{x:.0f}" for x in np.histogram(np.clip(r,-8,8),bins=16,range=(-8,8))[0]*1000/len(r)))
