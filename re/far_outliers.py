#!/usr/bin/env python3
"""How often does a frame measure a surface BEHIND a surface known from close range (= false match, "too far")?
Uses the far_bias.py cache + near map. Per reference-distance bin: fraction of pixels measured >10 % behind / in front
/ within +-10 % of the near-mapped surface (no selection window).  far_outliers.py captures/live16"""
import sys, os, json, numpy as np
sys.argv += [] ; rec = sys.argv[1]
sys.argv = [sys.argv[0], rec, "--iters", "0"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "far_bias.py")).read().split("print(f\"{N} frames")[0])
M = near_map(k, d0, a.near)
E, Zr_all = [], []
for j in range(N):
    T = Ts[j]; Pc = (M - T[:3, 3]) @ T[:3, :3]; Pc = Pc[Pc[:, 2] > 0.3]
    u = np.round(f2 * Pc[:, 0] / Pc[:, 2] + cx2).astype(int); v = np.round(f2 * Pc[:, 1] / Pc[:, 2] + cy2).astype(int)
    inb = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1); u, v, zz = u[inb], v[inb], Pc[inb, 2]
    zb = np.full((h, w), np.inf, np.float32)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            np.minimum.at(zb, (v + dv, u + du), zz)
    Zm = depth(Ds[j], k, d0)
    sel = np.isfinite(zb) & (zb > 1.4) & (Zm > 0)
    E.append(Zm[sel] / zb[sel] - 1); Zr_all.append(zb[sel])
E = np.concatenate(E); Zr = np.concatenate(Zr_all)
print(f"k={k} d0={d0}")
for lo, hi in ((1.4, 1.8), (1.8, 2.2), (2.2, 2.6), (2.6, 3.0), (3.0, 3.5), (3.5, 4.5)):
    m = (Zr >= lo) & (Zr < hi)
    if m.sum() > 500:
        e = E[m]
        print(f"ref {lo:.1f}-{hi:.1f} m: n={m.sum():7d}  behind>10%: {np.mean(e > 0.10)*100:5.1f} %  behind>30%: {np.mean(e > 0.30)*100:5.1f} %  "
              f"in front>10%: {np.mean(e < -0.10)*100:5.1f} %  within: {np.mean(np.abs(e) <= 0.10)*100:5.1f} %")
