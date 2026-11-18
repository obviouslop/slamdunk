#!/usr/bin/env python3
"""Fusion bias meter: per-frame alignment offset of each new frame vs the map of earlier frames (no feedback).
Negative dy = map lies below the new measurement (floor-dominated data: 'map behind the surface')."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo; from tsdf import TSDF
ls = LiveStereo(scale=0.35, max_depth=2.5)
z = np.load(sys.argv[1]); Zs, Cs, Ts = z["Z"], z["C"], z["T"]
N = int(os.environ.get("NFR", len(Zs)))
for v in sys.argv[2:]:
    kw = dict(voxel=0.02, max_depth=2.5)
    for item in v.split(",")[1:]:
        k, val = item.split("="); kw[k] = float(val)
    ts = TSDF(**kw); dy = []; dd = []
    for j in range(N):
        Zj = Zs[j].astype(np.float32)
        if j >= 3:
            Ta, info = ts.align(ls, Zj, Ts[j], prior_rot=0.005, prior_trans=0.05)
            if Ta is not None:
                d = Ta[:3, 3] - Ts[j][:3, 3]; dy.append(d[1]); dd.append(np.linalg.norm(d))
        ts.integrate(ls, Zj, Cs[j], Ts[j])
    dy = np.array(dy)
    print(f"{v:36s} dy median {np.median(dy)*100:+.2f} cm (first 40: {np.median(dy[:40])*100:+.2f})  |d| median {np.median(dd)*100:.2f} cm", flush=True)
