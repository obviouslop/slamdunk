#!/usr/bin/env python3
"""Synthetic check of tools/planes.py: floor / ceiling / wall / table (+ clutter, noise, a slightly tilted wall), stable ids over updates."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from planes import PlaneTracker
rng = np.random.default_rng(1)
def sheet(n, f): return f(rng.uniform(0, 1, n), rng.uniform(0, 1, n))
def cloud():
    fl = sheet(60000, lambda a, b: np.c_[a * 4 - 2, rng.normal(0, .01, len(a)), b * 4 - 2])
    ce = sheet(30000, lambda a, b: np.c_[a * 4 - 2, 2.5 + rng.normal(0, .01, len(a)), b * 4 - 2])
    wa = sheet(30000, lambda a, b: np.c_[a * 4 - 2, b * 2.5 + 0.0, -2 + 0.07 * (b - .5) + rng.normal(0, .01, len(a))])   # ~1.6 deg tilt
    tb = sheet(10000, lambda a, b: np.c_[a * 1 - .5, .75 + rng.normal(0, .01, len(a)), b * 1 - .5])
    cl = rng.uniform([-2, 0, -2], [2, 2.5, 2], (3000, 3))
    return np.vstack([fl, ce, wa, tb, cl]).astype(np.float32)
pt = PlaneTracker(path="captures/scratch/pcplanes_test.json")
for k in range(6):
    t0 = time.time(); it = pt.update(cloud() + rng.normal(0, .003, (1, 3)).astype(np.float32))
    print(k, f"{(time.time()-t0)*1000:.0f} ms", [(p["id"], p["label"], np.round(p["n"], 2).tolist(), round(p["d"], 2), round(p["hu"], 2), round(p["hv"], 2)) for p in it])
lab = sorted(p["label"] for p in pt.items); ids = sorted(p["id"] for p in pt.items)
assert lab == ["ceiling", "floor", "table", "wall"], lab
assert ids == [1, 2, 3, 4] or len(set(ids)) == 4 and max(ids) <= 6, ids
w = [p for p in pt.items if p["label"] == "wall"][0]; assert abs(w["n"][1]) < 1e-6, w["n"]
print("ok")
