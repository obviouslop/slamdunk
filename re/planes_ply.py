#!/usr/bin/env python3
"""Run tools/planes.py on a saved PLY (e.g. captures/sessions/scan_*.ply) and print the tracked planes. usage: planes_ply.py FILE.ply [--n 3]"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from planes import PlaneTracker


def read_ply(fn):
    with open(fn, "rb") as fh:
        props = []; n = 0
        while True:
            l = fh.readline().decode().strip()
            if l.startswith("element vertex"): n = int(l.split()[-1])
            elif l.startswith("property"): props.append(l.split()[1:])
            elif l == "end_header": break
        tm = {"float": "<f4", "double": "<f8", "uchar": "u1"}
        d = np.frombuffer(fh.read(), np.dtype([(nm, tm[t]) for t, nm in props]), count=n)
    return np.stack([d["x"], d["y"], d["z"]], 1)


P = read_ply(sys.argv[1]); reps = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3
pt = PlaneTracker(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "scratch", "pcplanes_test.json"))
for k in range(reps):
    t0 = time.time(); it = pt.update(P)
print(f"{len(P)} pts, {(time.time()-t0)*1000:.0f} ms/update, extent {P.min(0).round(1)} .. {P.max(0).round(1)}")
for p in it: print("  ", p["id"], p["label"], np.round(p["n"], 2), f'd={p["d"]:.2f}', f'{2*p["hu"]:.1f}x{2*p["hv"]:.1f} m', p["npts"], "voxels")
