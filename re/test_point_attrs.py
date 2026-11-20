#!/usr/bin/env python3
"""Per-point acquisition time / view distance: a plane seen first from 2.0 m (t=10), then from 0.8 m (t=20) must end up
with d=0.8 and t=20, and the far view after the near one must not move the surface or the stamp."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from tsdf import TSDF
class LS:  # minimal stand-in for LiveStereo
    f = 200.0; baseline = 0.1; P1 = np.array([[200, 0, 50.0], [0, 200, 50.0]])
n = 100; ls = LS()
def frame(z_plane, bias=0.0):
    return np.full((n, n), z_plane + bias, np.float32)
def T(z):  # camera looks along +z; plane at world z=3
    M = np.eye(4); M[2, 3] = z; return M
ts = TSDF(voxel=0.02, max_depth=3.5, carve_depth=4.0)
rgb = np.full((n, n, 3), 100, np.uint8)
for k in range(6): ts.integrate(ls, frame(2.0, 0.03), rgb, T(1.0), t=10 + k * .1)       # far view, biased +3 cm
P, C, w, t, d = ts.extract(min_frames=3, attrs=True)
print("far :", len(P), "z", np.median(P[:, 2]).round(3), "d", np.median(d).round(2), "t", np.median(t).round(1))
for k in range(6): ts.integrate(ls, frame(0.8), rgb, T(2.2), t=20 + k * .1)             # near view, exact
P, C, w, t, d = ts.extract(min_frames=3, attrs=True)
m = (np.abs(P[:, 0]) < 0.15) & (np.abs(P[:, 1]) < 0.15)
print("near:", len(P), "z", np.median(P[m, 2]).round(3), "d", np.median(d[m]).round(2), "t", np.median(t[m]).round(1))
assert abs(np.median(d[m]) - 0.8) < 0.03 and np.median(t[m]) >= 20
z_near = np.median(P[m, 2])
for k in range(10): ts.integrate(ls, frame(2.0, 0.03), rgb, T(1.0), t=30 + k * .1)      # far view again
P, C, w, t, d = ts.extract(min_frames=3, attrs=True); m = (np.abs(P[:, 0]) < 0.15) & (np.abs(P[:, 1]) < 0.15)
print("far again:", "z", np.median(P[m, 2]).round(3), "d", np.median(d[m]).round(2), "t", np.median(t[m]).round(1))
assert abs(np.median(P[m, 2]) - 3.0) < 0.015 and np.median(d[m]) < 1.0 and np.median(t[m]) < 25
print("OK")
