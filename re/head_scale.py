#!/usr/bin/env python3
"""Tracking-scale check: the same static floor seen from different head heights (live13 frames 305-341, controllers
lying on the floor). Per frame: RANSAC floor plane from stereo (metric) -> floor height in STAGE = eye height minus
the metric distance. If SteamVR's head translation had a scale error s, floor_y would change by (s-1)*d(eye_y).
Also the controllers: their tracked height vs the stereo floor in each frame.
  head_scale.py captures/live13 305 341"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo
from tsdf import bin_raw10
rec, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
ls = LiveStereo(scale=0.35, max_depth=4.0)
recs = {json.loads(l)["i"]: json.loads(l) for l in open(os.path.join(rec, "frames.jsonl"))}
n = ls.size[0]; vv, uu = np.mgrid[0:n:2, 0:n:2]
rng = np.random.default_rng(0)
rows = []
for i in range(a, b + 1, step):
    if i not in recs or not recs[i]['left']['pose_valid']: continue
    r = recs[i]
    cL = bin_raw10(open(os.path.join(rec, f"{i:05d}_left.y10p"), "rb").read()); cR = bin_raw10(open(os.path.join(rec, f"{i:05d}_right.y10p"), "rb").read())
    Z, _, _ = ls.depth(cL, cR, far=4.0, half=True); Z = Z[::2, ::2]
    T = ls.world_from_rect(r["left"]["pose"]); eye = T[1, 3]
    m = Z > 0
    X = np.stack([(uu[m] - ls.P1[0, 2]) / ls.f * Z[m], (vv[m] - ls.P1[1, 2]) / ls.f * Z[m], Z[m]], 1)
    P = X @ T[:3, :3].T + T[:3, 3]
    cand = P[P[:, 1] < eye - 0.3]                       # below the eyes
    if len(cand) < 2000: continue
    hy, he = np.histogram(cand[:, 1], bins=np.arange(-0.5, eye - 0.25, 0.01))
    if hy.max() < 800: continue
    pk = np.flatnonzero(hy >= 0.25 * hy.max())[0]              # lowest strong height peak = floor (not table tops)
    pk = pk + np.argmax(hy[pk:pk + 4])
    y0 = he[pk] + 0.005
    fl = cand[np.abs(cand[:, 1] - y0) < 0.03]
    fy = np.median(fl[:, 1]); dist = np.median(np.linalg.norm(fl - T[:3, 3], axis=1))
    rows.append((i, eye, fy, dist, len(fl)))
    if len(sys.argv) <= 4: print(f"frame {i}: eye y {eye:.3f}  stereo floor y {fy:+.3f}  (floor {dist:.2f} m away, {len(fl)} px)")
R = np.array(rows)
A = np.c_[R[:, 1], np.ones(len(R))]; sl = np.linalg.lstsq(A, R[:, 2], rcond=None)[0][0]
print(f"\nfloor y vs eye height: slope {sl:+.4f}  (0 = consistent; s-1 = slope -> head translation scale {1 + sl:.4f})")
for lo, hi in ((0, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 2.2)):
    m = (R[:, 1] >= lo) & (R[:, 1] < hi)
    if m.sum(): print(f"eye {lo:.1f}-{hi:.1f} m: {m.sum():3d} frames  floor y median {np.median(R[m, 2]):+.3f}  (floor seen at {np.median(R[m, 3]):.2f} m)")
