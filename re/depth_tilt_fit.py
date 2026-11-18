#!/usr/bin/env python3
"""Drift-free estimate of a disparity offset d0 (Z = fB/(disp+d0)) from single frames.
A constant disparity offset maps planes to planes but TILTS them (a plane is linear in disparity space).
The floor seen in one frame must be level (gravity is known from tracking), so we fit each frame's
dominant floor plane by RANSAC in disparity space (inliers independent of d0) and find the d0 that makes
the floors horizontal. Also reports per-frame eye-to-floor heights for the scale check."""
import sys, numpy as np
z = np.load(sys.argv[1]); O, D, disp, fB, F, UV = z["O"], z["D"], z["disp"], float(z["fB"]), z["frame"], z["uv"].astype(float)
rng = np.random.default_rng(0)

def ransac_plane_disp(uv, d, iters=300, tol=0.25):
    best = None; A = np.c_[uv, np.ones(len(d))]
    for _ in range(iters):
        i = rng.choice(len(d), 3, replace=False)
        try: p = np.linalg.solve(A[i], d[i])
        except np.linalg.LinAlgError: continue
        inl = np.abs(A @ p - d) < tol
        if best is None or inl.sum() > best.sum(): best = inl
    p, *_ = np.linalg.lstsq(A[best], d[best], rcond=None)
    return np.abs(A @ p - d) < tol

planes = []
for f in range(F.max() + 1):
    s = np.flatnonzero(F == f)
    Dn = D[s] / np.linalg.norm(D[s], axis=1, keepdims=True)
    P = O[s] + D[s] * (fB / disp[s])[:, None]
    cand = s[(Dn[:, 1] < -0.35) & (P[:, 1] < O[s][0, 1] - 1.0)]      # looking down, > 1 m below the eyes
    if len(cand) < 3000: continue
    inl = ransac_plane_disp(UV[cand], disp[cand])
    idx = cand[inl]
    if len(idx) < 2000: continue
    planes.append(idx)

def tilt(idx, d0):
    P = O[idx] + D[idx] * (fB / np.maximum(disp[idx] + d0, 0.3))[:, None]
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); n = vt[2] * np.sign(vt[2][1])
    # signed tilt towards the viewing direction (horizontal component of the mean ray)
    v = D[idx].mean(0); v[1] = 0; v /= np.linalg.norm(v)
    return np.degrees(np.arcsin(np.clip(n @ v, -1, 1))), np.degrees(np.arccos(np.clip(n[1], -1, 1))), c[1], P

print(f"{len(planes)} frames with a floor plane")
for d0 in (-0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    t = np.array([tilt(i, d0)[:2] for i in planes])
    print(f"d0={d0:+.2f}: signed tilt median {np.median(t[:,0]):+.2f} deg  mean {t[:,0].mean():+.2f}, |tilt| median {np.median(t[:,1]):.2f}")

# ---- joint fit: d0 + small camera rotation error (rect frame) -------------------------------------
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
ls = LiveStereo(scale=0.35, max_depth=4.0); cx, cy, f = ls.P1[0, 2], ls.P1[1, 2], ls.f
data = []
for idx in planes:
    r = np.c_[(UV[idx, 0] - cx) / f, (UV[idx, 1] - cy) / f, np.ones(len(idx))]
    # world-from-rect rotation of this frame from D = Rw r (exact, all rays share Rw)
    Rw, _ = Rot.align_vectors(D[idx][:400], r[:400]); Rw = Rw.as_matrix()
    sub = rng.choice(len(idx), min(len(idx), 1500), replace=False)
    data.append((O[idx[0]], Rw, r[sub], disp[idx[sub]]))
def normals(x):
    d0, w = x[0], x[1:4]; Rc = Rot.from_rotvec(w).as_matrix(); out = []
    for o, Rw, r, d in data:
        P = (r * (fB / np.maximum(d + d0, 0.3))[:, None]) @ (Rw @ Rc).T
        c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); n = vt[2] * np.sign(vt[2][1]); out.append(n)
    return np.array(out)
res = lambda x: np.r_[normals(x)[:, [0, 2]].ravel()]
for x0 in ([0, 0, 0, 0], [0.5, 0, 0, 0]):
    sol = least_squares(res, x0, loss="soft_l1", f_scale=0.03)
    n = normals(sol.x); tl = np.degrees(np.arccos(np.clip(n[:, 1], -1, 1)))
    print(f"start {x0}: d0={sol.x[0]:+.3f} px, rot err (rect frame) {np.degrees(sol.x[1:]).round(2)} deg, "
          f"|tilt| median {np.median(tl):.2f} deg (was {np.median(np.degrees(np.arccos(normals([0,0,0,0])[:,1]))):.2f})")
for d0fix in (0.0,):
    s2 = least_squares(lambda w: res(np.r_[d0fix, w]), [0, 0, 0], loss="soft_l1", f_scale=0.03)
    n = normals(np.r_[d0fix, s2.x]); print(f"d0 fixed {d0fix}: rot {np.degrees(s2.x).round(2)} |tilt| median {np.median(np.degrees(np.arccos(n[:,1]))):.2f}")
